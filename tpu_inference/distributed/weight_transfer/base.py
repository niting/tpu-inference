# SPDX-License-Identifier: Apache-2.0
"""JAX-native weight transfer engine contract for RL weight sync.

vLLM ships a weight-transfer subsystem (``vllm.distributed.weight_transfer``)
whose engine base class is torch-typed: the constructor takes a
``torch.device`` and a ``torch.nn.Module``, the concrete ``update_weights``
calls ``torch.accelerator.synchronize()``, and the receive path assumes
``model.load_weights([(name, tensor)])``. None of that is meaningful on the
JAX TPU path, and none of the shipped engines are reusable (they all call
``torch.cuda.current_stream()``). So we define a parallel, JAX-native engine
contract here instead of subclassing.

What we *do* keep identical is the worker-facing method names
(``init_weight_transfer_engine``, ``start_weight_update``, ``update_weights``,
``finish_weight_update``), because vLLM dispatches to workers by method name
through ``collective_rpc``. Keeping the names means everything above the
worker -- ``LLM``, ``AsyncLLM``, ``EngineCore``, the executors, and the
weight-update HTTP routes -- drives ``TPUWorker`` unmodified.

Scope: the flax_nnx (JAX) model path only. The torchax/vLLM-PyTorch path
stores weights as a flat ``dict[str, jax.Array]`` keyed by dotted torch
parameter names and is not handled here yet.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar

import jax
from jax.sharding import NamedSharding

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class ParamMeta:
    """Serializable description of one weight tensor.

    This is what a trainer needs in order to produce a compatible update: the
    logical shape, the dtype, and how the array is laid out across the
    sampler's mesh. Everything is plain Python so it survives the RPC hop to
    an out-of-process orchestrator.
    """

    shape: Tuple[int, ...]
    dtype: str
    # PartitionSpec entries as plain data, e.g. ("model", None). None when the
    # array is not backed by a NamedSharding (single-device / unsharded).
    sharding_spec: Optional[Tuple[Any, ...]] = None
    # Mesh axis sizes the spec refers to, e.g. {"model": 8, "data": 1}.
    mesh_shape: Optional[Tuple[Tuple[str, int], ...]] = None

    @classmethod
    def from_array(cls, array: jax.Array) -> "ParamMeta":
        sharding = getattr(array, "sharding", None)
        spec = None
        mesh_shape = None
        if isinstance(sharding, NamedSharding):
            spec = tuple(sharding.spec)
            mesh_shape = tuple(sharding.mesh.shape.items())
        return cls(
            shape=tuple(array.shape),
            dtype=str(array.dtype),
            sharding_spec=spec,
            mesh_shape=mesh_shape,
        )


@dataclass
class WeightTransferInitInfo:
    """Marker base for backend-specific init payloads."""


@dataclass
class WeightTransferUpdateInfo:
    """Marker base for backend-specific update payloads."""


TInitInfo = TypeVar("TInitInfo", bound=WeightTransferInitInfo)
TUpdateInfo = TypeVar("TUpdateInfo", bound=WeightTransferUpdateInfo)


class JaxWeightTransferEngine(ABC, Generic[TInitInfo, TUpdateInfo]):
    """Transport-agnostic receive side of an RL weight sync.

    Lifecycle, mirroring vLLM's four phases:

    1. ``init_transfer_engine`` -- once, before the training loop. Establishes
       whatever channel the backend needs.
    2. ``start_weight_update`` -- once per update. Engine-specific preparation.
       KV-cache teardown is *not* done here; the worker owns that, because it
       is transport-independent.
    3. ``update_weights`` -- one or more times, one chunk each.
    4. ``finish_weight_update`` -- once, post-processing.

    Subclasses set ``init_info_cls`` / ``update_info_cls`` so the untyped
    dicts arriving over ``collective_rpc`` can be validated into dataclasses.
    """

    init_info_cls: type
    update_info_cls: type

    def __init__(self, config: Any, vllm_config: Any, runner: Any) -> None:
        """
        Args:
            config: The ``WeightTransferConfig`` selecting this backend.
            vllm_config: Full vLLM config, for parallel/model settings.
            runner: The ``TPUModelRunner`` whose weights will be overwritten.
                Held rather than the weights themselves because
                ``runner.state`` / ``runner.state_leaves`` are rebound on
                every update.
        """
        self.config = config
        self.vllm_config = vllm_config
        self.runner = runner

    # --- payload validation -------------------------------------------------

    def parse_init_info(self, init_dict: Dict[str, Any]) -> TInitInfo:
        try:
            return self.init_info_cls(**init_dict)
        except TypeError as e:
            raise ValueError(
                f"Invalid init_info for {type(self).__name__}: {e}") from e

    def parse_update_info(self, update_dict: Dict[str, Any]) -> TUpdateInfo:
        try:
            return self.update_info_cls(**update_dict)
        except TypeError as e:
            raise ValueError(
                f"Invalid update_info for {type(self).__name__}: {e}") from e

    # --- lifecycle ----------------------------------------------------------

    @abstractmethod
    def init_transfer_engine(self, init_info: TInitInfo) -> None:
        """Establish the transport. Called once, before the training loop."""

    @abstractmethod
    def start_weight_update(self) -> None:
        """Prepare for a new update. Often a no-op."""

    @abstractmethod
    def receive_weights(self, update_info: TUpdateInfo) -> None:
        """Receive one chunk and write it into ``self.runner``'s state."""

    @abstractmethod
    def finish_weight_update(self) -> None:
        """Finalize the update. Often a no-op."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release transport resources."""

    def update_weights(self, update_info: Dict[str, Any]) -> None:
        """Receive one chunk, then block until the new weights are live.

        The barrier is the JAX analogue of vLLM's
        ``torch.accelerator.synchronize()``: transfers may be asynchronous, and
        the next forward pass must not race the write.
        """
        typed = self.parse_update_info(update_info)
        self.receive_weights(typed)
        jax.block_until_ready(self.runner.state_leaves)
