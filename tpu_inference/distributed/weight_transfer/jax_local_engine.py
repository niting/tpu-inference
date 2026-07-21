# SPDX-License-Identifier: Apache-2.0
"""Colocated (in-process) weight transfer backend.

This is the reference backend and the one that matches how RL frameworks
drive tpu-inference today: the trainer and the sampler live in the same
process and share a JAX client, so "transport" is just handing over a pytree
of ``jax.Array`` and letting JAX reshard it.

It exists for two reasons:

1. It gives the vLLM-native weight-update API (``start_weight_update`` /
   ``update_weights`` / ``finish_weight_update``) a working implementation on
   TPU today, so callers can migrate off the bespoke
   ``collective_rpc("sync_weights", ...)`` entry point.
2. It is the control for the networked backend: same worker-side lifecycle,
   same KV-cache handling, only the transport differs.

Because the payload carries live ``jax.Array`` objects (and optionally a
callable), this backend only works when the caller shares a process with the
worker -- i.e. an in-process or uniproc executor. A cross-process backend must
serialize, which is what the Raiden backend is for.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from tpu_inference.distributed.weight_transfer.base import (
    JaxWeightTransferEngine, WeightTransferInitInfo, WeightTransferUpdateInfo)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@dataclass
class JaxLocalInitInfo(WeightTransferInitInfo):
    """Defaults applied to every subsequent update.

    Trainer-to-sampler parameter name mappings rarely change between updates,
    so they can be supplied once here instead of on every chunk.

    Attributes:
        mappings: ``{src_path: (tgt_path, sharding)}``, where paths are
            dot-joined flat state keys and may contain ``*`` wildcards for
            layer indices. Matches ``transfer_state_with_mappings``.
        transpose_keys: ``{leaf_name: permutation}`` applied before the write.
    """

    mappings: Dict[str, Tuple[str, Tuple[str,
                                         ...]]] = field(default_factory=dict)
    transpose_keys: Dict[str, Tuple[int, ...]] = field(default_factory=dict)


@dataclass
class JaxLocalUpdateInfo(WeightTransferUpdateInfo):
    """One chunk of weights.

    Attributes:
        weights: Source state (an ``nnx.State``) holding the new values.
        mappings: Overrides the init-time mappings when set.
        transpose_keys: Overrides the init-time transpose keys when set.
        reshard_fn: Optional ``(src, tgt) -> src`` hook that reshards the
            source onto the sampler's mesh before the write. When supplied,
            the per-parameter ``shard_put`` is skipped, matching
            ``TPUModelRunner._sync_weights``.
    """

    weights: Any
    mappings: Optional[Dict[str, Tuple[str, Tuple[str, ...]]]] = None
    transpose_keys: Optional[Dict[str, Tuple[int, ...]]] = None
    reshard_fn: Optional[Callable[[Any, Any], Any]] = None


class JaxLocalWeightTransferEngine(JaxWeightTransferEngine[JaxLocalInitInfo,
                                                           JaxLocalUpdateInfo]
                                   ):
    """Applies weights handed over in-process."""

    init_info_cls = JaxLocalInitInfo
    update_info_cls = JaxLocalUpdateInfo

    def __init__(self, config: Any, vllm_config: Any, runner: Any) -> None:
        super().__init__(config, vllm_config, runner)
        self._mappings: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
        self._transpose_keys: Dict[str, Tuple[int, ...]] = {}

    def init_transfer_engine(self, init_info: JaxLocalInitInfo) -> None:
        self._mappings = dict(init_info.mappings)
        self._transpose_keys = dict(init_info.transpose_keys)
        logger.info(
            "Colocated weight transfer initialized with %d mapping(s) and "
            "%d transpose key(s).", len(self._mappings),
            len(self._transpose_keys))

    def start_weight_update(self) -> None:
        """No-op: there is no channel to open and no staging buffer."""

    def receive_weights(self, update_info: JaxLocalUpdateInfo) -> None:
        mappings = (self._mappings
                    if update_info.mappings is None else update_info.mappings)
        transpose_keys = (self._transpose_keys if update_info.transpose_keys
                          is None else update_info.transpose_keys)
        self.runner._sync_weights(
            updated_weights=update_info.weights,
            mappings=mappings,
            transpose_keys=transpose_keys,
            reshard_fn=update_info.reshard_fn,
        )

    def finish_weight_update(self) -> None:
        """No-op: weights are written in place by ``receive_weights``."""

    def shutdown(self) -> None:
        """No-op: nothing was allocated."""
