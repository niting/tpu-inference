# SPDX-License-Identifier: Apache-2.0
"""Payload types for RL weight sync.

The API itself lives on `TPUWorker` (`init_weight_transfer_engine`,
`start_weight_update`, `update_weights`, `finish_weight_update`) -- the same
method names vLLM standardized on, because `collective_rpc` dispatches to
workers by name. This module holds only the data that crosses that boundary.

Two transports, distinguished by whether `weights` is set:

- Set: a colocated trainer handed over a live pytree. Only possible when the
  trainer shares a process and a JAX client with the sampler, which is how RL
  frameworks drive tpu-inference today.
- Unset: the trainer is a separate process and pushed straight into this
  worker's HBM over Raiden, so there is nothing to apply here. See
  docs/developer_guides/rl_weight_sync.md.
"""

from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Optional, Tuple

import jax
from jax.sharding import NamedSharding


@dataclass(frozen=True)
class ParamMeta:
    """Serializable description of one weight tensor.

    What a trainer needs in order to produce a compatible update: logical
    shape, dtype, and how the array is laid out across the sampler's mesh.
    Plain Python so it survives the RPC hop to an out-of-process orchestrator.

    This matters most for the Raiden push path, where the trainer must lay the
    bytes out to match the sampler exactly -- the sampler receives into
    pre-shaped buffers and does no mapping of its own.
    """

    shape: Tuple[int, ...]
    dtype: str
    # PartitionSpec entries as plain data, e.g. ("model", None). None when the
    # array is not backed by a NamedSharding.
    sharding_spec: Optional[Tuple[Any, ...]] = None
    # Mesh axis sizes the spec refers to, e.g. (("model", 8), ("data", 1)).
    mesh_shape: Optional[Tuple[Tuple[str, int], ...]] = None

    @classmethod
    def from_array(cls, array: jax.Array) -> "ParamMeta":
        sharding = getattr(array, "sharding", None)
        spec = None
        mesh_shape = None
        if isinstance(sharding, NamedSharding):
            spec = tuple(sharding.spec)
            mesh_shape = tuple(sharding.mesh.shape.items())
        return cls(shape=tuple(array.shape),
                   dtype=str(array.dtype),
                   sharding_spec=spec,
                   mesh_shape=mesh_shape)


@dataclass
class WeightUpdateRequest:
    """One chunk of a weight update.

    All fields are optional: an empty request is the Raiden push case, where
    the trainer has already written into this worker's buffers.

    Attributes:
        weights: Source state (an `nnx.State`) holding the new values.
            Colocated transport only.
        mappings: `{src_path: (tgt_path, sharding)}` over dot-joined flat state
            keys, `*` wildcards allowed for layer indices. Matches
            `transfer_state_with_mappings`.
        transpose_keys: `{leaf_name: permutation}` applied before the write.
        reshard_fn: Optional `(src, tgt) -> src` hook that reshards the source
            onto the sampler's mesh. When set, the per-parameter `shard_put` is
            skipped, matching `TPUModelRunner._sync_weights`.
    """

    weights: Any = None
    mappings: Optional[Dict[str, Tuple[str, Tuple[str, ...]]]] = None
    transpose_keys: Optional[Dict[str, Tuple[int, ...]]] = None
    reshard_fn: Optional[Callable[[Any, Any], Any]] = None

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str,
                                              Any]]) -> "WeightUpdateRequest":
        """Build from the untyped dict that arrives over `collective_rpc`."""
        if not payload:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                f"Unknown weight update field(s): {sorted(unknown)}. "
                f"Expected any of: {sorted(known)}.")
        return cls(**payload)
