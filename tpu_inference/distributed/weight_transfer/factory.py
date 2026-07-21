# SPDX-License-Identifier: Apache-2.0
"""Registry for JAX weight transfer backends.

Mirrors ``vllm.distributed.weight_transfer.WeightTransferEngineFactory`` so
that a backend is selected by ``--weight-transfer-config '{"backend": ...}'``.
``WeightTransferConfig.backend`` is typed ``Literal[...] | str`` upstream, so
TPU-only backend names pass validation without patching vLLM.
"""

from typing import Any, Dict, Type

from tpu_inference.distributed.weight_transfer.base import \
    JaxWeightTransferEngine
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_REGISTRY: Dict[str, Type[JaxWeightTransferEngine]] = {}
_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    """Import backends that ship with tpu-inference.

    Deferred so that importing the factory does not drag in every backend's
    optional dependencies (e.g. tpu_raiden).
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from tpu_inference.distributed.weight_transfer.jax_local_engine import \
        JaxLocalWeightTransferEngine
    register_engine("jax_local", JaxLocalWeightTransferEngine)


def register_engine(name: str,
                    engine_cls: Type[JaxWeightTransferEngine]) -> None:
    """Register a backend under ``name``, overwriting any previous entry."""
    if not issubclass(engine_cls, JaxWeightTransferEngine):
        raise TypeError(
            f"{engine_cls!r} must subclass JaxWeightTransferEngine.")
    if name in _REGISTRY and _REGISTRY[name] is not engine_cls:
        logger.warning("Overriding weight transfer backend %r (%s -> %s).",
                       name, _REGISTRY[name].__name__, engine_cls.__name__)
    _REGISTRY[name] = engine_cls


def get_engine_cls(name: str) -> Type[JaxWeightTransferEngine]:
    _ensure_builtins()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown weight transfer backend {name!r}. "
            f"Registered backends: {sorted(_REGISTRY)}. Register a custom "
            "backend with "
            "tpu_inference.distributed.weight_transfer.register_engine().")
    return _REGISTRY[name]


def create_engine(vllm_config: Any, runner: Any) -> JaxWeightTransferEngine:
    """Build the engine named by ``vllm_config.weight_transfer_config``."""
    config = getattr(vllm_config, "weight_transfer_config", None)
    if config is None:
        raise ValueError(
            "create_engine called with no weight_transfer_config set.")
    engine_cls = get_engine_cls(config.backend)
    logger.info("Creating weight transfer engine %r (%s).", config.backend,
                engine_cls.__name__)
    return engine_cls(config=config, vllm_config=vllm_config, runner=runner)
