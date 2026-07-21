# SPDX-License-Identifier: Apache-2.0
"""JAX-native weight transfer for RL weight sync on TPU.

See ``base.py`` for why this parallels rather than subclasses
``vllm.distributed.weight_transfer``.
"""

from tpu_inference.distributed.weight_transfer.base import (
    JaxWeightTransferEngine, ParamMeta, WeightTransferInitInfo,
    WeightTransferUpdateInfo)
from tpu_inference.distributed.weight_transfer.factory import (create_engine,
                                                               get_engine_cls,
                                                               register_engine)
from tpu_inference.distributed.weight_transfer.worker_mixin import \
    WeightTransferWorkerMixin

__all__ = [
    "JaxWeightTransferEngine",
    "ParamMeta",
    "WeightTransferInitInfo",
    "WeightTransferUpdateInfo",
    "create_engine",
    "get_engine_cls",
    "register_engine",
    "WeightTransferWorkerMixin",
]
