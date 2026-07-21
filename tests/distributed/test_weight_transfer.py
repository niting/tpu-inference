# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the JAX weight transfer contract and worker state machine.

These exercise the session bookkeeping and the registry against fakes; they
need no TPU and no model.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest

from tpu_inference.distributed.weight_transfer import factory
from tpu_inference.distributed.weight_transfer.base import (
    JaxWeightTransferEngine, ParamMeta, WeightTransferInitInfo,
    WeightTransferUpdateInfo)
from tpu_inference.distributed.weight_transfer.jax_local_engine import (
    JaxLocalUpdateInfo, JaxLocalWeightTransferEngine)
from tpu_inference.distributed.weight_transfer.worker_mixin import \
    WeightTransferWorkerMixin


@dataclass
class _FakeInitInfo(WeightTransferInitInfo):
    channel: str = "default"


@dataclass
class _FakeUpdateInfo(WeightTransferUpdateInfo):
    payload: Any = None


class _FakeEngine(JaxWeightTransferEngine):
    """Records lifecycle calls so tests can assert ordering."""

    init_info_cls = _FakeInitInfo
    update_info_cls = _FakeUpdateInfo

    def __init__(self, config=None, vllm_config=None, runner=None):
        super().__init__(config, vllm_config, runner)
        self.calls: List[str] = []
        self.received: List[Any] = []
        self.fail_on_start = False
        self.fail_on_receive = False

    def init_transfer_engine(self, init_info):
        self.calls.append(f"init:{init_info.channel}")

    def start_weight_update(self):
        if self.fail_on_start:
            raise RuntimeError("start boom")
        self.calls.append("start")

    def receive_weights(self, update_info):
        if self.fail_on_receive:
            raise RuntimeError("receive boom")
        self.calls.append("receive")
        self.received.append(update_info.payload)

    def finish_weight_update(self):
        self.calls.append("finish")

    def shutdown(self):
        self.calls.append("shutdown")


class _FakeRunner:
    """Stands in for TPUModelRunner."""

    def __init__(self):
        self.state_leaves = (jnp.zeros((2, 2)), )
        self.kv_events: List[str] = []
        self.sync_calls: List[Dict[str, Any]] = []

    def delete_kv_cache(self):
        self.kv_events.append("delete")

    def reinitialize_kv_cache(self):
        self.kv_events.append("reinit")

    def _sync_weights(self, **kwargs):
        self.sync_calls.append(kwargs)

    def get_weight_metadata(self):
        return {"w": ParamMeta(shape=(2, 2), dtype="float32")}


class _FakeWorker(WeightTransferWorkerMixin):

    def __init__(self, engine: Optional[JaxWeightTransferEngine],
                 runner: _FakeRunner):
        self.vllm_config = MagicMock()
        self.model_runner = runner
        self.weight_transfer_engine = engine


@pytest.fixture
def runner():
    return _FakeRunner()


@pytest.fixture
def engine(runner):
    return _FakeEngine(runner=runner)


@pytest.fixture
def worker(engine, runner):
    return _FakeWorker(engine, runner)


# --- happy path -------------------------------------------------------------


def test_full_update_cycle_calls_engine_in_order(worker, engine):
    worker.init_weight_transfer_engine({"channel": "nccl-ish"})
    worker.start_weight_update()
    worker.update_weights({"payload": "chunk0"})
    worker.update_weights({"payload": "chunk1"})
    worker.finish_weight_update()

    assert engine.calls == [
        "init:nccl-ish", "start", "receive", "receive", "finish"
    ]
    assert engine.received == ["chunk0", "chunk1"]


def test_kv_cache_is_freed_and_restored(worker, runner):
    worker.start_weight_update()
    assert runner.kv_events == ["delete"]
    worker.finish_weight_update()
    assert runner.kv_events == ["delete", "reinit"]


def test_free_kv_cache_can_be_disabled(worker, runner):
    worker.start_weight_update(free_kv_cache=False)
    worker.finish_weight_update()
    assert runner.kv_events == []


def test_multiple_sequential_sessions(worker, runner):
    for _ in range(3):
        worker.start_weight_update()
        worker.update_weights({"payload": None})
        worker.finish_weight_update()
    assert runner.kv_events == ["delete", "reinit"] * 3
    assert worker._weight_update_active is False


# --- error handling ---------------------------------------------------------


def test_update_without_start_raises(worker):
    with pytest.raises(RuntimeError, match="start_weight_update must be"):
        worker.update_weights({"payload": None})


def test_double_start_raises(worker):
    worker.start_weight_update()
    with pytest.raises(RuntimeError, match="already"):
        worker.start_weight_update()


def test_unconfigured_worker_raises_helpful_error(runner):
    worker = _FakeWorker(engine=None, runner=runner)
    with pytest.raises(RuntimeError, match="not configured"):
        worker.start_weight_update()


def test_failed_start_restores_kv_cache(worker, engine, runner):
    engine.fail_on_start = True
    with pytest.raises(RuntimeError, match="start boom"):
        worker.start_weight_update()
    # HBM must be back to a serving state, and no session left dangling.
    assert runner.kv_events == ["delete", "reinit"]
    assert worker._weight_update_active is False


def test_failed_receive_ends_session_and_finish_restores_kv(
        worker, engine, runner):
    engine.fail_on_receive = True
    worker.start_weight_update()
    with pytest.raises(RuntimeError, match="receive boom"):
        worker.update_weights({"payload": None})
    assert worker._weight_update_active is False
    # Further chunks are refused rather than silently applied.
    with pytest.raises(RuntimeError, match="start_weight_update must be"):
        worker.update_weights({"payload": None})
    worker.finish_weight_update()
    assert runner.kv_events == ["delete", "reinit"]


def test_finish_is_idempotent_for_kv_cache(worker, runner):
    worker.start_weight_update()
    worker.finish_weight_update()
    worker.finish_weight_update()
    # The cache is reallocated exactly once.
    assert runner.kv_events.count("reinit") == 1


def test_invalid_init_info_raises_value_error(worker):
    with pytest.raises(ValueError, match="Invalid init_info"):
        worker.init_weight_transfer_engine({"not_a_field": 1})


def test_invalid_update_info_raises_value_error(worker):
    worker.start_weight_update()
    with pytest.raises(ValueError, match="Invalid update_info"):
        worker.update_weights({"not_a_field": 1})


# --- registry ---------------------------------------------------------------


def test_builtin_backend_is_registered():
    assert factory.get_engine_cls("jax_local") is JaxLocalWeightTransferEngine


def test_unknown_backend_lists_registered_names():
    with pytest.raises(ValueError, match="jax_local"):
        factory.get_engine_cls("does-not-exist")


def test_register_and_create_custom_backend(runner):
    factory.register_engine("test_fake", _FakeEngine)
    vllm_config = MagicMock()
    vllm_config.weight_transfer_config.backend = "test_fake"
    created = factory.create_engine(vllm_config=vllm_config, runner=runner)
    assert isinstance(created, _FakeEngine)
    assert created.runner is runner


def test_register_rejects_non_engine():
    with pytest.raises(TypeError):
        factory.register_engine("bad", dict)


def test_create_engine_without_config_raises(runner):
    vllm_config = MagicMock(spec=[])  # no weight_transfer_config attribute
    with pytest.raises(ValueError, match="no weight_transfer_config"):
        factory.create_engine(vllm_config=vllm_config, runner=runner)


def test_init_weight_transfer_is_noop_when_unconfigured(runner):
    worker = _FakeWorker(engine=None, runner=runner)
    worker.vllm_config = MagicMock(spec=[])
    worker.init_weight_transfer()
    assert worker.weight_transfer_engine is None


# --- colocated backend ------------------------------------------------------


def test_jax_local_engine_forwards_to_sync_weights(runner):
    engine = JaxLocalWeightTransferEngine(config=None,
                                          vllm_config=None,
                                          runner=runner)
    engine.init_transfer_engine(
        engine.parse_init_info({
            "mappings": {
                "a": ("b", ("model", ))
            },
            "transpose_keys": {
                "kernel": (1, 0)
            },
        }))
    engine.receive_weights(JaxLocalUpdateInfo(weights="STATE"))

    assert len(runner.sync_calls) == 1
    call = runner.sync_calls[0]
    assert call["updated_weights"] == "STATE"
    assert call["mappings"] == {"a": ("b", ("model", ))}
    assert call["transpose_keys"] == {"kernel": (1, 0)}
    assert call["reshard_fn"] is None


def test_jax_local_per_update_overrides_win(runner):
    engine = JaxLocalWeightTransferEngine(config=None,
                                          vllm_config=None,
                                          runner=runner)
    engine.init_transfer_engine(
        engine.parse_init_info({"mappings": {
            "a": ("b", ())
        }}))
    engine.receive_weights(
        JaxLocalUpdateInfo(weights="STATE", mappings={"c": ("d", ())}))
    assert runner.sync_calls[0]["mappings"] == {"c": ("d", ())}


def test_update_weights_blocks_until_ready(runner):
    """The base update_weights must await the write, like torch's sync."""
    engine = JaxLocalWeightTransferEngine(config=None,
                                          vllm_config=None,
                                          runner=runner)
    engine.update_weights({"weights": "STATE"})
    assert len(runner.sync_calls) == 1


# --- metadata ---------------------------------------------------------------


def test_param_meta_from_sharded_array():
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices[:1], ("model", ))
    sharding = jax.sharding.NamedSharding(mesh,
                                          jax.sharding.PartitionSpec("model"))
    array = jax.device_put(jnp.zeros((4, 8), dtype=jnp.bfloat16), sharding)

    meta = ParamMeta.from_array(array)
    assert meta.shape == (4, 8)
    assert meta.dtype == "bfloat16"
    assert meta.sharding_spec == ("model", )
    assert meta.mesh_shape == (("model", 1), )


def test_param_meta_is_plain_data():
    """Metadata must survive an RPC hop to an out-of-process orchestrator."""
    import pickle

    meta = ParamMeta.from_array(jnp.zeros((2, 3), dtype=jnp.float32))
    assert pickle.loads(pickle.dumps(meta)) == meta


def test_worker_get_weight_metadata_delegates(worker):
    assert "w" in worker.get_weight_metadata()
