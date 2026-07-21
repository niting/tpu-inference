# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the RL weight-update API on TPUWorker.

Exercises the session state machine and payload handling against a fake model
runner; needs no TPU and no model.
"""

import pickle
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import pytest

from tpu_inference.distributed.weight_transfer import (ParamMeta,
                                                       WeightUpdateRequest)
from tpu_inference.worker.tpu_worker import TPUWorker


class _FakeRunner:
    """Stands in for TPUModelRunner."""

    def __init__(self):
        self.state_leaves = (jnp.zeros((2, 2)), )
        self.kv_events: List[str] = []
        self.sync_calls: List[Dict[str, Any]] = []
        self.fail_on_sync = False

    def delete_kv_cache(self):
        self.kv_events.append("delete")

    def reinitialize_kv_cache(self):
        self.kv_events.append("reinit")

    def _sync_weights(self, **kwargs):
        if self.fail_on_sync:
            raise RuntimeError("sync boom")
        self.sync_calls.append(kwargs)

    def get_weight_metadata(self):
        return {"w": ParamMeta(shape=(2, 2), dtype="float32")}


@pytest.fixture
def runner():
    return _FakeRunner()


@pytest.fixture
def worker(runner):
    """A TPUWorker with only the attributes the weight-update API touches.

    `TPUWorker.__init__` stands up devices and a model runner, so bypass it.
    """
    w = TPUWorker.__new__(TPUWorker)
    w.model_runner = runner
    return w


# --- happy path -------------------------------------------------------------


def test_full_update_cycle(worker, runner):
    worker.start_weight_update()
    worker.update_weights({"weights": "STATE"})
    worker.finish_weight_update()

    assert len(runner.sync_calls) == 1
    assert runner.sync_calls[0]["updated_weights"] == "STATE"
    assert worker._weight_update_active is False


def test_chunked_update_applies_every_chunk(worker, runner):
    worker.start_weight_update()
    worker.update_weights({"weights": "CHUNK0"})
    worker.update_weights({"weights": "CHUNK1"})
    worker.finish_weight_update()
    assert [c["updated_weights"]
            for c in runner.sync_calls] == ["CHUNK0", "CHUNK1"]


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
        worker.update_weights({"weights": "STATE"})
        worker.finish_weight_update()
    assert runner.kv_events == ["delete", "reinit"] * 3


def test_reshard_fn_is_forwarded(worker, runner):
    fn = lambda src, tgt: src
    worker.start_weight_update()
    worker.update_weights({"weights": "STATE", "reshard_fn": fn})
    assert runner.sync_calls[0]["reshard_fn"] is fn


def test_mappings_default_to_empty(worker, runner):
    worker.start_weight_update()
    worker.update_weights({"weights": "STATE"})
    assert runner.sync_calls[0]["mappings"] == {}


# --- error handling ---------------------------------------------------------


def test_update_without_start_raises(worker):
    with pytest.raises(RuntimeError, match="start_weight_update must be"):
        worker.update_weights({"weights": "STATE"})


def test_double_start_raises(worker):
    worker.start_weight_update()
    with pytest.raises(RuntimeError, match="already"):
        worker.start_weight_update()


def test_unknown_field_raises(worker):
    worker.start_weight_update()
    with pytest.raises(ValueError, match="Unknown weight update field"):
        worker.update_weights({"weights": "STATE", "nonsense": 1})


def test_failed_sync_ends_session_and_finish_restores_kv(worker, runner):
    runner.fail_on_sync = True
    worker.start_weight_update()
    with pytest.raises(RuntimeError, match="sync boom"):
        worker.update_weights({"weights": "STATE"})
    assert worker._weight_update_active is False
    # Further chunks are refused rather than silently applied.
    with pytest.raises(RuntimeError, match="start_weight_update must be"):
        worker.update_weights({"weights": "STATE"})
    worker.finish_weight_update()
    assert runner.kv_events == ["delete", "reinit"]


def test_finish_is_idempotent_for_kv_cache(worker, runner):
    worker.start_weight_update()
    worker.finish_weight_update()
    worker.finish_weight_update()
    assert runner.kv_events.count("reinit") == 1


# --- push transport ---------------------------------------------------------


def test_empty_update_is_a_noop(worker, runner):
    """Raiden push: the trainer already wrote into HBM, nothing to apply."""
    worker.start_weight_update()
    worker.update_weights({})
    worker.finish_weight_update()
    assert runner.sync_calls == []
    # The KV cache is still cycled -- that is the part the sampler owns.
    assert runner.kv_events == ["delete", "reinit"]


def test_empty_update_still_requires_a_session(worker):
    with pytest.raises(RuntimeError, match="start_weight_update must be"):
        worker.update_weights({})


def test_init_weight_transfer_engine_accepts_and_is_harmless(worker, runner):
    worker.init_weight_transfer_engine({})
    worker.start_weight_update()
    worker.update_weights({"weights": "STATE"})
    worker.finish_weight_update()
    assert len(runner.sync_calls) == 1


# --- request type -----------------------------------------------------------


def test_from_dict_accepts_empty():
    assert WeightUpdateRequest.from_dict(None) == WeightUpdateRequest()
    assert WeightUpdateRequest.from_dict({}) == WeightUpdateRequest()


# --- metadata ---------------------------------------------------------------


def test_param_meta_from_sharded_array():
    mesh = jax.sharding.Mesh(jax.devices()[:1], ("model", ))
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
    meta = ParamMeta.from_array(jnp.zeros((2, 3), dtype=jnp.float32))
    assert pickle.loads(pickle.dumps(meta)) == meta


def test_worker_get_weight_metadata_delegates(worker):
    assert "w" in worker.get_weight_metadata()
