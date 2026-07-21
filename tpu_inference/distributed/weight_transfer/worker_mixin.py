# SPDX-License-Identifier: Apache-2.0
"""Worker-side weight-update session state machine.

vLLM puts these four methods on ``vllm.v1.worker.gpu_worker.Worker``, not on
``WorkerBase``. ``TPUWorker`` subclasses ``WorkerBase``, so it inherits none of
them. Since ``collective_rpc`` dispatches to workers *by method name*, all we
need is duck-typed methods with matching names -- which is what this mixin
provides.

It is a separate mixin rather than inline methods on ``TPUWorker`` so the
session bookkeeping can be tested against a fake runner, without standing up
a TPU worker.

Requires the host class to provide ``self.vllm_config`` and
``self.model_runner``.
"""

from typing import Any, Dict, Optional

from tpu_inference.distributed.weight_transfer import factory
from tpu_inference.distributed.weight_transfer.base import \
    JaxWeightTransferEngine
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class WeightTransferWorkerMixin:
    """Implements vLLM's worker-side weight update contract for TPU."""

    # Declared here so the attributes exist even when the host class never
    # calls init_weight_transfer() (i.e. weight transfer is not configured).
    weight_transfer_engine: Optional[JaxWeightTransferEngine] = None
    _weight_update_active: bool = False
    _kv_cache_freed: bool = False

    def init_weight_transfer(self) -> None:
        """Build the configured engine. Call after the model is loaded.

        No-op when ``weight_transfer_config`` is unset, so a normal serving
        run pays nothing.
        """
        config = getattr(self.vllm_config, "weight_transfer_config", None)
        if config is None:
            return
        self.weight_transfer_engine = factory.create_engine(
            vllm_config=self.vllm_config, runner=self.model_runner)

    def _check_weight_transfer_engine(self) -> JaxWeightTransferEngine:
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer is not configured. Start the engine with "
                "weight_transfer_config set, e.g. "
                "--weight-transfer-config '{\"backend\": \"jax_local\"}'.")
        return self.weight_transfer_engine

    def init_weight_transfer_engine(self, init_info: Dict[str, Any]) -> None:
        """Establish the transport. Called once, before the training loop."""
        engine = self._check_weight_transfer_engine()
        engine.init_transfer_engine(engine.parse_init_info(init_info))

    def start_weight_update(self, free_kv_cache: bool = True) -> None:
        """Open a weight update session.

        Args:
            free_kv_cache: Drop the KV cache before the transfer. On by
                default because receiving a full set of weights roughly
                doubles peak HBM, and because serving decode from KV computed
                under the old weights is incorrect. The paired
                ``finish_weight_update`` reallocates it. Pass False only when
                the caller has already freed the cache itself.

        Note: the prefix cache lives on the scheduler, not the worker, so
        callers must additionally call ``reset_prefix_cache()`` on the engine.
        """
        engine = self._check_weight_transfer_engine()

        if self._weight_update_active:
            raise RuntimeError(
                "start_weight_update called while an update is already "
                "active. Call finish_weight_update first.")

        if free_kv_cache:
            self.model_runner.delete_kv_cache()
            self._kv_cache_freed = True

        try:
            engine.start_weight_update()
        except BaseException:
            # Put HBM back the way we found it so the worker stays usable.
            if self._kv_cache_freed:
                self.model_runner.reinitialize_kv_cache()
                self._kv_cache_freed = False
            raise

        self._weight_update_active = True

    def update_weights(self, update_info: Dict[str, Any]) -> None:
        """Receive one chunk of weights.

        May be called multiple times per session for chunked transfers.
        """
        engine = self._check_weight_transfer_engine()

        if not self._weight_update_active:
            raise RuntimeError(
                "start_weight_update must be called before update_weights.")

        try:
            engine.update_weights(update_info)
        except BaseException:
            # The model is now in an unknown state -- some chunks may have
            # landed. End the session so the caller cannot keep streaming into
            # it, and let finish_weight_update restore the KV cache.
            self._weight_update_active = False
            raise

    def finish_weight_update(self) -> None:
        """Close the session and restore serving state."""
        engine = self._check_weight_transfer_engine()

        if not self._weight_update_active and not self._kv_cache_freed:
            logger.warning(
                "finish_weight_update called with no active session; "
                "finalizing anyway.")

        # Deliberately tolerant rather than raising when no session is active:
        # a failed update_weights ends the session but still leaves the KV
        # cache freed, and this is the call that puts it back.
        try:
            engine.finish_weight_update()
        finally:
            self._weight_update_active = False
            if self._kv_cache_freed:
                self.model_runner.reinitialize_kv_cache()
                self._kv_cache_freed = False

    def get_weight_metadata(self) -> Dict[str, Any]:
        """Describe the sampler's weights so a trainer can target them."""
        return self.model_runner.get_weight_metadata()

    def shutdown_weight_transfer(self) -> None:
        if self.weight_transfer_engine is not None:
            self.weight_transfer_engine.shutdown()
            self.weight_transfer_engine = None
