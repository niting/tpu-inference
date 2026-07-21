# RL Weight Sync on TPU

How trainer weights get into a running tpu-inference sampler, what Phase 1
shipped, and what Phases 2 and 3 still have to build.

Audience: people wiring tpu-inference into an RL stack (trainer + orchestrator
+ sampler), and people implementing the Raiden transport.

---

## 1. Background

In online RL the sampler's weights must be refreshed from the trainer every
step (or every few steps). There are two deployment shapes:

**Colocated.** Trainer and sampler share a process and a JAX client. "Transfer"
is handing over a pytree of `jax.Array`; JAX reshards it. This is how tunix
drives tpu-inference today.

**Disaggregated.** Trainer and sampler are separate processes, usually on
separate TPU slices with different meshes. Weights must cross a network. This
is what [TPU Raiden](https://github.com/google/tpu-raiden) exists for, and it
is the case Phases 2 and 3 address.

vLLM standardized the sampler-side API for both in its
[native weight transfer subsystem](https://docs.vllm.ai/en/latest/training/weight_transfer/):
four phases — `init_weight_transfer_engine`, `start_weight_update`,
`update_weights`, `finish_weight_update` — over a pluggable transport.

---

## 2. Phase 1 (shipped)

### What was in the tree before

- `TPUWorker.sync_weights(...)` → `TPUModelRunner._sync_weights(...)`: a
  bespoke entry point reached via `collective_rpc("sync_weights", ...)`. It
  does name mapping, transposition, and `shard_put` into `runner.state`. It
  has no callers inside this repo; tunix calls it from outside.
- `delete_kv_cache` / `reinitialize_kv_cache` on worker, runner and
  `kv_cache_manager` — already exercised end to end by
  `tests/e2e/test_rl_integration.py`.
- Two Raiden **KV-cache** connectors (`distributed/tpu_raiden_connector.py`,
  `offload/raiden_offload_connector.py`). No weight-sync code at all.
- No reference to vLLM's `weight_transfer` subsystem anywhere.

### What Phase 1 added

`tpu_inference/distributed/weight_transfer/`:

| File | Contents |
|---|---|
| `base.py` | `JaxWeightTransferEngine` (the JAX-native engine contract), `ParamMeta`, init/update payload bases |
| `factory.py` | Backend registry: `register_engine`, `get_engine_cls`, `create_engine` |
| `worker_mixin.py` | `WeightTransferWorkerMixin` — the worker-side session state machine |
| `jax_local_engine.py` | `jax_local`: colocated reference backend |

Plus `TPUModelRunner.get_weight_metadata()` and `TPUWorker` now mixing in
`WeightTransferWorkerMixin`.

### Two design decisions worth knowing

**We do not subclass vLLM's `WeightTransferEngine`.** Its base is torch-bound
in ways that buy nothing on TPU: the constructor takes `torch.device` and
`torch.nn.Module`, the concrete `update_weights` calls
`torch.accelerator.synchronize()`, `WeightSource.__iter__` yields
`torch.Tensor`, and the receive path assumes
`model.load_weights([(name, tensor)])`. None of the shipped engines are
reusable either — NCCL, IPC and sparse-NCCL all call
`torch.cuda.current_stream()`. So `base.py` defines a parallel contract whose
`update_weights` ends in `jax.block_until_ready(runner.state_leaves)`, the
JAX analogue of the torch sync.

**We do keep vLLM's worker method names.** vLLM puts the four methods on
`vllm.v1.worker.gpu_worker.Worker`, not `WorkerBase`, and `TPUWorker`
subclasses `WorkerBase` — so it inherits none of them. But `collective_rpc`
dispatches **by method name**, so duck-typed methods are enough. The
consequence is that everything above the worker works unmodified: `LLM`,
`AsyncLLM`, `EngineCore`, all three TPU executors, and the weight-update HTTP
routes. `WeightTransferConfig.backend` is typed `Literal[...] | str` upstream,
so `backend="jax_local"` validates without patching vLLM.

Pause/resume for async RL also needs nothing from us: it is implemented purely
at the scheduler level upstream.

### Who owns the KV cache

The **worker**, not the engine. Receiving a full set of weights roughly
doubles peak HBM, and decoding from KV computed under old weights is wrong —
both are true regardless of transport, so `start_weight_update` frees the
cache and `finish_weight_update` reallocates it. Pass `free_kv_cache=False`
to opt out.

The prefix cache lives on the scheduler, so callers must still call
`reset_prefix_cache()` on the engine themselves.

### Usage

```python
llm = LLM(model=..., weight_transfer_config=WeightTransferConfig(backend="jax_local"))

llm.init_weight_transfer_engine(dict(mappings=..., transpose_keys=...))

# per RL step
llm.reset_prefix_cache()
llm.start_weight_update()
llm.update_weights(dict(weights=new_state))   # repeatable for chunked transfer
llm.finish_weight_update()
```

### Limitations

- **flax_nnx path only.** `get_weight_metadata()` raises `NotImplementedError`
  on the torchax path, which stores weights as a flat `dict[str, jax.Array]`
  keyed by dotted torch parameter names and needs its own key convention.
- **`jax_local` is in-process only.** Its payload carries live `jax.Array`
  objects and an optional callable, so it requires an in-process or uniproc
  executor.
- **No chunk accounting.** Nothing verifies that the chunks received in a
  session cover every parameter. See Phase 3.
- **No draft-model support.** `start_draft_weight_update` is not implemented;
  calling it raises `AttributeError`.

---

## 3. What Raiden actually offers today

Read this before designing against the RL orchestration proposal — several of
the APIs that proposal names do not exist.

### Exists, shipped in the wheel, tested

`tpu_raiden.api.jax.weight_synchronizer.WeightSynchronizer` — nanobind over
C++:

```python
WeightSynchronizer(jax_arrays: List[jax.Array], local_port=None, parallelism=1,
                   unsafe_skip_buffer_lock=False, listener_port=None, bind_ip=None)
  .pull_weights(source: str)          # "host:port"; network pull + H2D
  .d2h() / .h2d()                     # HBM <-> C++ host staging buffer
  .pull_weights_chunk(source, src_shard_idx, src_offset_bytes,
                      dst_shard_idx, dst_offset_bytes, size_bytes)
  .h2d_chunk(shard_idx, host_offset_bytes, device_offset_bytes, size_bytes)
  .get_host_buffer(layer_idx, shard_idx)   # zero-copy numpy view
  # props: local_port, listener_port, num_layers, num_shards, slice_byte_size
```

### Exists but unpackaged / untested

`tpu_raiden.rpc.raiden_controller` — `RaidenController`,
`RaidenControllerServer`, `RaidenControllerClientFacade`, `RaidenId`,
`register_work_unit`, `start_transfer`. **Not in the wheel's Bazel deps**, zero
test coverage for the server/facade, zero call sites anywhere in tpu-raiden.
Do not put it on a critical path yet.

### Does not exist

`coordinate_transfer`, `get_transfer_status`, `get_id()`, `get_layout()`,
and a Python `push_weights()` on the JAX side.

### Sharp edges

- **One `slice_byte_size` for all layers.** It is derived from
  `jax_arrays[0]` and applied to every entry; mismatches throw. Real model
  weights are heterogeneous — see 4.1.
- **No `push_weights()` in the JAX Python API.** You push by sending a
  4-byte-length-prefixed `ControlRequest` protobuf to the C++ listener over
  IPv6 loopback. Only the Torch API has a Python `push_weights`.
- **`pull_weights()` from an unstaged source silently delivers zeros.** It
  reads the source's *host staging buffer*, so the source must `d2h()` first.
  No error is raised.
- **Resharding is a prototype.** `resharding_engine.reshard_matrix` is rank-2,
  float32, single-array, `parallelism=1`. The underlying pieces
  (`resharding_planner.make_resharding_plan`, `pull_weights_chunk`,
  `h2d_chunk`) are general, but the planner carries a live TODO about TPU
  memory-tiling alignment.
- **Bare `host:port` endpoints are dangerous.** The KV connector already
  learned this: with more than one NUMA sub-manager, passing a bare endpoint
  hits a broadcast overload and silently corrupts roughly half the payload.
  Weight sync must publish the full per-shard endpoint list the same way
  (`get_local_endpoints()`).

---

## 4. Phase 2 — the Raiden transport

Goal: a `raiden` backend that moves weights from an out-of-process trainer
into a running sampler, with the sampler-side API from Phase 1 unchanged.

### 4.1 Problem: heterogeneous parameter shapes

`WeightSynchronizer` wants a list of arrays of identical per-shard byte size.
A transformer's parameters are nothing like that. Three options:

| Option | Cost | Verdict |
|---|---|---|
| **A. One synchronizer per (shape, dtype, sharding) bucket** | N ports, N connections, N handshakes per sync; bookkeeping is simple | Works today. Poor at scale — a 70B model has dozens of distinct shapes. |
| **B. Pack into a few uniform buffers** | One `jnp.concatenate` of flattened params into fixed-size slabs; receiver slices views back out. Costs a full extra HBM copy on both sides unless the model is loaded into the slab to begin with. | This is what vLLM's NCCL engine means by `packed=True`. Best throughput, most work. |
| **C. Extend Raiden to per-layer slice sizes** | Removes the constraint at the source. | **Preferred.** The constraint is an implementation detail of `WeightSynchronizerBase`, not a protocol limit. |

**Recommendation:** file C against tpu-raiden and implement A as the
interim backend, since A is a pure tpu-inference change and its bookkeeping is
reusable under C. Reach for B only if profiling shows per-connection overhead
dominating.

### 4.2 Problem: buffer lifetime vs. donation

This is the subtlest correctness risk in the design.

`WeightSynchronizer` takes a PJRT `BufferHoldAndAlias` on each array **at
construction**. Meanwhile tpu-inference:

- builds the jitted model with `donate_argnums=(0,)` in `create_jit_model`,
  so weight buffers are donation candidates; and
- **rebinds** `runner.state` and `runner.state_leaves` on every
  `_sync_weights` call, so the arrays a synchronizer holds are not the arrays
  the model will next execute against.

Any of three resolutions, in order of preference:

1. **Write in place.** Make the receive path H2D directly into the held
   buffers and *not* rebind `runner.state`. This is the only option that makes
   a long-lived synchronizer correct, and it is what `h2d_chunk` is for. It
   requires `_sync_weights`' "build a new state, then swap" structure to
   change for this backend.
2. **Reconstruct per sync.** Build a fresh `WeightSynchronizer` each update.
   Correct and simple; pays setup + `BufferHoldAndAlias` cost every step, and
   re-opens ports every step.
3. **Pin the weight buffers.** Disable donation for the weight argument. Costs
   peak HBM during forward.

Validate whichever is chosen with a test that syncs twice and asserts the
second sync actually lands — a stale hold fails silently, not loudly.

### 4.3 Sharding: pick the cheap answer first

Trainer mesh ≠ sampler mesh in general. Three strategies:

1. **Constrain the meshes equal.** Plain push/pull works with shipping Raiden.
   Start here.
2. **Reshard on the trainer before transfer.** The orchestration proposal's
   `convert(weights, dst_sharding_pytree)`. Correct and simple; costs a full
   extra HBM copy on the trainer. `get_weight_metadata()` from Phase 1 is
   exactly the input this needs.
3. **Raiden-native resharding** via `shard_push_schedules` /
   `pull_weights_chunk`. Fastest, but blocked on the planner becoming
   rank- and dtype-general.

Ship 1 or 2. Treat 3 as Phase 3.

### 4.4 Rendezvous: keep the controller out of vLLM

The orchestration proposal offers two placements for the Raiden controller.
**Choose Option 1 — controller on the `SamplerWorker`, outside vLLM.**

- `RaidenControllerServer` is unpackaged and untested, so we want to be able
  to swap it for plain orchestrator-mediated rendezvous without touching
  engine code.
- It avoids threading controller IDs down through `EngineCore` → executor →
  every `TPUWorker`.
- The orchestrator already holds handles to both worker sets, so it is the
  natural place to exchange endpoints.

The **only** thing that must live inside the worker process is the
`WeightSynchronizer`, because only that process holds the PJRT buffers.

Endpoint publication should reuse the KV connector's existing pattern:
`get_host_ip()`, a dedicated port (add `TPU_WEIGHT_TRANSFER_PORT` alongside
`TPU_KV_TRANSFER_PORT`), and the full per-shard endpoint list.

### 4.5 Proposed shape

```python
@dataclass
class RaidenInitInfo(WeightTransferInitInfo):
    trainer_endpoints: list[dict]   # [{"endpoint": "ip:port", "shards": [...]}]
    parallelism: int = 1
    mode: str = "pull"              # "pull" | "push"

@dataclass
class RaidenUpdateInfo(WeightTransferUpdateInfo):
    param_names: list[str]          # which params this chunk covers
    bucket_id: int                  # which synchronizer group

class RaidenWeightTransferEngine(JaxWeightTransferEngine):
    init_info_cls, update_info_cls = RaidenInitInfo, RaidenUpdateInfo
```

- `init_transfer_engine`: bucket `runner.state` (4.1), construct one
  `WeightSynchronizer` per bucket, publish endpoints.
- `start_weight_update`: no-op (the worker already freed the KV cache).
- `receive_weights`: `pull_weights()` per bucket, or wait for a push.
- `finish_weight_update`: rebind or verify `runner.state_leaves` (4.2).
- `shutdown`: drop synchronizers, close ports.

Everything is plain data over `collective_rpc`, so this backend works
cross-process — unlike `jax_local`.

### 4.6 Phase 2 checklist

- [ ] tpu-raiden: heterogeneous per-layer slice sizes (4.1 option C)
- [ ] `TPU_WEIGHT_TRANSFER_PORT` + per-shard endpoint publication
- [ ] Bucketing of `runner.state` into synchronizer groups
- [ ] Resolve buffer lifetime (4.2); regression test for a **second** sync
- [ ] `RaidenWeightTransferEngine` + registration as `"raiden"`
- [ ] Multi-host test: 2 hosts, equal meshes, bit-exact weight parity
- [ ] Guard against the bare-`host:port` broadcast hazard

---

## 5. Phase 3 — resharding, robustness, scale

### 5.1 Cross-mesh resharding

Needs, from tpu-raiden:

- N-D resharding planner (today: rank-2 only) — `compute_nd_shard_slices`
  already generalizes the slice math, so the gap is in `resharding_engine`.
- dtype generality (today: float32 byte math is hardcoded).
- TPU memory-tiling alignment (live TODO in `resharding_planner`) — padding
  sub-blocks to multiples of 8/32/128 bytes. Until this lands, resharding is
  only safe when the sharded dimensions are large and evenly divisible.

From tpu-inference: feed `make_resharding_plan` from `get_weight_metadata()`
on both sides and drive `pull_weights_chunk` + `h2d_chunk` per chunk.

### 5.2 Async and error handling

- `RaidenControllerClientFacade.start_transfer` is **synchronous and
  blocking**, and `req_id` is accepted but never serialized. Asynchronous
  status (the proposal's `get_transfer_status`) does not exist and must be
  built before the orchestrator can overlap transfer with anything.
- A failed transfer currently leaves the sampler with partially-updated
  weights and no way to detect it. Phase 3 should add a session manifest: the
  set of parameters a session promised, checked at `finish_weight_update`, so
  an incomplete update fails loudly instead of serving a chimera.
- Decide recovery policy: refuse to resume, or re-pull the whole set.

### 5.3 Scale and coverage

- Chunked/pipelined transfer so a large model overlaps network and H2D.
- Quantization hook in `finish_weight_update` (vLLM's FP8 example).
- torchax path support — needs a key convention bridging dotted torch
  parameter names to the flat nnx paths `get_weight_metadata()` returns.
- Draft-model retargeting (`start_draft_weight_update`).
- LoRA-only sync: much smaller payloads; likely worth a dedicated path.
- Metrics: bytes moved, wall time, HBM high-water — mirroring
  `tpu_connector_stats.py`.

### 5.4 Open questions

1. Does the trainer own resharding, or does Raiden? Affects whether 5.1 is
   needed at all.
2. Does the sampler ever become a *sender* (KV migration for disaggregated
   rollouts)? The proposal implies yes; nothing here supports it.
3. One controller per sampler replica, or one global? Affects failure blast
   radius when a replica dies mid-transfer.

---

## 6. References

- vLLM native weight transfer:
  <https://docs.vllm.ai/en/latest/training/weight_transfer/>
- vLLM async RL: <https://docs.vllm.ai/en/latest/training/async_rl.md>
- tpu-raiden resharding design: `tpu_raiden/frameworks/jax/resharding_technical_report.md`
- tpu-raiden weight sync example: `examples/weight_sync/weight_sync_single_host.py`
- Existing KV transport in this repo: `tpu_inference/distributed/tpu_raiden_connector.py`
