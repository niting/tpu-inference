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
`update_weights`, `finish_weight_update` — over a pluggable transport. We keep
those names for consistency with the GPU path.

**Decision: the disaggregated path is push.** The trainer drives the transfer
and the sampler is a passive receiver. Everything below assumes it; where it
changes a design, that is called out.

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

Four methods on `TPUWorker` — `init_weight_transfer_engine`,
`start_weight_update`, `update_weights`, `finish_weight_update` — plus
`TPUModelRunner.get_weight_metadata()` and one data module,
`tpu_inference/distributed/weight_transfer.py`, holding `ParamMeta` and
`WeightUpdateRequest`.

That is the whole surface. There is no engine class, no backend registry and
no per-transport subclass: transport is a property of the request, not of an
object graph.

### Three design decisions worth knowing

**Method names are the integration point.** vLLM puts these four on
`vllm.v1.worker.gpu_worker.Worker`, not `WorkerBase`, and `TPUWorker`
subclasses `WorkerBase` — so it inherits none of them. But `collective_rpc`
dispatches **by method name**, so defining them on `TPUWorker` is sufficient
for `LLM`, `AsyncLLM`, `EngineCore`, all three TPU executors and the
weight-update HTTP routes to drive a TPU sampler unmodified. Pause/resume for
async RL needs nothing from us either — it is implemented purely at the
scheduler level upstream.

**We do not use vLLM's `WeightTransferEngine` abstraction.** It is torch-bound
in ways that buy nothing here: the constructor takes `torch.device` and
`torch.nn.Module`, the concrete `update_weights` calls
`torch.accelerator.synchronize()`, `WeightSource.__iter__` yields
`torch.Tensor`, and the receive path assumes
`model.load_weights([(name, tensor)])`. None of the shipped engines are
reusable either — NCCL, IPC and sparse-NCCL all call
`torch.cuda.current_stream()`. The JAX analogue of that sync is
`jax.block_until_ready(runner.state_leaves)`, which `update_weights` does
directly.

**Transport is selected by the payload.** `WeightUpdateRequest.weights`
carries a live pytree from a colocated trainer. An *empty* request is the
Raiden push case: the trainer wrote straight into this worker's HBM, so
`update_weights` has nothing to apply. No backend registry, no transport enum
-- presence of `weights` is the whole signal.

### Who owns the KV cache

The **worker**, not the engine. Receiving a full set of weights roughly
doubles peak HBM, and decoding from KV computed under old weights is wrong —
both are true regardless of transport, so `start_weight_update` frees the
cache and `finish_weight_update` reallocates it. Pass `free_kv_cache=False`
to opt out.

The prefix cache lives on the scheduler, so callers must still call
`reset_prefix_cache()` on the engine themselves.

### Usage

Colocated trainer, today:

```python
llm.reset_prefix_cache()
llm.start_weight_update()                     # frees the KV cache
llm.update_weights(dict(weights=new_state))   # repeatable for chunked transfer
llm.finish_weight_update()                    # reallocates the KV cache
```

Raiden push, once Phase 2 lands — same phases, and the trainer's push happens
between `start` and `finish`:

```python
llm.init_weight_transfer_engine({})            # binds WeightSynchronizers
sampler_eps = llm.collective_rpc("get_weight_transfer_endpoints")

llm.reset_prefix_cache()
llm.start_weight_update()                      # frees KV; workers now receivers
trainer.start_transfer(sampler_eps)            # pushes; only the trainer can
trainer.await_transfer()                       # observe completion
llm.finish_weight_update()                     # reallocates the KV cache
```

No engine configuration is required: `LLM.init_weight_transfer_engine` and
friends are pure `collective_rpc` calls upstream, with no guard on
`weight_transfer_config`.

### Limitations

- **flax_nnx path only.** `get_weight_metadata()` raises `NotImplementedError`
  on the torchax path, which stores weights as a flat `dict[str, jax.Array]`
  keyed by dotted torch parameter names and needs its own key convention.
- **Colocated transport only.** `weights` carries live `jax.Array` objects and
  an optional callable, so it requires an in-process or uniproc executor. The
  push path's phases already behave correctly (the KV cache is cycled, the
  session is guarded); what is missing is the `WeightSynchronizer` — Phase 2.
- **No endpoint publication.** `get_weight_transfer_endpoints()` does not
  exist yet. For push it is the single most important missing piece: the
  trainer cannot push without it.
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
construction**, so it writes into whatever arrays existed at that moment.

The steady-state forward pass is *not* a problem: `run_model` and
`run_draft_model` use `donate_argnums=1`, which is the KV cache — argument 0
is `state_leaves` and is **not** donated. (`create_jit_model` does use
`donate_argnums=(0,)`, but that runs once on the load path, before any
synchronizer exists.)

The problem is `_sync_weights`. It rebuilds the state:
`transfer_state_with_mappings` calls `set_value(...)` on each target param and
returns a new state, so `runner.state` / `runner.state_leaves` come to point at
*new* `jax.Array`s. A synchronizer constructed earlier still holds the old
buffers, and would write into arrays the model no longer executes against —
silently.

Resolution, in order of preference:

1. **Write in place, and don't call `_sync_weights` at all on this path.**
   Raiden's `h2d`/`h2d_chunk` DMA directly into the held buffers, so the arrays
   never need replacing and `runner.state` stays untouched. This is both the
   correct answer and the fast one. It does mean the *trainer* must supply data
   already named and laid out for the sampler — which is exactly what
   `get_weight_metadata()` is for. Name mapping and transposition move to the
   trainer; the sampler just receives bytes.
2. **Reconstruct per sync.** Build a fresh `WeightSynchronizer` each update.
   Correct and simple; pays setup + `BufferHoldAndAlias` + port binding every
   step.

Validate whichever is chosen with a test that syncs **twice** and asserts the
second sync actually lands — a stale hold fails silently, not loudly.

### 4.3 Sharding: pick the cheap answer first

Trainer mesh ≠ sampler mesh in general. Three strategies:

1. **Constrain the meshes equal.** Plain push works with shipping Raiden.
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

**We are going with push**: the trainer drives the transfer and the sampler is
a passive receiver. That decision removes work rather than adding it, because
Raiden's receiver auto-H2Ds in C++ on data receipt (`OnDataReceived()` →
`H2d()` + `Await()`). No new abstraction is needed; the phases stay as they
are:

- `init_weight_transfer_engine`: bucket `runner.state` (4.1), construct one
  `WeightSynchronizer` per bucket bound to the live weight buffers, and record
  this worker's per-shard endpoints. Must happen in the worker process —
  `UnpackJaxArrays` needs raw `xla::PjRtBuffer*` and this process's
  `addressable_shards`.
- **New:** `get_weight_transfer_endpoints()` returns that per-shard list for
  the orchestrator to hand to the trainer. This is the piece push cannot work
  without.
- `start_weight_update`: unchanged. Frees the KV cache; workers are now
  passive receivers.
- `update_weights`: **stays a no-op for push.** Nothing is handed over. Kept
  for the colocated path and for vLLM contract parity.
- `finish_weight_update`: unchanged today. Gains verification once chunk
  accounting exists (5.2).

Note what push does *not* need: no `pull_weights` call, no per-chunk payload,
no transport field. The sampler never learns the trainer's address — only the
reverse.

**Completion is not observable on the sampler.** The JAX `WeightSynchronizer`
exposes no receiver-side `poll`/`wait`/byte-count; `OnDataReceived` is
C++-internal. So the orchestrator must gate `finish_weight_update` on the
*trainer's* transfer status, which is exactly what the RL orchestration
proposal does with `get_transfer_status(req_id)`. This is structurally forced,
not a preference.

### 4.6 Phase 2 checklist

- [ ] tpu-raiden: heterogeneous per-layer slice sizes (4.1 option C)
- [ ] `TPU_WEIGHT_TRANSFER_PORT` + per-shard endpoint publication
- [ ] Bucketing of `runner.state` into synchronizer groups
- [ ] Resolve buffer lifetime (4.2); regression test for a **second** sync
- [ ] `get_weight_transfer_endpoints()` on worker and runner
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
- Decide recovery policy: refuse to resume, or re-push the whole set.

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
