# Inference

For training-to-inference weight transfer (`NewInferenceWorkerWrap`, broadcast vs. CUDA IPC, lifecycle), see [`weight_sync.md`](weight_sync.md).

## Architecture

- Key abstractions: `RemoteInferenceClient` , `ServerGroup`, `VLLMServerActor`, `VLLMRouter`
- `RemoteInferenceClient` interacts with HTTP endpoints: 
    - **Data plane**: Interact with router for completions requests.
    - **Control plane**: Fan-out to individual server URLs for weight sync, pause/resume.
- Shared inference interfaces and types live in `inference_servers/base.py` (`InferenceEngineInterface`, `InferenceEngineInput`/`Output`, `ConversationType`); shared helpers (`build_engine_runtime_env`, `get_sampling_params_for_backend`) live in `inference_servers/engine_utils.py`.

## vLLM Router

- `VLLMRouter` in `skyrl/backends/skyrl_train/inference_servers/vllm_router.py` wraps a child process running `vllm-router`. 

## Rollout Routing Replay (R3) stash

- With `enable_return_routed_experts=true`, each `VLLMServerActor` holds a `RoutedExpertsStash` (`inference_servers/routed_experts_stash.py`) keyed by `(model name, digest of prompt + response tokens)`.
- `/skyrl/v1/completions` wraps vLLM's completions handler: it stashes each choice's routing server-side and strips it from the response, so no client ever carries routing. Used by both the Tinker engine's sample path and the Tinker API's non-colocated forwarding client.
- The trainer pulls routing at `forward_backward` time via `/skyrl/v1/routed_experts/fetch` (control-plane fan-out, `.npz` payload). Lifecycle: `/skyrl/v1/routed_experts/weight_sync` (staleness-based drop, called after each `save_weights_for_sampler`) and `/skyrl/v1/routed_experts/clear` (model deletion).
- `/skyrl/v1/generate` still returns routing inline for the native RL path (`skyrl_gym_generator`).

## PD Disaggregation

Prefill-Decode disaggregation:
- **Config**: `enable_pd=true` and `num_prefill` passed to `ServerGroup` constructor. Requires a `kv_connector`
- **Server groups**: Separate prefill and decode `ServerGroup`s, one per engine.

## Key Config Knobs

All under `generator.inference_engine.*`:
- `enforce_eager` (bool, default true): With `enforce_eager=false`, there can be more mismatch between inference logprobs and trainer logprobs. It is recommended to use off policy correction methods like Truncated Importance Sampling (see `docs/content/docs/algorithms/off_policy_correction.mdx` for details) to prevent logprobs drift. 
- `gpu_memory_utilization` (float, default 0.8)
- `max_num_batched_tokens` (int, default 8192)
- `max_num_seqs` (int, default 1024)
- `enable_prefix_caching` (bool, default true)
- `enable_chunked_prefill` (bool, default true)
- `distributed_executor_backend` ("ray" or "mp")
- `engine_init_kwargs` (dict, pass-through to vLLM EngineArgs)

## Placement
- Colocated: vLLM and training workers (FSDP/Megatron) are placed on the same set of GPUs. We offload/backload each component as needed. During weight syncing, model weights from vLLM as well as model weights from the training workers remain on GPU
- Non-colocated: vLLM and training workers (FSDP/Megatron) are placed on a different set of GPUs. This reduces the number of available GPUs per component by half, but is in fact the preferred setup for agentic RL with SkyRL. This is because non-colocated setups allow for asynchronous training, where training and inference can progress together. Inference is typically dominated by a long tail of stragglers, and is also typically the time consuming component, and thus using half the number of GPUs doesn't affect inference time for a batch as much.
