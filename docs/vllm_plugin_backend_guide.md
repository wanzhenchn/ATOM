# ATOM vLLM Plugin Backend

ATOM can work as the vLLM out-of-tree (OOT) plugin backend — installed as a separate Python package and plugged into vLLM through vLLM's official plugin interfaces. This keeps the integration clean while letting ATOM reuse the mature serving and runtime features already provided by vLLM.

This integration follows the direction described in the [RFC to enable ATOM as a vLLM out-of-tree platform](https://github.com/ROCm/ATOM/issues/201). The high-level idea is that vLLM remains the framework-level runtime, while ATOM focuses on model-level and kernel-level optimization for AMD GPUs. In this mode, ATOM serves as the optimized execution backend and an incubation layer for new kernels, fusions, and model implementations before they are mature enough to be upstreamed.

## 1. Architecture
### 1.1 Design overview
In practice, the responsibilities are split as follows:

| Layer | Responsibility |
|---|---|
| vLLM | API server, CLI, engine, scheduler, worker orchestration, cache management, and framework-level features |
| ATOM | Platform plugin, model registry overrides, model wrappers, attention backends, and the optimized execution path built around ATOM/AITER integrations |
| Integration boundary | vLLM calls the official plugin hooks, while ATOM implements the required platform and model interfaces without changing vLLM source |

This relationship is important: ATOM is not replacing vLLM as a serving framework. Instead, ATOM plugs optimized model execution components into the extension points that vLLM already exposes.

### 1.2 How it works
When the `atom` package is installed in the same Python environment as `vllm`, two entry points are exposed following the official vLLM plugin convention:

```toml
[project.entry-points."vllm.platform_plugins"]
atom = "atom.plugin.vllm.register:register_platform"

[project.entry-points."vllm.general_plugins"]
atom_model_registry = "atom.plugin.vllm.register:register_model"
```

During `vllm serve` startup, vLLM scans installed Python packages, loads these entry points, and activates the ATOM hooks:

- `register_platform()` returns `atom.plugin.vllm.platform.ATOMPlatform`, so vLLM resolves `current_platform` to the ATOM platform.
- `register_model()` updates selected vLLM `ModelRegistry` entries to ATOM wrappers such as `ATOMForCausalLM` and `ATOMMoEForCausalLM`.
- When vLLM constructs attention layers, `ATOMPlatform.get_attn_backend_cls()` returns `atom.model_ops.attentions.aiter_attention.AiterBackend` or `atom.model_ops.attentions.aiter_mla.AiterMLABackend`.
- When a supported model is instantiated, the ATOM wrapper creates the ATOM plugin config, initializes the ATOM/AITER runtime state, and constructs the ATOM model implementation.
- vLLM continues to drive request scheduling and serving, while the hot model execution path runs through ATOM model code, ATOM attention backends, and AITER-backed kernels.

### 1.3 Plugin lifecycle

```
vLLM startup
│
├─ 1. register_platform()
│     ├─ _set_framework_backbone("vllm")
│     └─ return "atom.plugin.vllm.platform.ATOMPlatform"
│
├─ 2. register_model()
│     ├─ Override ModelRegistry for supported architectures
│     ├─ patch_vllm_mla_attention()
│     └─ Patch Attention.process_weights_after_loading
│
├─ 3. vLLM loads model → ATOMModelBase.__init__()
│     ├─ generate_atom_config_for_plugin_mode(vllm_config)
│     │     └─ _generate_atom_config_from_vllm_config()
│     │           ├─ Build PluginConfig (vLLM-specific fields)
│     │           └─ Build ATOM Config (model, TP, KV cache, etc.)
│     ├─ set_attn_cls() → ops.Attention = PagedAttention
│     ├─ init_aiter_dist() → initialize AITER distributed env
│     └─ Construct ATOM model (e.g., DeepseekV3ForCausalLM)
│
├─ 4. ATOMPlatform.get_attn_backend_cls()
│     ├─ MLA model → AiterMLABackend
│     └─ MHA model → AiterBackend
│
└─ 5. Forward pass
      ├─ vLLM calls ATOMModelBase.forward()
      ├─ Delegates to self.model(input_ids, positions, ...)
      └─ Attention uses ATOM's AITER kernels via plugin decorators
```

### 1.4 Key Modules

| Module | Purpose |
|---|---|
| `atom.plugin.vllm.register` | vLLM plugin entry points for platform and model registration |
| `atom.plugin.vllm.platform` | The ATOM platform class exposed to vLLM |
| `atom.plugin.vllm.model_wrapper` | ATOM model wrappers used by vLLM model construction |
| `atom.model_ops.attentions.aiter_attention` | ATOM MHA attention backend for vLLM plugin mode |
| `atom.model_ops.attentions.aiter_mla` | ATOM MLA attention backend for vLLM plugin mode |

### 1.5 Component Diagram

```
atom/plugin/
├── __init__.py              # Public API: is_vllm, is_plugin_mode
├── prepare.py               # Framework detection and state management
├── config.py                # PluginConfig + vLLM-to-ATOM config translation
├── register.py              # set_attn_cls, init_aiter_dist
├── attention.py             # vLLM attention metadata builders and backend decorators
├── attention_mha.py         # MHA (PagedAttention) plugin-mode decorator
├── attention_mla.py         # MLA plugin-mode methods and decorator
├── moe.py                   # FusedMoE decorator for plugin mode
└── vllm/
    ├── __init__.py           # vLLM sub-package exports
    ├── register.py           # register_platform(), register_model()
    ├── platform.py           # ATOMPlatform (RocmPlatform subclass)
    ├── model_wrapper.py      # ATOMModelBase, ATOMForCausalLM, ATOMMoEForCausalLM
    └── mla_patch.py          # Patches vLLM MLAAttention for ATOM MLA integration
```

---

## 2. Configuration Translation

When vLLM constructs an ATOM model, `generate_atom_config_for_plugin_mode()` translates
vLLM's `VllmConfig` into an ATOM `Config`. The translation preserves vLLM's
scheduling, caching, and parallelism decisions while injecting ATOM-specific
compilation and plugin settings.

### 2.1 `PluginConfig` Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `model_config` | `Any` | `None` | vLLM's model config object |
| `rank` | `int` | `0` | Current process rank |
| `is_plugin_mode` | `bool` | `False` | Always `True` when running as a plugin |
| `is_vllm` | `bool` | `False` | `True` when running inside vLLM |
| `vllm_scheduler_config` | `Any` | `None` | vLLM scheduler config |
| `vllm_cache_config` | `Any` | `None` | vLLM cache config |
| `vllm_quant_config` | `Any` | `None` | vLLM quantization config |
| `vllm_use_atom_attention` | `bool` | `False` | Whether ATOM attention is active |

### 2.2 vLLM Config Mapping

The following table shows how vLLM config fields map to ATOM `Config` fields:

| ATOM `Config` Field | Source (vLLM) |
|---|---|
| `model` | `model_config.model` |
| `max_num_batched_tokens` | `scheduler_config.max_num_batched_tokens` |
| `max_num_seqs` | `scheduler_config.max_num_seqs` |
| `max_model_len` | `model_config.max_model_len` (or `scheduler_config.max_model_len`) |
| `gpu_memory_utilization` | `cache_config.gpu_memory_utilization` |
| `tensor_parallel_size` | `parallel_config.tensor_parallel_size` |
| `kv_cache_block_size` | `cache_config.block_size` |
| `num_kvcache_blocks` | `cache_config.num_gpu_blocks` |
| `kv_cache_dtype` | `cache_config.cache_dtype` |
| `enable_prefix_caching` | `cache_config.enable_prefix_caching` |
| `enable_expert_parallel` | `parallel_config.enable_expert_parallel` |
| `compilation_config.level` | `compilation_config.mode` |
| `enforce_eager` | Always `True` (ATOM does not use its own CUDA graph logic in plugin mode) |

**CUDA graphs vs torch.compile:**

- **CUDA graphs** — In plugin mode, ATOM sets `enforce_eager=True` and
  `use_cudagraph=False` in its own `Config`, meaning ATOM's CUDA graph capture
  and replay logic is completely disabled. CUDA graph management is fully
  delegated to vLLM — vLLM decides when to capture, which batch sizes to graph,
  and how to replay. ATOM's attention backends cooperate by implementing
  `build_for_cudagraph_capture()` so that vLLM can capture ATOM kernels inside
  its own CUDA graphs.

- **torch.compile** — In contrast, torch.compile is handled entirely by ATOM,
  not by vLLM. ATOM's `@support_torch_compile` decorator wraps each model's
  `forward` method and routes compilation through ATOM's own `VllmBackend`.
  The compilation level is derived from vLLM's `compilation_config.mode`
  (e.g., `PIECEWISE`), but the actual compilation pipeline — including graph
  splitting, Inductor invocation, and compiled-graph caching — is ATOM's own
  implementation.

  Graph splitting is a key difference: ATOM splits the `torch.fx` graph at
  attention boundaries (the `unified_attention` op registered by vLLM) so that
  each piecewise subgraph can be compiled and cached independently. This split
  strategy is defined in ATOM's `split_graph()` / `_split_judge_func()` and is
  independent of vLLM's compilation backend.

---

## 3. Attention Integration

vLLM's OOT plugin interface allows an external platform to supply its own
attention backend. ATOM hooks into this by overriding
`ATOMPlatform.get_attn_backend_cls()` — the only contract point between vLLM and
the plugin for attention dispatch.

### 3.1 How the Backend Is Selected

When vLLM resolves the attention backend for a model, it calls the platform's
`get_attn_backend_cls()`. ATOM's implementation returns one of two backends based
on the model's attention type:

| Model Attention Type | Returned Backend | Example Models |
|---|---|---|
| MLA (`use_mla == True`) | `AiterMLABackend` | DeepSeek-R1, Kimi-K2 |
| Standard MHA | `AiterBackend` | Qwen3, Llama |

Setting `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1` causes `ATOMPlatform` to delegate
back to the parent `RocmPlatform.get_attn_backend_cls()`, restoring vLLM's
built-in ROCm attention path.

### 3.2 Backend–vLLM Contract

Each ATOM backend fulfills vLLM's `AttentionBackend` interface by providing:

- **Attention implementation class** — `PagedAttentionImpl` for MHA or
   `MLAAttention` for MLA. These are ATOM's own attention implementations
   decorated at import time with plugin-mode methods (via
   `PagedAttentionImplDecoratorForPluginMode` / `MLAAttentionImplDecoratorForPluginMode`)
   so they expose the `forward_impl_plugin_mode` entry point that vLLM calls.

- **Metadata builder class** — translates vLLM's `CommonAttentionMetadata` into
   the metadata structure the ATOM kernels expect. The builders are similarly
   decorated (via `AiterAttentionMetadataBuilderDecoratorForPluginMode` /
   `AiterMLAAttentionMetadataBuilderDecoratorForPluginMode`) to inherit from
   vLLM's `AttentionMetadataBuilder` while injecting ATOM-specific `build()`
   logic.

- **Static properties** — `get_kv_cache_shape`, `get_supported_kernel_block_sizes`,
   `get_supported_head_sizes`, etc. These tell vLLM how to allocate and manage
   KV cache blocks in the format ATOM's kernels expect.

### 3.3 Key Design Points

- **Decorator-based injection** — ATOM does not fork or subclass vLLM's attention
  classes directly. Instead, Python decorators dynamically replace base classes
  and inject methods at import time, keeping ATOM's attention code decoupled from
  vLLM's internal class hierarchy.

- **`forward_includes_kv_cache_update = True`** — both backends declare that the
  KV cache write happens inside the forward pass. This tells vLLM to skip its
  separate cache-update step and gives ATOM full control over the
  RoPE → cache → attention pipeline.

- **`accept_output_buffer`** — set to `False` for MHA (ATOM allocates its own
  output tensor) and `True` for MLA (vLLM provides the output buffer). This
  reflects the different memory-management needs of each attention type.

- **Extend workspace** — the MHA metadata builder allocates an `extend_workspace`
  buffer outside of vLLM's memory accounting for chunked-prefill KV gathering.
  If you encounter OOM during chunked prefill, consider lowering
  `gpu_memory_utilization`.



## 4. Supported Models
Currently, the plugin backend supports the following model architectures:

| HF architecture | ATOM model implementation | Model family example |
|---|---|---|
| `Qwen3ForCausalLM` | `atom.models.qwen3.Qwen3ForCausalLM` | Qwen3 dense |
| `Qwen3MoeForCausalLM` | `atom.models.qwen3_moe.Qwen3MoeForCausalLM` | Qwen3 MoE |
| `GptOssForCausalLM` | `atom.models.gpt_oss.GptOssForCausalLM` | GPT-OSS |
| `DeepseekV3ForCausalLM` | `atom.models.deepseek_v2.DeepseekV3ForCausalLM` | DeepSeek-R1 / DeepSeek V3 / Kimi-K2 style models |
| `Glm4MoeForCausalLM` | `atom.models.glm4_moe.Glm4MoeForCausalLM` | GLM-4-MoE |

`Kimi-K2` is also supported. Although it is usually loaded with `--trust-remote-code`, it shares the same DeepSeek-style MLA+MoE architecture path and reuses `atom.models.deepseek_v2.DeepseekV3ForCausalLM` in the ATOM vLLM OOT backend.

---

## 5. Installation and Quick Start

### 5.1 Prerequisites

- AMD Instinct MI300X / MI300A / MI355X GPUs

### 5.2 Set Up the Environment

The recommended approach is to pull an official ATOM + vLLM Docker image from
[Docker Hub](https://hub.docker.com/r/rocm/atom-dev/tags?name=vllm). These
images ship with ROCm, PyTorch, AITER, ATOM, and a compatible vLLM build
pre-installed — no manual dependency management is required.

Pull the latest OOT image:

```bash
docker pull rocm/atom-dev:vllm-latest
```

If you need an OOT docker image for a specific vLLM version or a specific release date, browse the available tags on [Docker Hub](https://hub.docker.com/r/rocm/atom-dev/tags) and pull the exact tag you need there. For example, to pull the OOT docker adapted to vLLM `0.17.0` on `2026-03-15`:

```bash
docker pull rocm/atom-dev:vllm-v0.17.0-nightly_20260315
```

### 5.3 Launch vLLM with ATOM Plugin

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options, OpenAI-compatible API usage, and client patterns, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

```bash
vllm serve ${model} \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --no-enable-prefix-caching
```

ATOM will log its activation at startup:

```
INFO atom: Register model DeepseekV3ForCausalLM to vLLM with atom.plugin.vllm.model_wrapper:ATOMMoEForCausalLM
INFO atom: Use atom attention backend
```

### 5.4 Benchmark Serving
Users can use the default vllm bench commands for performance benchmarking.
```bash
vllm bench serve \
    --host localhost \
    --port 8000 \
    --model ${model} \
    --dataset-name random \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --max-concurrency 64 \
    --num-prompts 640 \
    --trust_remote_code \
    --percentile-metrics ttft,tpot,itl,e2el
```

### 5.5 Enable Profiling

If you want to collect profiles, add the recommended commands by vLLM with `--profiler-config "$profiler_config"`.

```bash
profiler_dir=./

profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":true,"torch_profiler_record_shapes":true}' \
    "${profiler_dir}")
```


### 5.6 Disable ATOM Plugin

This is intended for **debugging only**. When the ATOM plugin is disabled, vLLM
falls back to its built-in ROCm path, which may encounter version mismatches
with the AITER library bundled in the environment. To run pure vLLM without ATOM,
set environment variables before launching:

```bash
# Disable the entire ATOM plugin (platform + models)
export ATOM_DISABLE_VLLM_PLUGIN=1

# Or disable only ATOM attention (keep ATOM models but use vLLM attention)
export ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1
```

---

## 6. Environment Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `ATOM_DISABLE_VLLM_PLUGIN` | bool | `0` (false) | Set to `1` to disable the entire ATOM vLLM plugin (platform + model registration). vLLM runs in pure ROCm mode. |
| `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION` | bool | `0` (false) | Set to `1` to disable only ATOM's attention backends. ATOM models are still used, but attention falls back to vLLM's default ROCm backend. |
| `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION` | bool | `0` (false) | Enable QK-norm + RoPE + cache + quant fusion in attention. Recommended for Qwen3-MoE models. |


## Source Files

| File | Description |
|------|-------------|
| `atom/plugin/__init__.py` | Public API: `is_vllm`, `is_plugin_mode` |
| `atom/plugin/prepare.py` | Framework detection and `_CURRENT_FRAMEWORK` state management |
| `atom/plugin/config.py` | `PluginConfig` dataclass, `generate_atom_config_for_plugin_mode()`, vLLM config translator |
| `atom/plugin/register.py` | `set_attn_cls()`, `init_aiter_dist()` |
| `atom/plugin/attention.py` | vLLM attention metadata builders, backend decorators, `unified_attention_with_output_base_for_plugin_mode` |
| `atom/plugin/attention_mha.py` | MHA `PagedAttentionImpl` plugin-mode decorator |
| `atom/plugin/attention_mla.py` | MLA plugin-mode methods and `MLAAttentionImplDecoratorForPluginMode` |
| `atom/plugin/moe.py` | `FusedMoEDecoratorForPluginMode` — renames `FusedMoE` to `ATOMFusedMoE` in vLLM |
| `atom/plugin/vllm/__init__.py` | vLLM sub-package exports: `register_model`, `register_platform` |
| `atom/plugin/vllm/register.py` | `register_platform()`, `register_model()`, model registry overrides, attention patches |
| `atom/plugin/vllm/platform.py` | `ATOMPlatform` — `RocmPlatform` subclass selecting ATOM attention backends |
| `atom/plugin/vllm/model_wrapper.py` | `ATOMModelBase`, `ATOMForCausalLM`, `ATOMMoEForCausalLM` — vLLM model wrappers |
| `atom/plugin/vllm/mla_patch.py` | Patches vLLM `MLAAttention.process_weights_after_loading` and `forward_impl` |
| `pyproject.toml` | Entry point declarations for `vllm.platform_plugins` and `vllm.general_plugins` |
