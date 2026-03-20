# ATOM vLLM Plugin Guide

ATOM (AiTer Optimized Model) is AMD's lightweight LLM inference engine built on
[AITER](https://github.com/ROCm/aiter) kernels for ROCm/HIP GPUs. This guide
documents ATOM's **plugin mode**, which allows ATOM's optimized models, attention
backends, and MoE kernels to run inside [vLLM](https://github.com/vllm-project/vllm)
as an out-of-tree (OOT) backend.

> **Quick Reference**
>
> | Component | Import / Entry Point | Purpose |
> |-----------|---------------------|---------|
> | `register_platform` | `atom.plugin.vllm.register:register_platform` | vLLM platform plugin entry point — sets ATOM framework state and returns `ATOMPlatform` to control attention backend selection |
> | `register_model` | `atom.plugin.vllm.register:register_model` | vLLM general plugin entry point — overrides model registry with ATOM wrappers, patches MLA attention, and fixes `act_dtype` defaults |
> | `ATOMPlatform` | `atom.plugin.vllm.platform.ATOMPlatform` | ROCm platform subclass that selects ATOM attention backends |
> | `ATOMModelBase` | `atom.plugin.vllm.model_wrapper.ATOMModelBase` | Base wrapper adapting ATOM models to vLLM's `VllmModel` interface |
> | `PluginConfig` | `atom.plugin.config.PluginConfig` | Dataclass carrying plugin-specific configuration |
> | `generate_atom_config_for_plugin_mode` | `atom.plugin.config` | Builds an ATOM `Config` from vLLM's config |

---

## 1. Overview

When ATOM is installed alongside vLLM, it automatically registers itself via
Python entry points. vLLM discovers ATOM at startup and delegates model
construction, attention computation, and weight loading to ATOM's optimized
implementations — no code changes to vLLM are required.

Key benefits of plugin mode:

- **Rapid PoC for new ROCm hardware and optimizations** — the plugin architecture
  enables fast prototyping and validation of new AMD GPU generations, novel kernel
  implementations, and experimental optimization ideas (e.g., fused operators,
  new quantization schemes, MoE scheduling strategies) without forking or modifying
  vLLM itself. New hardware support and performance innovations land in ATOM first
  and are immediately available to the entire vLLM user base.
- **Full ecosystem compatibility** — ATOM serves as a transparent acceleration
  layer beneath vLLM. Serving APIs, CLI flags, OpenAI-compatible endpoints, and
  client tooling all remain identical — users simply deploy vLLM within a
  ROCm environment that ships ATOM pre-installed and gain optimized performance
  with zero workflow changes.

---

## 2. Supported Models

The plugin currently overrides the following model architectures in vLLM's
`ModelRegistry`:

| HuggingFace Architecture | ATOM Model Class | Wrapper | Category |
|---|---|---|---|
| `Qwen3ForCausalLM` | `atom.models.qwen3.Qwen3ForCausalLM` | `ATOMForCausalLM` | Dense |
| `Qwen3MoeForCausalLM` | `atom.models.qwen3_moe.Qwen3MoeForCausalLM` | `ATOMMoEForCausalLM` | MoE |
| `DeepseekV3ForCausalLM` | `atom.models.deepseek_v2.DeepseekV3ForCausalLM` | `ATOMMoEForCausalLM` | MoE |
| `GptOssForCausalLM` | `atom.models.gpt_oss.GptOssForCausalLM` | `ATOMMoEForCausalLM` | MoE |
| `Glm4MoeForCausalLM` | `atom.models.glm4_moe.Glm4MoeForCausalLM` | `ATOMMoEForCausalLM` | MoE |

Two wrapper classes cover all models:

- **`ATOMForCausalLM`** — for dense (non-MoE) causal language models.
- **`ATOMMoEForCausalLM`** — for Mixture-of-Experts causal language models.

Both inherit from `ATOMModelBase`, which implements `VllmModel`, `SupportsQuant`,
and `SupportsPP` interfaces.

---

## 3. Installation and Quick Start

### 3.1 Prerequisites

- AMD Instinct MI300X / MI300A / MI355X GPUs

### 3.2 Set Up the Environment

The recommended approach is to pull an official ATOM + vLLM Docker image from
[Docker Hub](https://hub.docker.com/r/rocm/atom-dev/tags?name=vllm). These
images ship with ROCm, PyTorch, AITER, ATOM, and a compatible vLLM build
pre-installed — no manual dependency management is required.

Pull the latest OOT image:

```bash
docker pull rocm/atom-dev:vllm-latest
```

Or pin to a specific vLLM version and nightly date for reproducibility:

```bash
docker pull rocm/atom-dev:vllm-v0.17.0-nightly_20260315
```

ATOM's vLLM entry points are registered automatically at install time. Once
the container is running, `vllm serve` will pick up ATOM with no additional
configuration.

### 3.3 Launch vLLM with ATOM Plugin

```bash
vllm serve amd/DeepSeek-R1-0528-MXFP4 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel
```

ATOM will log its activation at startup:

```
INFO atom: Register model DeepseekV3ForCausalLM to vLLM with atom.plugin.vllm.model_wrapper:ATOMMoEForCausalLM
INFO atom: Use atom attention backend
```

### 3.4 Disable the Plugin

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

## 4. Architecture

### 4.1 Plugin Discovery

ATOM registers two entry points in
[`pyproject.toml`](https://github.com/ROCm/ATOM/blob/main/pyproject.toml#L29-L39)
under the `vllm.platform_plugins` and `vllm.general_plugins` groups. These are
the **sole entry points** through which vLLM discovers and activates the ATOM
OOT plugin — all subsequent ATOM behavior (custom platform, model overrides,
attention backends, MoE kernels) flows from these two registrations:

```toml
[project.entry-points."vllm.platform_plugins"]
# Installed by default; disable at runtime with ATOM_DISABLE_VLLM_PLUGIN=1
atom = "atom.plugin.vllm.register:register_platform"

[project.entry-points."vllm.general_plugins"]
# Installed by default; disable at runtime with ATOM_DISABLE_VLLM_PLUGIN=1
atom_model_registry = "atom.plugin.vllm.register:register_model"
```

vLLM calls these at startup:

#### `register_platform()`

This is the **platform plugin entry point**. vLLM invokes it during platform
resolution to determine which hardware platform class to use.

- **When disabled** (`ATOM_DISABLE_VLLM_PLUGIN=1`): returns `None`, causing
  vLLM to fall back to its built-in `RocmPlatform` with no ATOM involvement.
- **When enabled** (default): calls `_set_framework_backbone("vllm")` to set
  ATOM's internal framework state to vLLM plugin mode, then returns the
  fully-qualified class name `"atom.plugin.vllm.platform.ATOMPlatform"`. vLLM
  imports this class and installs it as the active platform.

`ATOMPlatform` is a subclass of vLLM's `RocmPlatform`. Its primary
responsibility is overriding `get_attn_backend_cls()` to route attention
dispatch to ATOM's optimized backends (`AiterBackend` for MHA,
`AiterMLABackend` for MLA) instead of vLLM's default ROCm attention. See
[Section 6](#6-attention-integration) for details.

#### `register_model()`

This is the **general plugin entry point**. vLLM invokes it after platform
setup to allow plugins to perform additional initialization.

`register_model()` performs three tasks:

1. **Model registry override** — iterates over `_VLLM_MODEL_REGISTRY_OVERRIDES`
   (a mapping from HuggingFace architecture names to ATOM wrapper classes) and
   calls `vllm.ModelRegistry.register_model()` for each entry. This redirects
   vLLM's model construction so that when a user loads, e.g.,
   `DeepseekV3ForCausalLM`, vLLM instantiates ATOM's `ATOMMoEForCausalLM`
   wrapper instead of its built-in implementation. After updating the registry,
   it clears vLLM's model-loading LRU caches to ensure the overrides take
   effect immediately. The currently registered overrides are:

   | HuggingFace Architecture | ATOM Wrapper |
   |---|---|
   | `Qwen3ForCausalLM` | `ATOMForCausalLM` (dense) |
   | `Qwen3MoeForCausalLM` | `ATOMMoEForCausalLM` (MoE) |
   | `DeepseekV3ForCausalLM` | `ATOMMoEForCausalLM` (MoE) |
   | `GptOssForCausalLM` | `ATOMMoEForCausalLM` (MoE) |
   | `Glm4MoeForCausalLM` | `ATOMMoEForCausalLM` (MoE) |

2. **MLA attention patch** — calls `patch_vllm_mla_attention()` to patch vLLM's
   `MLAAttention` class so that its `process_weights_after_loading` and
   `forward_impl` methods are compatible with ATOM's MLA backend.

3. **Attention `process_weights_after_loading` patch** — patches both
   `vllm.Attention` and `vllm.MLAAttention` to supply a default `act_dtype`
   parameter (`torch.bfloat16`) to `process_weights_after_loading()`. This
   avoids a missing-argument error when ATOM's weight loader calls the method
   without explicitly passing `act_dtype`.

### 4.2 Plugin Lifecycle

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

### 4.3 Component Diagram

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

## 5. Configuration Translation

When vLLM constructs an ATOM model, `generate_atom_config_for_plugin_mode()` translates
vLLM's `VllmConfig` into an ATOM `Config`. The translation preserves vLLM's
scheduling, caching, and parallelism decisions while injecting ATOM-specific
compilation and plugin settings.

### 5.1 `PluginConfig` Fields

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

### 5.2 vLLM Config Mapping

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

## 6. Attention Integration

vLLM's OOT plugin interface allows an external platform to supply its own
attention backend. ATOM hooks into this by overriding
`ATOMPlatform.get_attn_backend_cls()` — the only contract point between vLLM and
the plugin for attention dispatch.

### 6.1 How the Backend Is Selected

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

### 6.2 Backend–vLLM Contract

Each ATOM backend fulfills vLLM's `AttentionBackend` interface by providing:

1. **Attention implementation class** — `PagedAttentionImpl` for MHA or
   `MLAAttention` for MLA. These are ATOM's own attention implementations
   decorated at import time with plugin-mode methods (via
   `PagedAttentionImplDecoratorForPluginMode` / `MLAAttentionImplDecoratorForPluginMode`)
   so they expose the `forward_impl_plugin_mode` entry point that vLLM calls.

2. **Metadata builder class** — translates vLLM's `CommonAttentionMetadata` into
   the metadata structure the ATOM kernels expect. The builders are similarly
   decorated (via `AiterAttentionMetadataBuilderDecoratorForPluginMode` /
   `AiterMLAAttentionMetadataBuilderDecoratorForPluginMode`) to inherit from
   vLLM's `AttentionMetadataBuilder` while injecting ATOM-specific `build()`
   logic.

3. **Static properties** — `get_kv_cache_shape`, `get_supported_kernel_block_sizes`,
   `get_supported_head_sizes`, etc. These tell vLLM how to allocate and manage
   KV cache blocks in the format ATOM's kernels expect.

### 6.3 Key Design Points

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

---

## 7. Model Wrapper

`ATOMModelBase` is the bridge between vLLM's model interface and ATOM's model
implementations.

### 7.1 Initialization

When vLLM instantiates a model, `ATOMModelBase.__init__` performs:

1. Stores vLLM config references (`model_config`, `cache_config`, `quant_config`, etc.)
2. Calls `generate_atom_config_for_plugin_mode(vllm_config)` to build the ATOM config
3. Calls `set_attn_cls()` to set `ops.Attention = PagedAttention` for vLLM mode
4. Calls `init_aiter_dist()` to initialize AITER's distributed environment (NCCL, TP/EP groups)
5. Looks up the ATOM model class from `_ATOM_MODEL_CLASSES` and constructs it
6. Initializes pipeline-parallel group references (`pp_group`, `tp_group`)

### 7.2 Forward Pass

```python
def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
    # Pipeline parallel: use intermediate tensors from previous stage
    if not self.pp_group.is_first_rank:
        input_ids = None
        inputs_embeds = intermediate_tensors["hidden_states"]

    # Copy positions for MLA attention in graph mode
    if "positions" in self.atom_config.compilation_config.static_forward_context:
        buf = self.atom_config.compilation_config.static_forward_context["positions"]
        buf[:positions.numel()].copy_(positions)

    hidden_states = self.model(input_ids=input_ids, positions=positions, ...)

    if not self.pp_group.is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states})
    return hidden_states
```

### 7.3 Weight Loading

`ATOMModelBase.load_weights()` delegates to `load_model_in_plugin_mode()`, which
uses ATOM's custom weight loader. The vLLM-provided `weights` iterator is not
used directly — ATOM loads weights from the model path using its own safetensors
loader with support for sharded checkpoints and quantization-aware loading.

---

## 8. Environment Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `ATOM_DISABLE_VLLM_PLUGIN` | bool | `0` (false) | Set to `1` to disable the entire ATOM vLLM plugin (platform + model registration). vLLM runs in pure ROCm mode. |
| `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION` | bool | `0` (false) | Set to `1` to disable only ATOM's attention backends. ATOM models are still used, but attention falls back to vLLM's default ROCm backend. |
| `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION` | bool | `0` (false) | Enable QK-norm + RoPE + cache + quant fusion in attention. Recommended for Qwen3-MoE models. |

---

## 9. Known Limitations

- **CUDA graphs** — ATOM's own CUDA graph logic is disabled in plugin mode.
  CUDA graph capture and replay are managed entirely by vLLM.

- **`max_num_batched_tokens` threshold** — When `max_num_batched_tokens >= 18432`,
  there is a known issue with illegal memory access in the ASM `fused_moe` kernel.
  If encountered, reduce `max_num_batched_tokens` or switch to the CK `fused_moe`
  kernel.

- **Multi-modality** — `ATOMPlatform` does not yet override
  `get_supported_vit_attn_backends()`, so Vision Transformer attention in
  multi-modal models falls back to vLLM's default backend.

---

## 10. Troubleshooting

### Plugin not activating

Verify ATOM is installed and entry points are registered:

```bash
pip show atom
python -c "from atom.plugin.vllm.register import register_platform; print(register_platform())"
```

Expected output: `atom.plugin.vllm.platform.ATOMPlatform`

### Falling back to vLLM attention

Check that `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION` is not set:

```bash
echo $ATOM_DISABLE_VLLM_PLUGIN_ATTENTION
```

Look for this log line at startup:

```
INFO atom: Use atom attention backend
```

If you see `Fallback to original vLLM attention backend`, the attention plugin
is disabled.

### Illegal memory access in MoE

If you encounter `illegal memory access` errors with large batch sizes, try:

```bash
vllm serve <model> --max-num-batched-tokens 16384
```

Or switch to the CK fused_moe kernel by setting the appropriate ATOM environment
variable.

---

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
