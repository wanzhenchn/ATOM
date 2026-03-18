window.BENCHMARK_DATA = {
  "lastUpdate": 1773870386593,
  "repoUrl": "https://github.com/ROCm/ATOM",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "id": "9567af916d7b6fa8d66607d668eec41464448a8c",
          "message": "perf: remove ATOM_DISABLE_MMAP from benchmark workflow to speed up model loading",
          "timestamp": "2026-03-14T10:29:47Z",
          "url": "https://github.com/ROCm/ATOM/commit/9567af916d7b6fa8d66607d668eec41464448a8c"
        },
        "date": 1773484446473,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 0/0 c=128 throughput (tok/s)",
            "value": 4262.89,
            "unit": "tok/s"
          },
          {
            "name": "DeepSeek-R1-0528 0/0 c=128 TPOT (ms)",
            "value": 28.81,
            "unit": "ms"
          },
          {
            "name": "DeepSeek-R1-0528 0/0 c=128 throughput (tok/s)",
            "value": 6914.47,
            "unit": "tok/s"
          },
          {
            "name": "DeepSeek-R1-0528 0/0 c=128 TPOT (ms)",
            "value": 17.39,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "id": "3bb2ae88fa67eb5fd45c6e352b10c32928850314",
          "message": "fix: regression analysis skipped + MTP not distinguished in report\n\n- Add always() to regression-analysis job if-condition (fix: skipped when\n  summarize job uses if: always())\n- Add --use-chat-template for MTP benchmark via bench_args matrix field\n- Detect variant tag (e.g. mtp3) from result filename in summarize.py\n  to distinguish MTP vs non-MTP in regression report\n- Replace jq with python3 for JSON injection in atom_test.sh\n- Pass BENCH_EXTRA_ARGS through to benchmark_serving",
          "timestamp": "2026-03-14T11:05:13Z",
          "url": "https://github.com/ROCm/ATOM/commit/3bb2ae88fa67eb5fd45c6e352b10c32928850314"
        },
        "date": 1773488649095,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 throughput (tok/s)",
            "value": 4348.28,
            "unit": "tok/s"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TPOT (ms)",
            "value": 28.32,
            "unit": "ms"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 throughput (tok/s)",
            "value": 5380.99,
            "unit": "tok/s"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TPOT (ms)",
            "value": 22.46,
            "unit": "ms"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "id": "d218ecff3fe490d92fb6244876dd1e538b1b6943",
          "message": "feat: custom benchmark dashboard, MTP variant display, fix-then-sweep rule\n\nDashboard:\n- Add custom dark-theme dashboard with tabs (Overview/Throughput vs Latency/\n  Throughput/Latency/Historical)\n- Interactive vs Token Throughput scatter chart (like InferenceX)\n- Overview table with commit + CI run links\n- Add TTFT, Total Throughput metrics to dashboard entries\n- Use cdnjs.cloudflare.com CDN for Chart.js\n\nsummarize.py:\n- Extract _display_model() to eliminate duplicate variant logic\n- _config_key reuses _display_model\n\nWorkflow:\n- Detect variant from filename for dashboard trend lines\n- Add run URL as extra field for click-to-CI-run\n- auto-push=false, deploy custom index.html to gh-pages\n- Fix: save dashboard template before git checkout gh-pages\n- Fix: git checkout back to SHA instead of detached HEAD\n- Restore ATOM_DISABLE_MMAP=true\n- Regression trace: group by variant, correct server/bench args\n- Reduce profile prompts to CONC*2\n- Remove unused import os\n\nCLAUDE.md:\n- Add Fix-then-sweep rule to Critical Rules",
          "timestamp": "2026-03-14T13:02:29Z",
          "url": "https://github.com/ROCm/ATOM/commit/d218ecff3fe490d92fb6244876dd1e538b1b6943"
        },
        "date": 1773494563791,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 throughput (tok/s)",
            "value": 4274.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 Total Tput (tok/s)",
            "value": 8557.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TTFT (ms)",
            "value": 318.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TPOT (ms)",
            "value": 28.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 throughput (tok/s)",
            "value": 6038.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 Total Tput (tok/s)",
            "value": 12071.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TTFT (ms)",
            "value": 581.14,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TPOT (ms)",
            "value": 40.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 throughput (tok/s)",
            "value": 5467.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 10963.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TTFT (ms)",
            "value": 334.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TPOT (ms)",
            "value": 22.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 throughput (tok/s)",
            "value": 7283.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 14583.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TTFT (ms)",
            "value": 534.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TPOT (ms)",
            "value": 33.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23088513632"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "id": "b16705c354088e18f754560da3ccfb13102db46d",
          "message": "fix: pkill in regression trace exits non-zero when no process found, causing step failure",
          "timestamp": "2026-03-14T14:41:35Z",
          "url": "https://github.com/ROCm/ATOM/commit/b16705c354088e18f754560da3ccfb13102db46d"
        },
        "date": 1773499986910,
        "tool": "customBiggerIsBetter",
        "benches": []
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "id": "3f34dbbe0cb99a8bc2e683788a9a999597e7489c",
          "message": "fix: benchmark job failures and Copilot review\n\nCritical fixes:\n- Remove backslash on $model_path in docker exec (all models got empty --model)\n- Fix GLM-5-FP8 HF path: amd/GLM-5-FP8 → zai-org/GLM-5-FP8\n\nCopilot review fixes:\n- Fix invalid CSS: hsl() backgroundColor → hsla()\n- Remove unnecessary # noqa: F811 on record_function import\n- Batch llvm-cxxfilt demangle calls (single subprocess)\n- Update stale prefill comment\n- Guard ZeroDivisionError in draft step overhead\n- Truncate prefill ctx label for large batches (>5 seqs)",
          "timestamp": "2026-03-14T15:04:33Z",
          "url": "https://github.com/ROCm/ATOM/commit/3f34dbbe0cb99a8bc2e683788a9a999597e7489c"
        },
        "date": 1773502931429,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 throughput (tok/s)",
            "value": 4261.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 Total Tput (tok/s)",
            "value": 8533.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TTFT (ms)",
            "value": 343.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TPOT (ms)",
            "value": 28.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 throughput (tok/s)",
            "value": 2894.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5789.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TTFT (ms)",
            "value": 213.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TPOT (ms)",
            "value": 21.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 throughput (tok/s)",
            "value": 5330.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 10689.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TTFT (ms)",
            "value": 337.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TPOT (ms)",
            "value": 22.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 throughput (tok/s)",
            "value": 3754.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 7523.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TTFT (ms)",
            "value": 236.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TPOT (ms)",
            "value": 16.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 throughput (tok/s)",
            "value": 2608.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 Total Tput (tok/s)",
            "value": 5222.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TTFT (ms)",
            "value": 422.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TPOT (ms)",
            "value": 47.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 throughput (tok/s)",
            "value": 1695.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3391.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TTFT (ms)",
            "value": 286.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TPOT (ms)",
            "value": 36.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 throughput (tok/s)",
            "value": 8128.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 Total Tput (tok/s)",
            "value": 16274.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 TTFT (ms)",
            "value": 183.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 TPOT (ms)",
            "value": 15.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 throughput (tok/s)",
            "value": 5648.93,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 Total Tput (tok/s)",
            "value": 11300.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 TTFT (ms)",
            "value": 118.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 TPOT (ms)",
            "value": 10.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090517508"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "24c76973ae282c1fc30f5f971bc058c9bee22a31",
          "message": "docs: update README, add DeepSeek/GPT-OSS recipes, fix stale branch refs (#328)\n\n* docs: update README, add DeepSeek/GPT-OSS recipes, fix stale branch refs\n\nREADME:\n- Add CI/Benchmark/Dashboard badges\n- Add Qwen3-Next to model table\n- Add MTP serving example\n- Add live benchmark dashboard section with feature list\n- Add profiler manual curl API usage\n- Add Env Vars to docs table\n- Add DeepSeek-R1, GPT-OSS, Qwen3-Next recipes and framework integration links\n\nNew recipes:\n- DeepSeek-R1: BF16/MXFP4/MTP3 configs, performance baselines, accuracy\n- GPT-OSS: single GPU and DP+EP configs\n\nRecipe fixes:\n- GLM-5: zlr/glm5 branch → main\n- Kimi-K2: guanbao/kimi_k2_fp4 branch → main\n- Qwen3-235b: specific commit refs → main\n- SGLang: update PR status, remove TODO\n- All: git checkout specific_branch_or_commit → main\n\nHousekeeping:\n- .gitignore: exclude .claude/plan/\n- Remove accidentally committed plan file\n\n* fix: docs deploy to /docs/ subdir, restore benchmark-dashboard on gh-pages",
          "timestamp": "2026-03-14T17:26:07Z",
          "url": "https://github.com/ROCm/ATOM/commit/24c76973ae282c1fc30f5f971bc058c9bee22a31"
        },
        "date": 1773527592500,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 throughput (tok/s)",
            "value": 93.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 Total Tput (tok/s)",
            "value": 186.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 TTFT (ms)",
            "value": 68.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 TPOT (ms)",
            "value": 10.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 throughput (tok/s)",
            "value": 4201.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 Total Tput (tok/s)",
            "value": 8412.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TTFT (ms)",
            "value": 321.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TPOT (ms)",
            "value": 29.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 throughput (tok/s)",
            "value": 1135.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2282.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 TTFT (ms)",
            "value": 137.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 TPOT (ms)",
            "value": 13.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 throughput (tok/s)",
            "value": 179.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 Total Tput (tok/s)",
            "value": 355.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 TTFT (ms)",
            "value": 91.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 TPOT (ms)",
            "value": 11.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 throughput (tok/s)",
            "value": 5430.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 Total Tput (tok/s)",
            "value": 10855.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TTFT (ms)",
            "value": 1381.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TPOT (ms)",
            "value": 44.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 throughput (tok/s)",
            "value": 1664.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3324.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 TTFT (ms)",
            "value": 156.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 TPOT (ms)",
            "value": 18.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 throughput (tok/s)",
            "value": 342.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 Total Tput (tok/s)",
            "value": 689.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TTFT (ms)",
            "value": 94.42,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TPOT (ms)",
            "value": 11.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 throughput (tok/s)",
            "value": 2751.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5504.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TTFT (ms)",
            "value": 216.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TPOT (ms)",
            "value": 22.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 throughput (tok/s)",
            "value": 586.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1168.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 TTFT (ms)",
            "value": 101.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 TPOT (ms)",
            "value": 13.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 throughput (tok/s)",
            "value": 91.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 Total Tput (tok/s)",
            "value": 103.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 TTFT (ms)",
            "value": 65.97,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 TPOT (ms)",
            "value": 10.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 throughput (tok/s)",
            "value": 1129.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 Total Tput (tok/s)",
            "value": 1270.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 TTFT (ms)",
            "value": 123.29,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 TPOT (ms)",
            "value": 13.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 throughput (tok/s)",
            "value": 178.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 Total Tput (tok/s)",
            "value": 200.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 TTFT (ms)",
            "value": 95.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 TPOT (ms)",
            "value": 11.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 throughput (tok/s)",
            "value": 1750.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 Total Tput (tok/s)",
            "value": 1968.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 TTFT (ms)",
            "value": 146.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 TPOT (ms)",
            "value": 17.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 throughput (tok/s)",
            "value": 342.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 Total Tput (tok/s)",
            "value": 386.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 TTFT (ms)",
            "value": 110.99,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 TPOT (ms)",
            "value": 11.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 throughput (tok/s)",
            "value": 632.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 Total Tput (tok/s)",
            "value": 712.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 TTFT (ms)",
            "value": 121.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 TPOT (ms)",
            "value": 12.29,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 throughput (tok/s)",
            "value": 83.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 Total Tput (tok/s)",
            "value": 747.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 TTFT (ms)",
            "value": 263.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 TPOT (ms)",
            "value": 11.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 throughput (tok/s)",
            "value": 2148.28,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 Total Tput (tok/s)",
            "value": 19394.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 TTFT (ms)",
            "value": 1876.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 TPOT (ms)",
            "value": 56.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 throughput (tok/s)",
            "value": 852.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 Total Tput (tok/s)",
            "value": 7693.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 TTFT (ms)",
            "value": 474.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 TPOT (ms)",
            "value": 17.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 throughput (tok/s)",
            "value": 161.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 Total Tput (tok/s)",
            "value": 1443.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 TTFT (ms)",
            "value": 300.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 TPOT (ms)",
            "value": 12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 throughput (tok/s)",
            "value": 1174.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 Total Tput (tok/s)",
            "value": 10505.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 TTFT (ms)",
            "value": 688.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 TPOT (ms)",
            "value": 25.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 throughput (tok/s)",
            "value": 297.28,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 Total Tput (tok/s)",
            "value": 2672.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 TTFT (ms)",
            "value": 320.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 TPOT (ms)",
            "value": 12.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 throughput (tok/s)",
            "value": 1656.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 Total Tput (tok/s)",
            "value": 14929.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 TTFT (ms)",
            "value": 1147.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 TPOT (ms)",
            "value": 36.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 throughput (tok/s)",
            "value": 460.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 Total Tput (tok/s)",
            "value": 4098.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 TTFT (ms)",
            "value": 1273.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 TPOT (ms)",
            "value": 15.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 throughput (tok/s)",
            "value": 4931.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 9889.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TTFT (ms)",
            "value": 1023.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TPOT (ms)",
            "value": 23.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 throughput (tok/s)",
            "value": 1453.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2926.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 TTFT (ms)",
            "value": 139.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 TPOT (ms)",
            "value": 10.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 throughput (tok/s)",
            "value": 7070.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 14157.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TTFT (ms)",
            "value": 546.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TPOT (ms)",
            "value": 34.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 throughput (tok/s)",
            "value": 2576.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 5153.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 TTFT (ms)",
            "value": 181.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 TPOT (ms)",
            "value": 11.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 throughput (tok/s)",
            "value": 619.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1247.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TTFT (ms)",
            "value": 95.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TPOT (ms)",
            "value": 6.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 throughput (tok/s)",
            "value": 3552.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 7117.98,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TTFT (ms)",
            "value": 247.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TPOT (ms)",
            "value": 17.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 throughput (tok/s)",
            "value": 984.18,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1963.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 TTFT (ms)",
            "value": 123.75,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 TPOT (ms)",
            "value": 7.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 throughput (tok/s)",
            "value": 1984.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 Total Tput (tok/s)",
            "value": 2233.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 TTFT (ms)",
            "value": 141.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 TPOT (ms)",
            "value": 7.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 throughput (tok/s)",
            "value": 3270.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 Total Tput (tok/s)",
            "value": 3676.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 TTFT (ms)",
            "value": 181.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 TPOT (ms)",
            "value": 9.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 throughput (tok/s)",
            "value": 690.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 Total Tput (tok/s)",
            "value": 778.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 TTFT (ms)",
            "value": 94.1,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 TPOT (ms)",
            "value": 5.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 throughput (tok/s)",
            "value": 1171.18,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 Total Tput (tok/s)",
            "value": 1318.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 TTFT (ms)",
            "value": 117.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 TPOT (ms)",
            "value": 6.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 throughput (tok/s)",
            "value": 2601.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 23552.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 TTFT (ms)",
            "value": 1917.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 TPOT (ms)",
            "value": 46.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 throughput (tok/s)",
            "value": 1188.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 10762.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 TTFT (ms)",
            "value": 512.42,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 TPOT (ms)",
            "value": 12.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 throughput (tok/s)",
            "value": 1616.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 14497.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 TTFT (ms)",
            "value": 683.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 TPOT (ms)",
            "value": 18.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 throughput (tok/s)",
            "value": 560.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 5052.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 TTFT (ms)",
            "value": 351.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 TPOT (ms)",
            "value": 6.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 throughput (tok/s)",
            "value": 2120.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 19168.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 TTFT (ms)",
            "value": 1176.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 TPOT (ms)",
            "value": 28.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 throughput (tok/s)",
            "value": 792.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 Total Tput (tok/s)",
            "value": 7066.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 TTFT (ms)",
            "value": 407.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 TPOT (ms)",
            "value": 9.41,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 throughput (tok/s)",
            "value": 43.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 Total Tput (tok/s)",
            "value": 87.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 TTFT (ms)",
            "value": 105.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 TPOT (ms)",
            "value": 22.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 throughput (tok/s)",
            "value": 2667.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 Total Tput (tok/s)",
            "value": 5341.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TTFT (ms)",
            "value": 415.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TPOT (ms)",
            "value": 46.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 throughput (tok/s)",
            "value": 566.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1138.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 TTFT (ms)",
            "value": 184.9,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 TPOT (ms)",
            "value": 27.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 throughput (tok/s)",
            "value": 85.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 Total Tput (tok/s)",
            "value": 168.93,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 TTFT (ms)",
            "value": 123.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 TPOT (ms)",
            "value": 23.28,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 throughput (tok/s)",
            "value": 3879.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 Total Tput (tok/s)",
            "value": 7755.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 TTFT (ms)",
            "value": 678.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 TPOT (ms)",
            "value": 63.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 throughput (tok/s)",
            "value": 978.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 Total Tput (tok/s)",
            "value": 1953.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 TTFT (ms)",
            "value": 224.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 TPOT (ms)",
            "value": 31.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 throughput (tok/s)",
            "value": 165.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 Total Tput (tok/s)",
            "value": 331.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 TTFT (ms)",
            "value": 139.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 TPOT (ms)",
            "value": 23.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 throughput (tok/s)",
            "value": 1639.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3279.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TTFT (ms)",
            "value": 296.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TPOT (ms)",
            "value": 37.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 throughput (tok/s)",
            "value": 318.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 Total Tput (tok/s)",
            "value": 635.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 TTFT (ms)",
            "value": 155.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 TPOT (ms)",
            "value": 24.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 throughput (tok/s)",
            "value": 43.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 Total Tput (tok/s)",
            "value": 49.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 TTFT (ms)",
            "value": 102.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 TPOT (ms)",
            "value": 22.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 throughput (tok/s)",
            "value": 570.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 Total Tput (tok/s)",
            "value": 641.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 TTFT (ms)",
            "value": 178.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 TPOT (ms)",
            "value": 27.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 throughput (tok/s)",
            "value": 84.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 Total Tput (tok/s)",
            "value": 94.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 TTFT (ms)",
            "value": 126.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 TPOT (ms)",
            "value": 23.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 throughput (tok/s)",
            "value": 1060.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 Total Tput (tok/s)",
            "value": 1192.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 TTFT (ms)",
            "value": 213.97,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 TPOT (ms)",
            "value": 29.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 throughput (tok/s)",
            "value": 162.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 Total Tput (tok/s)",
            "value": 183.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 TTFT (ms)",
            "value": 142.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 TPOT (ms)",
            "value": 24.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 throughput (tok/s)",
            "value": 304.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 Total Tput (tok/s)",
            "value": 342.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 TTFT (ms)",
            "value": 161.58,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 TPOT (ms)",
            "value": 25.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=1 throughput (tok/s)",
            "value": 255.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=1 Total Tput (tok/s)",
            "value": 509.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=1 TTFT (ms)",
            "value": 44.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=1 TPOT (ms)",
            "value": 3.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=1 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 throughput (tok/s)",
            "value": 8119.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 Total Tput (tok/s)",
            "value": 16256.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 TTFT (ms)",
            "value": 186.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 TPOT (ms)",
            "value": 15.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=16 throughput (tok/s)",
            "value": 2286.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=16 Total Tput (tok/s)",
            "value": 4596.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=16 TTFT (ms)",
            "value": 66.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=16 TPOT (ms)",
            "value": 6.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=2 throughput (tok/s)",
            "value": 494.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=2 Total Tput (tok/s)",
            "value": 979.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=2 TTFT (ms)",
            "value": 48.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=2 TPOT (ms)",
            "value": 3.99,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=2 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=256 throughput (tok/s)",
            "value": 11733.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=256 Total Tput (tok/s)",
            "value": 23456.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=256 TTFT (ms)",
            "value": 1104.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=256 TPOT (ms)",
            "value": 19.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=256 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=32 throughput (tok/s)",
            "value": 3613.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=32 Total Tput (tok/s)",
            "value": 7215.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=32 TTFT (ms)",
            "value": 84.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=32 TPOT (ms)",
            "value": 8.58,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 throughput (tok/s)",
            "value": 858.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1725.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 TTFT (ms)",
            "value": 46.41,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 TPOT (ms)",
            "value": 4.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 throughput (tok/s)",
            "value": 5564.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 Total Tput (tok/s)",
            "value": 11130.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 TTFT (ms)",
            "value": 132.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 TPOT (ms)",
            "value": 11.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=8 throughput (tok/s)",
            "value": 1455.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=8 Total Tput (tok/s)",
            "value": 2899.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=8 TTFT (ms)",
            "value": 47.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=8 TPOT (ms)",
            "value": 5.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=1 throughput (tok/s)",
            "value": 257.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=1 Total Tput (tok/s)",
            "value": 289.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=1 TTFT (ms)",
            "value": 46.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=1 TPOT (ms)",
            "value": 3.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=1 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=16 throughput (tok/s)",
            "value": 2444.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=16 Total Tput (tok/s)",
            "value": 2751.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=16 TTFT (ms)",
            "value": 68.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=16 TPOT (ms)",
            "value": 6.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=2 throughput (tok/s)",
            "value": 493.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=2 Total Tput (tok/s)",
            "value": 555.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=2 TTFT (ms)",
            "value": 51.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=2 TPOT (ms)",
            "value": 4.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=2 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=32 throughput (tok/s)",
            "value": 3832.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=32 Total Tput (tok/s)",
            "value": 4308.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=32 TTFT (ms)",
            "value": 84.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=32 TPOT (ms)",
            "value": 8.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=4 throughput (tok/s)",
            "value": 897.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=4 Total Tput (tok/s)",
            "value": 1012.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=4 TTFT (ms)",
            "value": 50.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=4 TPOT (ms)",
            "value": 4.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=8 throughput (tok/s)",
            "value": 1513.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=8 Total Tput (tok/s)",
            "value": 1702.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=8 TTFT (ms)",
            "value": 55.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=8 TPOT (ms)",
            "value": 5.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=1 throughput (tok/s)",
            "value": 245.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=1 Total Tput (tok/s)",
            "value": 2195.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=1 TTFT (ms)",
            "value": 143.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=1 TPOT (ms)",
            "value": 3.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=1 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=128 throughput (tok/s)",
            "value": 4489.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=128 Total Tput (tok/s)",
            "value": 40532.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=128 TTFT (ms)",
            "value": 900.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=128 TPOT (ms)",
            "value": 27.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=128 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=16 throughput (tok/s)",
            "value": 1899.93,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=16 Total Tput (tok/s)",
            "value": 17146.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=16 TTFT (ms)",
            "value": 252.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=16 TPOT (ms)",
            "value": 7.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=2 throughput (tok/s)",
            "value": 467.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=2 Total Tput (tok/s)",
            "value": 4162.93,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=2 TTFT (ms)",
            "value": 144.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=2 TPOT (ms)",
            "value": 4.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=2 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=32 throughput (tok/s)",
            "value": 2710.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=32 Total Tput (tok/s)",
            "value": 24240.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=32 TTFT (ms)",
            "value": 344.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=32 TPOT (ms)",
            "value": 11.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=4 throughput (tok/s)",
            "value": 793.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=4 Total Tput (tok/s)",
            "value": 7135.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=4 TTFT (ms)",
            "value": 158.42,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=4 TPOT (ms)",
            "value": 4.75,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=64 throughput (tok/s)",
            "value": 3648.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=64 Total Tput (tok/s)",
            "value": 32887.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=64 TTFT (ms)",
            "value": 547.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=64 TPOT (ms)",
            "value": 16.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=8 throughput (tok/s)",
            "value": 1309.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=8 Total Tput (tok/s)",
            "value": 11652.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=8 TTFT (ms)",
            "value": 191.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=8 TPOT (ms)",
            "value": 5.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23092471281"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "id": "d2d8136076a14a39a8db9677ab4c59109f2f969f",
          "message": "fix: remove || 'true' that swallowed explicit false in model enable check\n\nBoolean inputs.xxx is falsy when false, so `false || 'true'` always\nevaluates to 'true', making it impossible to disable a model via\nworkflow_dispatch. Input defaults already provide the true fallback.",
          "timestamp": "2026-03-15T02:48:55Z",
          "url": "https://github.com/ROCm/ATOM/commit/d2d8136076a14a39a8db9677ab4c59109f2f969f"
        },
        "date": 1773543539085,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 throughput (tok/s)",
            "value": 328.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 Total Tput (tok/s)",
            "value": 659.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TTFT (ms)",
            "value": 109.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TPOT (ms)",
            "value": 11.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 throughput (tok/s)",
            "value": 567.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1142.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TTFT (ms)",
            "value": 98.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TPOT (ms)",
            "value": 6.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 throughput (tok/s)",
            "value": 160,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 Total Tput (tok/s)",
            "value": 321.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 TTFT (ms)",
            "value": 139.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 TPOT (ms)",
            "value": 23.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 throughput (tok/s)",
            "value": 872.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1752.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 TTFT (ms)",
            "value": 49.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 TPOT (ms)",
            "value": 4.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23101832261"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "id": "0b4a27b3c37f79475cc5d70feea70cfb8ba75aa8",
          "message": "refactor: split regression-analysis into regression-rerun + profiler-analysis\n\n- regression-rerun (GPU): only on regression, re-runs configs with profiler\n- profiler-analysis (ubuntu-latest): runs on regression OR manual enable_profiler\n  - Regression mode: downloads traces from regression-rerun\n  - Manual mode: downloads traces from benchmark jobs\n  - Issue creation only for regression mode\n\nThis allows manual profiler data collection via workflow_dispatch with\nenable_profiler=true, without requiring an actual regression.",
          "timestamp": "2026-03-15T03:22:16Z",
          "url": "https://github.com/ROCm/ATOM/commit/0b4a27b3c37f79475cc5d70feea70cfb8ba75aa8"
        },
        "date": 1773545098403,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 throughput (tok/s)",
            "value": 331.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23102230111"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 Total Tput (tok/s)",
            "value": 666.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23102230111"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TTFT (ms)",
            "value": 133.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23102230111"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TPOT (ms)",
            "value": 11.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23102230111"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 throughput (tok/s)",
            "value": 583.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23102230111"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1174.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23102230111"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TTFT (ms)",
            "value": 284.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23102230111"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TPOT (ms)",
            "value": 6.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23102230111"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "24c76973ae282c1fc30f5f971bc058c9bee22a31",
          "message": "docs: update README, add DeepSeek/GPT-OSS recipes, fix stale branch refs (#328)\n\n* docs: update README, add DeepSeek/GPT-OSS recipes, fix stale branch refs\n\nREADME:\n- Add CI/Benchmark/Dashboard badges\n- Add Qwen3-Next to model table\n- Add MTP serving example\n- Add live benchmark dashboard section with feature list\n- Add profiler manual curl API usage\n- Add Env Vars to docs table\n- Add DeepSeek-R1, GPT-OSS, Qwen3-Next recipes and framework integration links\n\nNew recipes:\n- DeepSeek-R1: BF16/MXFP4/MTP3 configs, performance baselines, accuracy\n- GPT-OSS: single GPU and DP+EP configs\n\nRecipe fixes:\n- GLM-5: zlr/glm5 branch → main\n- Kimi-K2: guanbao/kimi_k2_fp4 branch → main\n- Qwen3-235b: specific commit refs → main\n- SGLang: update PR status, remove TODO\n- All: git checkout specific_branch_or_commit → main\n\nHousekeeping:\n- .gitignore: exclude .claude/plan/\n- Remove accidentally committed plan file\n\n* fix: docs deploy to /docs/ subdir, restore benchmark-dashboard on gh-pages",
          "timestamp": "2026-03-14T17:26:07Z",
          "url": "https://github.com/ROCm/ATOM/commit/24c76973ae282c1fc30f5f971bc058c9bee22a31"
        },
        "date": 1773614185530,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 throughput (tok/s)",
            "value": 79.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 Total Tput (tok/s)",
            "value": 158.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 TTFT (ms)",
            "value": 61.1,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 TPOT (ms)",
            "value": 12.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 throughput (tok/s)",
            "value": 4266.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 Total Tput (tok/s)",
            "value": 8543.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TTFT (ms)",
            "value": 315.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TPOT (ms)",
            "value": 28.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 throughput (tok/s)",
            "value": 1103.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2219.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 TTFT (ms)",
            "value": 143.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 TPOT (ms)",
            "value": 14.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 throughput (tok/s)",
            "value": 179.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 Total Tput (tok/s)",
            "value": 356.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 TTFT (ms)",
            "value": 93.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 TPOT (ms)",
            "value": 11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 throughput (tok/s)",
            "value": 5584.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 Total Tput (tok/s)",
            "value": 11164.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TTFT (ms)",
            "value": 1198.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TPOT (ms)",
            "value": 43.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 throughput (tok/s)",
            "value": 1654.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3304.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 TTFT (ms)",
            "value": 176.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 TPOT (ms)",
            "value": 18.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 throughput (tok/s)",
            "value": 335.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 Total Tput (tok/s)",
            "value": 673.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TTFT (ms)",
            "value": 105.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TPOT (ms)",
            "value": 11.44,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 throughput (tok/s)",
            "value": 2743.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5488.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TTFT (ms)",
            "value": 320.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TPOT (ms)",
            "value": 22.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 throughput (tok/s)",
            "value": 628.27,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1251.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 TTFT (ms)",
            "value": 121.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 TPOT (ms)",
            "value": 12.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 throughput (tok/s)",
            "value": 91.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 Total Tput (tok/s)",
            "value": 103.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 TTFT (ms)",
            "value": 67.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 TPOT (ms)",
            "value": 10.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 throughput (tok/s)",
            "value": 1130.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 Total Tput (tok/s)",
            "value": 1272.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 TTFT (ms)",
            "value": 152.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 TPOT (ms)",
            "value": 13.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 throughput (tok/s)",
            "value": 180.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 Total Tput (tok/s)",
            "value": 202.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 TTFT (ms)",
            "value": 94.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 TPOT (ms)",
            "value": 11.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 throughput (tok/s)",
            "value": 1735.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 Total Tput (tok/s)",
            "value": 1950.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 TTFT (ms)",
            "value": 166.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 TPOT (ms)",
            "value": 17.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 throughput (tok/s)",
            "value": 336.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 Total Tput (tok/s)",
            "value": 379.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 TTFT (ms)",
            "value": 110.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 TPOT (ms)",
            "value": 11.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 throughput (tok/s)",
            "value": 622.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 Total Tput (tok/s)",
            "value": 700.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 TTFT (ms)",
            "value": 104.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 TPOT (ms)",
            "value": 12.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 throughput (tok/s)",
            "value": 84.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 Total Tput (tok/s)",
            "value": 754.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 TTFT (ms)",
            "value": 271.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 TPOT (ms)",
            "value": 11.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 throughput (tok/s)",
            "value": 2139.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 Total Tput (tok/s)",
            "value": 19316.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 TTFT (ms)",
            "value": 1948.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 TPOT (ms)",
            "value": 56.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 throughput (tok/s)",
            "value": 850.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 Total Tput (tok/s)",
            "value": 7679.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 TTFT (ms)",
            "value": 481.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 TPOT (ms)",
            "value": 17.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 throughput (tok/s)",
            "value": 142.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 Total Tput (tok/s)",
            "value": 1265.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 TTFT (ms)",
            "value": 268.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 TPOT (ms)",
            "value": 13.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 throughput (tok/s)",
            "value": 1126.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 Total Tput (tok/s)",
            "value": 10071.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 TTFT (ms)",
            "value": 676.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 TPOT (ms)",
            "value": 27.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 throughput (tok/s)",
            "value": 296.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 Total Tput (tok/s)",
            "value": 2667.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 TTFT (ms)",
            "value": 418.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 TPOT (ms)",
            "value": 12.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 throughput (tok/s)",
            "value": 1687.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 Total Tput (tok/s)",
            "value": 15210.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 TTFT (ms)",
            "value": 1087.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 TPOT (ms)",
            "value": 36.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 throughput (tok/s)",
            "value": 534.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 Total Tput (tok/s)",
            "value": 4750.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 TTFT (ms)",
            "value": 386.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 TPOT (ms)",
            "value": 14.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 throughput (tok/s)",
            "value": 5188.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 10403.93,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TTFT (ms)",
            "value": 353.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TPOT (ms)",
            "value": 23.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 throughput (tok/s)",
            "value": 1429.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2878.27,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 TTFT (ms)",
            "value": 157.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 TPOT (ms)",
            "value": 10.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 throughput (tok/s)",
            "value": 6907.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 13830.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TTFT (ms)",
            "value": 549.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TPOT (ms)",
            "value": 34.9,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 throughput (tok/s)",
            "value": 2486.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 4972.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 TTFT (ms)",
            "value": 188.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 TPOT (ms)",
            "value": 12.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 throughput (tok/s)",
            "value": 410.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 826.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TTFT (ms)",
            "value": 97.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TPOT (ms)",
            "value": 9.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 throughput (tok/s)",
            "value": 3572.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 7158.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TTFT (ms)",
            "value": 245.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TPOT (ms)",
            "value": 17.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 throughput (tok/s)",
            "value": 934.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1865.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 TTFT (ms)",
            "value": 117.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 TPOT (ms)",
            "value": 8.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 throughput (tok/s)",
            "value": 1942.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 Total Tput (tok/s)",
            "value": 2185.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 TTFT (ms)",
            "value": 149.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 TPOT (ms)",
            "value": 7.97,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 throughput (tok/s)",
            "value": 3234.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 Total Tput (tok/s)",
            "value": 3636.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 TTFT (ms)",
            "value": 183.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 TPOT (ms)",
            "value": 9.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 throughput (tok/s)",
            "value": 690.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 Total Tput (tok/s)",
            "value": 778.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 TTFT (ms)",
            "value": 96.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 TPOT (ms)",
            "value": 5.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 throughput (tok/s)",
            "value": 1111.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 Total Tput (tok/s)",
            "value": 1250.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 TTFT (ms)",
            "value": 117.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 TPOT (ms)",
            "value": 7.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 throughput (tok/s)",
            "value": 2610.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 23637.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 TTFT (ms)",
            "value": 1906.28,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 TPOT (ms)",
            "value": 46.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 throughput (tok/s)",
            "value": 1189.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 10761.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 TTFT (ms)",
            "value": 522.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 TPOT (ms)",
            "value": 12.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 throughput (tok/s)",
            "value": 1533.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 13749.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 TTFT (ms)",
            "value": 872.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 TPOT (ms)",
            "value": 19.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 throughput (tok/s)",
            "value": 551.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 4972.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 TTFT (ms)",
            "value": 332.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 TPOT (ms)",
            "value": 6.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 throughput (tok/s)",
            "value": 2031.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 18359.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 TTFT (ms)",
            "value": 1314.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 TPOT (ms)",
            "value": 29.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 throughput (tok/s)",
            "value": 796.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 Total Tput (tok/s)",
            "value": 7102.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 TTFT (ms)",
            "value": 355.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 TPOT (ms)",
            "value": 9.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 throughput (tok/s)",
            "value": 47.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 Total Tput (tok/s)",
            "value": 94.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 TTFT (ms)",
            "value": 100.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 TPOT (ms)",
            "value": 21.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 throughput (tok/s)",
            "value": 2698.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 Total Tput (tok/s)",
            "value": 5403.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TTFT (ms)",
            "value": 397.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TPOT (ms)",
            "value": 45.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 throughput (tok/s)",
            "value": 570.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1147.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 TTFT (ms)",
            "value": 175.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 TPOT (ms)",
            "value": 27.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 throughput (tok/s)",
            "value": 91.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 Total Tput (tok/s)",
            "value": 180.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 TTFT (ms)",
            "value": 121.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 TPOT (ms)",
            "value": 21.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 throughput (tok/s)",
            "value": 3913.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 Total Tput (tok/s)",
            "value": 7822.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 TTFT (ms)",
            "value": 625.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 TPOT (ms)",
            "value": 63.14,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 throughput (tok/s)",
            "value": 978.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 Total Tput (tok/s)",
            "value": 1953.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 TTFT (ms)",
            "value": 231.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 TPOT (ms)",
            "value": 31.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 throughput (tok/s)",
            "value": 159.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 Total Tput (tok/s)",
            "value": 321.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 TTFT (ms)",
            "value": 139.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 TPOT (ms)",
            "value": 24.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 throughput (tok/s)",
            "value": 1686.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3374.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TTFT (ms)",
            "value": 284.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TPOT (ms)",
            "value": 36.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 throughput (tok/s)",
            "value": 316,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 Total Tput (tok/s)",
            "value": 629.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 TTFT (ms)",
            "value": 148.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 TPOT (ms)",
            "value": 24.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 throughput (tok/s)",
            "value": 44.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 Total Tput (tok/s)",
            "value": 49.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 TTFT (ms)",
            "value": 102.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 TPOT (ms)",
            "value": 22.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 throughput (tok/s)",
            "value": 556.6,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 Total Tput (tok/s)",
            "value": 626.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 TTFT (ms)",
            "value": 176.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 TPOT (ms)",
            "value": 27.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 throughput (tok/s)",
            "value": 84.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 Total Tput (tok/s)",
            "value": 94.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 TTFT (ms)",
            "value": 134.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 TPOT (ms)",
            "value": 23.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 throughput (tok/s)",
            "value": 1059.16,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 Total Tput (tok/s)",
            "value": 1190.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 TTFT (ms)",
            "value": 208.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 TPOT (ms)",
            "value": 29.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 throughput (tok/s)",
            "value": 159.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 Total Tput (tok/s)",
            "value": 179.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 TTFT (ms)",
            "value": 137.58,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 TPOT (ms)",
            "value": 24.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 throughput (tok/s)",
            "value": 298.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 Total Tput (tok/s)",
            "value": 335.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 TTFT (ms)",
            "value": 156.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 TPOT (ms)",
            "value": 26.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23115155923"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "id": "06fe50dc78a39b6706e2a00d26ee7767fbc8efb8",
          "message": "fix: address review findings and remove dead ATOM_GPT_OSS_MODEL env var\n\nReview fixes:\n- Fix performance_summary.md path in GitHub Issue (read from per-config\n  subdirectory instead of non-existent flat path)\n- Decouple Issue creation from regression-rerun success (file issues\n  even when GPU re-run job fails)\n- Add defensive .get() for isl/osl/conc in regression_rerun.py to\n  avoid KeyError on malformed regression entries\n\nCleanup:\n- Remove ATOM_GPT_OSS_MODEL=1 from atom-test.yaml and ci-pr-guide.md\n  (no code reads this env var; confirmed grep across ATOM, aiter, and\n  all installed Python packages)",
          "timestamp": "2026-03-16T14:53:19Z",
          "url": "https://github.com/ROCm/ATOM/commit/06fe50dc78a39b6706e2a00d26ee7767fbc8efb8"
        },
        "date": 1773674777168,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 throughput (tok/s)",
            "value": 4340.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 Total Tput (tok/s)",
            "value": 8690.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TTFT (ms)",
            "value": 324.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TPOT (ms)",
            "value": 28.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 throughput (tok/s)",
            "value": 5065.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 10157.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TTFT (ms)",
            "value": 360.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TPOT (ms)",
            "value": 23.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 throughput (tok/s)",
            "value": 2905.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 Total Tput (tok/s)",
            "value": 5817.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TTFT (ms)",
            "value": 353.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TPOT (ms)",
            "value": 42.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23150658270"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Thiago Pereira Rocha",
            "username": "thpereir",
            "email": "thiago.pereirarocha@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "d1bdbde820548e8c9e742b08c051b20917218e61",
          "message": "Fix Qwen3-Coder MXFP4 loading: fp4x2 copy_ and missing attention prefix (#321)",
          "timestamp": "2026-03-16T23:26:31Z",
          "url": "https://github.com/ROCm/ATOM/commit/d1bdbde820548e8c9e742b08c051b20917218e61"
        },
        "date": 1773736259371,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 throughput (tok/s)",
            "value": 97.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 Total Tput (tok/s)",
            "value": 194.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 TTFT (ms)",
            "value": 64.44,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 TPOT (ms)",
            "value": 10.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 throughput (tok/s)",
            "value": 4276.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 Total Tput (tok/s)",
            "value": 8561.81,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TTFT (ms)",
            "value": 319.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TPOT (ms)",
            "value": 28.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 throughput (tok/s)",
            "value": 1161.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2335.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 TTFT (ms)",
            "value": 134.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 TPOT (ms)",
            "value": 13.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 throughput (tok/s)",
            "value": 178.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 Total Tput (tok/s)",
            "value": 354.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 TTFT (ms)",
            "value": 77.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 TPOT (ms)",
            "value": 11.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 throughput (tok/s)",
            "value": 6348.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 Total Tput (tok/s)",
            "value": 12690.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TTFT (ms)",
            "value": 477.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TPOT (ms)",
            "value": 38.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 throughput (tok/s)",
            "value": 1751.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3497.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 TTFT (ms)",
            "value": 159.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 TPOT (ms)",
            "value": 17.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 throughput (tok/s)",
            "value": 350.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 Total Tput (tok/s)",
            "value": 704.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TTFT (ms)",
            "value": 91.44,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TPOT (ms)",
            "value": 10.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 throughput (tok/s)",
            "value": 2928.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5858.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TTFT (ms)",
            "value": 194.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TPOT (ms)",
            "value": 21.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 throughput (tok/s)",
            "value": 663.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1322.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 TTFT (ms)",
            "value": 111.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 TPOT (ms)",
            "value": 11.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 throughput (tok/s)",
            "value": 1147.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 Total Tput (tok/s)",
            "value": 1291.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 TTFT (ms)",
            "value": 112.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 TPOT (ms)",
            "value": 13.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 throughput (tok/s)",
            "value": 185.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 Total Tput (tok/s)",
            "value": 208.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 TTFT (ms)",
            "value": 88.99,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 TPOT (ms)",
            "value": 10.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 throughput (tok/s)",
            "value": 350.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 Total Tput (tok/s)",
            "value": 395.6,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 TTFT (ms)",
            "value": 91.28,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 TPOT (ms)",
            "value": 11.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 throughput (tok/s)",
            "value": 655.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 Total Tput (tok/s)",
            "value": 737.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 TTFT (ms)",
            "value": 114.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 TPOT (ms)",
            "value": 11.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 throughput (tok/s)",
            "value": 90.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 Total Tput (tok/s)",
            "value": 811.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 TTFT (ms)",
            "value": 270.99,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 TPOT (ms)",
            "value": 10.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 throughput (tok/s)",
            "value": 2247.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 Total Tput (tok/s)",
            "value": 20292.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 TTFT (ms)",
            "value": 1821.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 TPOT (ms)",
            "value": 54.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 throughput (tok/s)",
            "value": 899.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 Total Tput (tok/s)",
            "value": 8114.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 TTFT (ms)",
            "value": 459.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 TPOT (ms)",
            "value": 16.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 throughput (tok/s)",
            "value": 173.27,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 Total Tput (tok/s)",
            "value": 1544.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 TTFT (ms)",
            "value": 287.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 TPOT (ms)",
            "value": 11.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 throughput (tok/s)",
            "value": 1259.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 Total Tput (tok/s)",
            "value": 11262.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 TTFT (ms)",
            "value": 619.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 TPOT (ms)",
            "value": 24.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 throughput (tok/s)",
            "value": 316.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 Total Tput (tok/s)",
            "value": 2847.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 TTFT (ms)",
            "value": 310.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 TPOT (ms)",
            "value": 11.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 throughput (tok/s)",
            "value": 1788.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 Total Tput (tok/s)",
            "value": 16119.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 TTFT (ms)",
            "value": 1048.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 TPOT (ms)",
            "value": 34.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 throughput (tok/s)",
            "value": 571.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 Total Tput (tok/s)",
            "value": 5081.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 TTFT (ms)",
            "value": 341.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 TPOT (ms)",
            "value": 13.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 throughput (tok/s)",
            "value": 5375.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 10780.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TTFT (ms)",
            "value": 345.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TPOT (ms)",
            "value": 22.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 throughput (tok/s)",
            "value": 1623.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 3269.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 TTFT (ms)",
            "value": 143.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 TPOT (ms)",
            "value": 9.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 throughput (tok/s)",
            "value": 7332.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 14681.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TTFT (ms)",
            "value": 530.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TPOT (ms)",
            "value": 32.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 throughput (tok/s)",
            "value": 2577.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 5154.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 TTFT (ms)",
            "value": 182.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 TPOT (ms)",
            "value": 11.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 throughput (tok/s)",
            "value": 622.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1252.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TTFT (ms)",
            "value": 103.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TPOT (ms)",
            "value": 5.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 throughput (tok/s)",
            "value": 3743.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 7501.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TTFT (ms)",
            "value": 238.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TPOT (ms)",
            "value": 16.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 throughput (tok/s)",
            "value": 985.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1966.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 TTFT (ms)",
            "value": 116.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 TPOT (ms)",
            "value": 7.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 throughput (tok/s)",
            "value": 1999.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 Total Tput (tok/s)",
            "value": 2250.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 TTFT (ms)",
            "value": 147.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 TPOT (ms)",
            "value": 7.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 throughput (tok/s)",
            "value": 1175.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 Total Tput (tok/s)",
            "value": 1322.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 TTFT (ms)",
            "value": 120.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 TPOT (ms)",
            "value": 6.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 throughput (tok/s)",
            "value": 2612.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 23653.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 TTFT (ms)",
            "value": 1876.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 TPOT (ms)",
            "value": 45.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 throughput (tok/s)",
            "value": 1269.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 11487.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 TTFT (ms)",
            "value": 473.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 TPOT (ms)",
            "value": 11.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 throughput (tok/s)",
            "value": 1740.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 15609.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 TTFT (ms)",
            "value": 699.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 TPOT (ms)",
            "value": 17.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 throughput (tok/s)",
            "value": 596.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 5376.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 TTFT (ms)",
            "value": 318.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 TPOT (ms)",
            "value": 6.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 throughput (tok/s)",
            "value": 2158.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 19511.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 TTFT (ms)",
            "value": 1040.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 TPOT (ms)",
            "value": 27.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 throughput (tok/s)",
            "value": 862.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 Total Tput (tok/s)",
            "value": 7686.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 TTFT (ms)",
            "value": 394.82,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 TPOT (ms)",
            "value": 8.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 throughput (tok/s)",
            "value": 47.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 Total Tput (tok/s)",
            "value": 93.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 TTFT (ms)",
            "value": 100.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 TPOT (ms)",
            "value": 21.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 throughput (tok/s)",
            "value": 2904.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 Total Tput (tok/s)",
            "value": 5816.16,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TTFT (ms)",
            "value": 372.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TPOT (ms)",
            "value": 42.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 throughput (tok/s)",
            "value": 607.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1221.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 TTFT (ms)",
            "value": 152.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 TPOT (ms)",
            "value": 25.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 throughput (tok/s)",
            "value": 95.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 Total Tput (tok/s)",
            "value": 188.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 TTFT (ms)",
            "value": 111.33,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 TPOT (ms)",
            "value": 20.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 throughput (tok/s)",
            "value": 4242.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 Total Tput (tok/s)",
            "value": 8480.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 TTFT (ms)",
            "value": 603.1,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 TPOT (ms)",
            "value": 58.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 throughput (tok/s)",
            "value": 1045.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2088.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 TTFT (ms)",
            "value": 186.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 TPOT (ms)",
            "value": 29.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 throughput (tok/s)",
            "value": 1808.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3618.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TTFT (ms)",
            "value": 249.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TPOT (ms)",
            "value": 33.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 throughput (tok/s)",
            "value": 323.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 Total Tput (tok/s)",
            "value": 643.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 TTFT (ms)",
            "value": 127.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 TPOT (ms)",
            "value": 24.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 throughput (tok/s)",
            "value": 45.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 Total Tput (tok/s)",
            "value": 50.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 TTFT (ms)",
            "value": 99.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 TPOT (ms)",
            "value": 22.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 throughput (tok/s)",
            "value": 582.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 Total Tput (tok/s)",
            "value": 655.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 TTFT (ms)",
            "value": 153.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 TPOT (ms)",
            "value": 26.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 throughput (tok/s)",
            "value": 87.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 Total Tput (tok/s)",
            "value": 99.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 TTFT (ms)",
            "value": 114.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 TPOT (ms)",
            "value": 22.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 throughput (tok/s)",
            "value": 1082.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 Total Tput (tok/s)",
            "value": 1216.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 TTFT (ms)",
            "value": 180.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 TPOT (ms)",
            "value": 28.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 throughput (tok/s)",
            "value": 166.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 Total Tput (tok/s)",
            "value": 187.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 TTFT (ms)",
            "value": 116.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 TPOT (ms)",
            "value": 23.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 throughput (tok/s)",
            "value": 44.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 Total Tput (tok/s)",
            "value": 402.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 TTFT (ms)",
            "value": 714.42,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 TPOT (ms)",
            "value": 21.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 throughput (tok/s)",
            "value": 1373.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 Total Tput (tok/s)",
            "value": 12401.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 TTFT (ms)",
            "value": 3886.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 TPOT (ms)",
            "value": 87.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 throughput (tok/s)",
            "value": 450.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 Total Tput (tok/s)",
            "value": 4064.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 TTFT (ms)",
            "value": 1102.14,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 TPOT (ms)",
            "value": 33.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 throughput (tok/s)",
            "value": 84.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 Total Tput (tok/s)",
            "value": 749.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 TTFT (ms)",
            "value": 839.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 TPOT (ms)",
            "value": 22.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 throughput (tok/s)",
            "value": 733.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 Total Tput (tok/s)",
            "value": 6555.93,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 TTFT (ms)",
            "value": 1524.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 TPOT (ms)",
            "value": 40.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 throughput (tok/s)",
            "value": 152.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 Total Tput (tok/s)",
            "value": 1374.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 TTFT (ms)",
            "value": 802.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 TPOT (ms)",
            "value": 24.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 throughput (tok/s)",
            "value": 1030.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 Total Tput (tok/s)",
            "value": 9291.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 TTFT (ms)",
            "value": 2482.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 TPOT (ms)",
            "value": 58.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23156952997"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "ZhangLirong",
            "username": "ZhangLirong-amd",
            "email": "lirzhang@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "eca27526c649af78992b56a01ee83a3b34542657",
          "message": "log only once for MLA head padding (#345)\n\nCo-authored-by: root <root@hjbog-srdc-39.amd.com>",
          "timestamp": "2026-03-17T13:40:28Z",
          "url": "https://github.com/ROCm/ATOM/commit/eca27526c649af78992b56a01ee83a3b34542657"
        },
        "date": 1773782534596,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 throughput (tok/s)",
            "value": 97.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 Total Tput (tok/s)",
            "value": 195.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 TTFT (ms)",
            "value": 58.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 TPOT (ms)",
            "value": 10.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 throughput (tok/s)",
            "value": 4498.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 Total Tput (tok/s)",
            "value": 9007.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TTFT (ms)",
            "value": 320.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TPOT (ms)",
            "value": 27.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 throughput (tok/s)",
            "value": 1065.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2142.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 TTFT (ms)",
            "value": 130.84,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 TPOT (ms)",
            "value": 14.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 throughput (tok/s)",
            "value": 189.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 Total Tput (tok/s)",
            "value": 375.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 TTFT (ms)",
            "value": 86.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 TPOT (ms)",
            "value": 10.44,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 throughput (tok/s)",
            "value": 6358.64,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 Total Tput (tok/s)",
            "value": 12711.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TTFT (ms)",
            "value": 525.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TPOT (ms)",
            "value": 38.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 throughput (tok/s)",
            "value": 1678.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3350.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 TTFT (ms)",
            "value": 159.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 TPOT (ms)",
            "value": 18.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 throughput (tok/s)",
            "value": 350.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 Total Tput (tok/s)",
            "value": 705.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TTFT (ms)",
            "value": 90.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TPOT (ms)",
            "value": 10.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 throughput (tok/s)",
            "value": 2627.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5256.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TTFT (ms)",
            "value": 201.23,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TPOT (ms)",
            "value": 23.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 throughput (tok/s)",
            "value": 662.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1319.99,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 TTFT (ms)",
            "value": 106.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 TPOT (ms)",
            "value": 11.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 throughput (tok/s)",
            "value": 95.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 Total Tput (tok/s)",
            "value": 107.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 TTFT (ms)",
            "value": 69.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 TPOT (ms)",
            "value": 10.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 throughput (tok/s)",
            "value": 1152.05,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 Total Tput (tok/s)",
            "value": 1296.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 TTFT (ms)",
            "value": 110.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 TPOT (ms)",
            "value": 13.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 throughput (tok/s)",
            "value": 184.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 Total Tput (tok/s)",
            "value": 208.16,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 TTFT (ms)",
            "value": 88.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 TPOT (ms)",
            "value": 10.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 throughput (tok/s)",
            "value": 1799.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 Total Tput (tok/s)",
            "value": 2022.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 TTFT (ms)",
            "value": 135.98,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 TPOT (ms)",
            "value": 17.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 throughput (tok/s)",
            "value": 350.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 Total Tput (tok/s)",
            "value": 395.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 TTFT (ms)",
            "value": 106.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 TPOT (ms)",
            "value": 11.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 throughput (tok/s)",
            "value": 651.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 Total Tput (tok/s)",
            "value": 733.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 TTFT (ms)",
            "value": 110.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 TPOT (ms)",
            "value": 11.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 throughput (tok/s)",
            "value": 82.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 Total Tput (tok/s)",
            "value": 737.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 TTFT (ms)",
            "value": 271.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 TPOT (ms)",
            "value": 11.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 throughput (tok/s)",
            "value": 2235.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 Total Tput (tok/s)",
            "value": 20184.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 TTFT (ms)",
            "value": 1829.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 TPOT (ms)",
            "value": 54.42,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 throughput (tok/s)",
            "value": 902.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 Total Tput (tok/s)",
            "value": 8146.28,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 TTFT (ms)",
            "value": 462.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 TPOT (ms)",
            "value": 16.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 throughput (tok/s)",
            "value": 174.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 Total Tput (tok/s)",
            "value": 1552.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 TTFT (ms)",
            "value": 285.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 TPOT (ms)",
            "value": 11.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 throughput (tok/s)",
            "value": 1257.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 Total Tput (tok/s)",
            "value": 11247.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 TTFT (ms)",
            "value": 656.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 TPOT (ms)",
            "value": 24.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 throughput (tok/s)",
            "value": 316.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 Total Tput (tok/s)",
            "value": 2848.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 TTFT (ms)",
            "value": 285.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 TPOT (ms)",
            "value": 12.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 throughput (tok/s)",
            "value": 1752.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 Total Tput (tok/s)",
            "value": 15796.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 TTFT (ms)",
            "value": 1059.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 TPOT (ms)",
            "value": 34.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 throughput (tok/s)",
            "value": 548.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 Total Tput (tok/s)",
            "value": 4875.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 TTFT (ms)",
            "value": 508.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 TPOT (ms)",
            "value": 13.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 throughput (tok/s)",
            "value": 5380.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 10789.58,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TTFT (ms)",
            "value": 336.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TPOT (ms)",
            "value": 22.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 throughput (tok/s)",
            "value": 1528.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 3078.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 TTFT (ms)",
            "value": 147.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 TPOT (ms)",
            "value": 9.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 throughput (tok/s)",
            "value": 7320.68,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 14658.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TTFT (ms)",
            "value": 537.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TPOT (ms)",
            "value": 32.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 throughput (tok/s)",
            "value": 2580.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 5161.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 TTFT (ms)",
            "value": 184.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 TPOT (ms)",
            "value": 11.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 throughput (tok/s)",
            "value": 645.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1299.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TTFT (ms)",
            "value": 96.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TPOT (ms)",
            "value": 5.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 throughput (tok/s)",
            "value": 3738.27,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 7489.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TTFT (ms)",
            "value": 242.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TPOT (ms)",
            "value": 16.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 throughput (tok/s)",
            "value": 964.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1924.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 TTFT (ms)",
            "value": 119.68,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 TPOT (ms)",
            "value": 7.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 throughput (tok/s)",
            "value": 1977.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 Total Tput (tok/s)",
            "value": 2225.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 TTFT (ms)",
            "value": 146.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 TPOT (ms)",
            "value": 7.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 throughput (tok/s)",
            "value": 3301.33,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 Total Tput (tok/s)",
            "value": 3711.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 TTFT (ms)",
            "value": 178.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 TPOT (ms)",
            "value": 9.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 throughput (tok/s)",
            "value": 744.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 Total Tput (tok/s)",
            "value": 839.05,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 TTFT (ms)",
            "value": 99.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 TPOT (ms)",
            "value": 5.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 throughput (tok/s)",
            "value": 1144.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 Total Tput (tok/s)",
            "value": 1287.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 TTFT (ms)",
            "value": 118.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 TPOT (ms)",
            "value": 6.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 throughput (tok/s)",
            "value": 2614.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 23665.64,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 TTFT (ms)",
            "value": 1876.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 TPOT (ms)",
            "value": 45.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 throughput (tok/s)",
            "value": 1269.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 11491.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 TTFT (ms)",
            "value": 467,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 TPOT (ms)",
            "value": 11.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 throughput (tok/s)",
            "value": 1666.52,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 14945.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 TTFT (ms)",
            "value": 717.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 TPOT (ms)",
            "value": 18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 throughput (tok/s)",
            "value": 561.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 5061.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 TTFT (ms)",
            "value": 338.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 TPOT (ms)",
            "value": 6.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 throughput (tok/s)",
            "value": 2155.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 19483.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 TTFT (ms)",
            "value": 1096.1,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 TPOT (ms)",
            "value": 27.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 throughput (tok/s)",
            "value": 861.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 Total Tput (tok/s)",
            "value": 7685.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 TTFT (ms)",
            "value": 397.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 TPOT (ms)",
            "value": 8.53,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 throughput (tok/s)",
            "value": 44.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 Total Tput (tok/s)",
            "value": 88.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 TTFT (ms)",
            "value": 100.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 TPOT (ms)",
            "value": 22.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 throughput (tok/s)",
            "value": 2725.44,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 Total Tput (tok/s)",
            "value": 5456.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TTFT (ms)",
            "value": 371.14,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TPOT (ms)",
            "value": 45.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 throughput (tok/s)",
            "value": 606.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1220.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 TTFT (ms)",
            "value": 143.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 TPOT (ms)",
            "value": 25.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 throughput (tok/s)",
            "value": 95.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 Total Tput (tok/s)",
            "value": 188.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 TTFT (ms)",
            "value": 111.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 TPOT (ms)",
            "value": 20.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 throughput (tok/s)",
            "value": 4231.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 Total Tput (tok/s)",
            "value": 8459.64,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 TTFT (ms)",
            "value": 581.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 TPOT (ms)",
            "value": 58.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 throughput (tok/s)",
            "value": 1050.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2098.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 TTFT (ms)",
            "value": 190.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 TPOT (ms)",
            "value": 29.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 throughput (tok/s)",
            "value": 177.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 Total Tput (tok/s)",
            "value": 357.41,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 TTFT (ms)",
            "value": 119.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 TPOT (ms)",
            "value": 21.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 throughput (tok/s)",
            "value": 1804.42,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3609.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TTFT (ms)",
            "value": 250.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TPOT (ms)",
            "value": 34.01,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 throughput (tok/s)",
            "value": 319.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 Total Tput (tok/s)",
            "value": 635.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 TTFT (ms)",
            "value": 129.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 TPOT (ms)",
            "value": 24.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 throughput (tok/s)",
            "value": 45.18,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 Total Tput (tok/s)",
            "value": 50.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 TTFT (ms)",
            "value": 99.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 TPOT (ms)",
            "value": 22.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 throughput (tok/s)",
            "value": 582.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 Total Tput (tok/s)",
            "value": 655.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 TTFT (ms)",
            "value": 154.06,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 TPOT (ms)",
            "value": 26.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 throughput (tok/s)",
            "value": 87.82,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 Total Tput (tok/s)",
            "value": 98.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 TTFT (ms)",
            "value": 113.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 TPOT (ms)",
            "value": 22.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 throughput (tok/s)",
            "value": 1083.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 Total Tput (tok/s)",
            "value": 1217.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 TTFT (ms)",
            "value": 183.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 TPOT (ms)",
            "value": 28.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 throughput (tok/s)",
            "value": 166.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 Total Tput (tok/s)",
            "value": 187.71,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 TTFT (ms)",
            "value": 116.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 TPOT (ms)",
            "value": 23.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 throughput (tok/s)",
            "value": 306.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 Total Tput (tok/s)",
            "value": 345.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 TTFT (ms)",
            "value": 150.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 TPOT (ms)",
            "value": 25.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 throughput (tok/s)",
            "value": 44.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 Total Tput (tok/s)",
            "value": 394.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 TTFT (ms)",
            "value": 1121.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 TPOT (ms)",
            "value": 21.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 throughput (tok/s)",
            "value": 1368.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 Total Tput (tok/s)",
            "value": 12354.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 TTFT (ms)",
            "value": 3900.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 TPOT (ms)",
            "value": 87.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 throughput (tok/s)",
            "value": 411.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 Total Tput (tok/s)",
            "value": 3711.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 TTFT (ms)",
            "value": 1129.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 TPOT (ms)",
            "value": 36.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 throughput (tok/s)",
            "value": 85.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 Total Tput (tok/s)",
            "value": 758.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 TTFT (ms)",
            "value": 702.89,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 TPOT (ms)",
            "value": 22.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 throughput (tok/s)",
            "value": 726.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 Total Tput (tok/s)",
            "value": 6494.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 TTFT (ms)",
            "value": 1492.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 TPOT (ms)",
            "value": 41.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 throughput (tok/s)",
            "value": 137.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 Total Tput (tok/s)",
            "value": 1238.18,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 TTFT (ms)",
            "value": 829.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 TPOT (ms)",
            "value": 27.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 throughput (tok/s)",
            "value": 1017.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 Total Tput (tok/s)",
            "value": 9174.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 TTFT (ms)",
            "value": 2625.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 TPOT (ms)",
            "value": 58.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=8 throughput (tok/s)",
            "value": 272.64,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=8 Total Tput (tok/s)",
            "value": 2425.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=8 TTFT (ms)",
            "value": 851.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=8 TPOT (ms)",
            "value": 27.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23207450768"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Lingpeng Jin",
            "username": "valarLip",
            "email": "103567126+valarLip@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "7addbf15fcc5fafad5039e68f23f711a88f860a5",
          "message": "feat: add DeepSeek-R1-0528 MXFP4-MTP3 to nightly benchmark (#353)\n\n* feat: add DeepSeek-R1-0528 MXFP4-MTP3 to nightly benchmark\n\nAdd amd/DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8 with MTP3\nspeculative decoding to the benchmark matrix. Uses existing suffix\n\"-mtp3\" exclude rule to skip low-concurrency runs.\n\nLocal benchmark (MI300X, 1k/1k, conc=64):\n  Total Token throughput: 11100 tok/s\n  Mean TPOT: 10.90 ms\n\n* war\n\n* fix black\n\n* add --use-chat-template\n\n* fix black",
          "timestamp": "2026-03-18T16:30:11Z",
          "url": "https://github.com/ROCm/ATOM/commit/7addbf15fcc5fafad5039e68f23f711a88f860a5"
        },
        "date": 1773870385823,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 throughput (tok/s)",
            "value": 97.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 Total Tput (tok/s)",
            "value": 195.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 TTFT (ms)",
            "value": 65.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 TPOT (ms)",
            "value": 10.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 throughput (tok/s)",
            "value": 4353.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 Total Tput (tok/s)",
            "value": 8716.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TTFT (ms)",
            "value": 285.07,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 TPOT (ms)",
            "value": 28.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 throughput (tok/s)",
            "value": 960.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1932.05,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 TTFT (ms)",
            "value": 124.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 TPOT (ms)",
            "value": 16.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 throughput (tok/s)",
            "value": 157.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 Total Tput (tok/s)",
            "value": 312.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 TTFT (ms)",
            "value": 87.12,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 TPOT (ms)",
            "value": 12.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 throughput (tok/s)",
            "value": 6206.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 Total Tput (tok/s)",
            "value": 12407.03,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TTFT (ms)",
            "value": 535.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 TPOT (ms)",
            "value": 39.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 throughput (tok/s)",
            "value": 1691.56,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 Total Tput (tok/s)",
            "value": 3377.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 TTFT (ms)",
            "value": 158.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 TPOT (ms)",
            "value": 18.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 throughput (tok/s)",
            "value": 331.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 Total Tput (tok/s)",
            "value": 666.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TTFT (ms)",
            "value": 106.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 TPOT (ms)",
            "value": 11.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 throughput (tok/s)",
            "value": 2923.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 Total Tput (tok/s)",
            "value": 5848.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TTFT (ms)",
            "value": 199.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 TPOT (ms)",
            "value": 21.04,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 throughput (tok/s)",
            "value": 662.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1319.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 TTFT (ms)",
            "value": 121.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 TPOT (ms)",
            "value": 11.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 throughput (tok/s)",
            "value": 95.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 Total Tput (tok/s)",
            "value": 106.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 TTFT (ms)",
            "value": 68.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 TPOT (ms)",
            "value": 10.51,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 throughput (tok/s)",
            "value": 1144.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 Total Tput (tok/s)",
            "value": 1288.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 TTFT (ms)",
            "value": 112.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 TPOT (ms)",
            "value": 13.58,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 throughput (tok/s)",
            "value": 184.34,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 Total Tput (tok/s)",
            "value": 207.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 TTFT (ms)",
            "value": 86.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 TPOT (ms)",
            "value": 10.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 throughput (tok/s)",
            "value": 1803.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 Total Tput (tok/s)",
            "value": 2027.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 TTFT (ms)",
            "value": 134.69,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 TPOT (ms)",
            "value": 17.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 throughput (tok/s)",
            "value": 346.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 Total Tput (tok/s)",
            "value": 390.64,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 TTFT (ms)",
            "value": 105.88,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 TPOT (ms)",
            "value": 11.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 throughput (tok/s)",
            "value": 656.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 Total Tput (tok/s)",
            "value": 739.24,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 TTFT (ms)",
            "value": 108.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 TPOT (ms)",
            "value": 11.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 throughput (tok/s)",
            "value": 91.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 Total Tput (tok/s)",
            "value": 819.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 TTFT (ms)",
            "value": 267.54,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 TPOT (ms)",
            "value": 10.65,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 throughput (tok/s)",
            "value": 2221.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 Total Tput (tok/s)",
            "value": 20055.65,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 TTFT (ms)",
            "value": 1878.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 TPOT (ms)",
            "value": 54.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 throughput (tok/s)",
            "value": 821.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 Total Tput (tok/s)",
            "value": 7417.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 TTFT (ms)",
            "value": 472.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 TPOT (ms)",
            "value": 18.41,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 throughput (tok/s)",
            "value": 174.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 Total Tput (tok/s)",
            "value": 1551.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 TTFT (ms)",
            "value": 284.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 TPOT (ms)",
            "value": 11.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 throughput (tok/s)",
            "value": 1222.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 Total Tput (tok/s)",
            "value": 10934.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 TTFT (ms)",
            "value": 676.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 TPOT (ms)",
            "value": 24.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 throughput (tok/s)",
            "value": 305.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 Total Tput (tok/s)",
            "value": 2745.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 TTFT (ms)",
            "value": 309.52,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 TPOT (ms)",
            "value": 12.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 throughput (tok/s)",
            "value": 1787.21,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 Total Tput (tok/s)",
            "value": 16109.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 TTFT (ms)",
            "value": 1063.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 TPOT (ms)",
            "value": 34.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 throughput (tok/s)",
            "value": 570.78,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 Total Tput (tok/s)",
            "value": 5077.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 TTFT (ms)",
            "value": 340.19,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 TPOT (ms)",
            "value": 13.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 throughput (tok/s)",
            "value": 5337.62,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 10703.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TTFT (ms)",
            "value": 343.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 TPOT (ms)",
            "value": 22.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 throughput (tok/s)",
            "value": 1378.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 2775.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 TTFT (ms)",
            "value": 173.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 TPOT (ms)",
            "value": 11.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 throughput (tok/s)",
            "value": 7104.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 14225.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TTFT (ms)",
            "value": 543.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 TPOT (ms)",
            "value": 33.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 throughput (tok/s)",
            "value": 2515.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 5031.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 TTFT (ms)",
            "value": 184.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 TPOT (ms)",
            "value": 12.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 throughput (tok/s)",
            "value": 531.96,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1070.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TTFT (ms)",
            "value": 93.03,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 TPOT (ms)",
            "value": 7.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 throughput (tok/s)",
            "value": 3727.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 7467.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TTFT (ms)",
            "value": 246.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 TPOT (ms)",
            "value": 16.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 throughput (tok/s)",
            "value": 917.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 1830.17,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 TTFT (ms)",
            "value": 122.29,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 TPOT (ms)",
            "value": 8.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 throughput (tok/s)",
            "value": 2015.92,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 Total Tput (tok/s)",
            "value": 2268.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 TTFT (ms)",
            "value": 146.14,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 TPOT (ms)",
            "value": 7.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 throughput (tok/s)",
            "value": 3176.5,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 Total Tput (tok/s)",
            "value": 3571.02,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 TTFT (ms)",
            "value": 181.42,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 TPOT (ms)",
            "value": 9.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 throughput (tok/s)",
            "value": 727.28,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 Total Tput (tok/s)",
            "value": 820.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 TTFT (ms)",
            "value": 95.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 TPOT (ms)",
            "value": 5.41,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 throughput (tok/s)",
            "value": 1151.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 Total Tput (tok/s)",
            "value": 1296.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 TTFT (ms)",
            "value": 117.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 TPOT (ms)",
            "value": 6.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 throughput (tok/s)",
            "value": 2594.38,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 23488.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 TTFT (ms)",
            "value": 1878.38,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 TPOT (ms)",
            "value": 46.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 throughput (tok/s)",
            "value": 1205.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 10910.35,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 TTFT (ms)",
            "value": 512.02,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 TPOT (ms)",
            "value": 12.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 throughput (tok/s)",
            "value": 1736.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 15572.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 TTFT (ms)",
            "value": 691.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 TPOT (ms)",
            "value": 17.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 throughput (tok/s)",
            "value": 500.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 4512.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 TTFT (ms)",
            "value": 331.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 TPOT (ms)",
            "value": 7.34,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 throughput (tok/s)",
            "value": 2173.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 19647.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 TTFT (ms)",
            "value": 1098.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 TPOT (ms)",
            "value": 27.56,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 throughput (tok/s)",
            "value": 827.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 Total Tput (tok/s)",
            "value": 7381.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 TTFT (ms)",
            "value": 431.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 TPOT (ms)",
            "value": 8.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-mtp3 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=128 throughput (tok/s)",
            "value": 5180.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=128 Total Tput (tok/s)",
            "value": 10387.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=128 TTFT (ms)",
            "value": 586.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=128 TPOT (ms)",
            "value": 22.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=16 throughput (tok/s)",
            "value": 1983.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=16 Total Tput (tok/s)",
            "value": 3993.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=16 TTFT (ms)",
            "value": 162.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=16 TPOT (ms)",
            "value": 7.6,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=256 throughput (tok/s)",
            "value": 6938.77,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=256 Total Tput (tok/s)",
            "value": 13892.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=256 TTFT (ms)",
            "value": 655.75,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=256 TPOT (ms)",
            "value": 34.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=32 throughput (tok/s)",
            "value": 2639.75,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=32 Total Tput (tok/s)",
            "value": 5279.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=32 TTFT (ms)",
            "value": 256.26,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=32 TPOT (ms)",
            "value": 11.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=4 throughput (tok/s)",
            "value": 719.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1447.4,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=4 TTFT (ms)",
            "value": 125.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=4 TPOT (ms)",
            "value": 5.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=64 throughput (tok/s)",
            "value": 4346.3,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=64 Total Tput (tok/s)",
            "value": 8708.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=64 TTFT (ms)",
            "value": 280.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=64 TPOT (ms)",
            "value": 13.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=8 throughput (tok/s)",
            "value": 1230.09,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=8 Total Tput (tok/s)",
            "value": 2455.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=8 TTFT (ms)",
            "value": 136.2,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=8 TPOT (ms)",
            "value": 6.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=16 throughput (tok/s)",
            "value": 2385.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=16 Total Tput (tok/s)",
            "value": 2685.13,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=16 TTFT (ms)",
            "value": 159.08,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=16 TPOT (ms)",
            "value": 6.41,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=32 throughput (tok/s)",
            "value": 3802.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=32 Total Tput (tok/s)",
            "value": 4275.08,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=32 TTFT (ms)",
            "value": 227.63,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=32 TPOT (ms)",
            "value": 8.09,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=4 throughput (tok/s)",
            "value": 868.98,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=4 Total Tput (tok/s)",
            "value": 979.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=4 TTFT (ms)",
            "value": 125.85,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=4 TPOT (ms)",
            "value": 4.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=8 throughput (tok/s)",
            "value": 1418.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=8 Total Tput (tok/s)",
            "value": 1596.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=8 TTFT (ms)",
            "value": 141.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=8 TPOT (ms)",
            "value": 5.37,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=128 throughput (tok/s)",
            "value": 2342.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=128 Total Tput (tok/s)",
            "value": 21206.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=128 TTFT (ms)",
            "value": 2338.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=128 TPOT (ms)",
            "value": 51.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=16 throughput (tok/s)",
            "value": 1198.94,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=16 Total Tput (tok/s)",
            "value": 10853.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=16 TTFT (ms)",
            "value": 658.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=16 TPOT (ms)",
            "value": 12.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=32 throughput (tok/s)",
            "value": 1584.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=32 Total Tput (tok/s)",
            "value": 14209.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=32 TTFT (ms)",
            "value": 892.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=32 TPOT (ms)",
            "value": 18.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=4 throughput (tok/s)",
            "value": 616.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=4 Total Tput (tok/s)",
            "value": 5553.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=4 TTFT (ms)",
            "value": 433.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=4 TPOT (ms)",
            "value": 5.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=64 throughput (tok/s)",
            "value": 1521.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=64 Total Tput (tok/s)",
            "value": 13753.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=64 TTFT (ms)",
            "value": 1295.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=64 TPOT (ms)",
            "value": 39.18,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=8 throughput (tok/s)",
            "value": 766.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=8 Total Tput (tok/s)",
            "value": 6841.69,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=8 TTFT (ms)",
            "value": 567.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=8 TPOT (ms)",
            "value": 9.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "DeepSeek-R1-0528-MoE-MXFP4-Attn-MTP-PTPC-FP8-mtp3 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 throughput (tok/s)",
            "value": 43.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 Total Tput (tok/s)",
            "value": 86.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 TTFT (ms)",
            "value": 99.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 TPOT (ms)",
            "value": 22.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 throughput (tok/s)",
            "value": 2891.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 Total Tput (tok/s)",
            "value": 5789.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TTFT (ms)",
            "value": 370.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 TPOT (ms)",
            "value": 42.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 throughput (tok/s)",
            "value": 576.95,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 Total Tput (tok/s)",
            "value": 1160.04,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 TTFT (ms)",
            "value": 157.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 TPOT (ms)",
            "value": 26.95,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 throughput (tok/s)",
            "value": 84.63,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 Total Tput (tok/s)",
            "value": 167.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 TTFT (ms)",
            "value": 113.67,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 TPOT (ms)",
            "value": 23.46,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 throughput (tok/s)",
            "value": 4130.59,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 Total Tput (tok/s)",
            "value": 8257.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 TTFT (ms)",
            "value": 624.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 TPOT (ms)",
            "value": 59.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=256 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 throughput (tok/s)",
            "value": 1019.83,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 Total Tput (tok/s)",
            "value": 2036.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 TTFT (ms)",
            "value": 197.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 TPOT (ms)",
            "value": 30.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 throughput (tok/s)",
            "value": 170.39,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 Total Tput (tok/s)",
            "value": 342.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 TTFT (ms)",
            "value": 114.4,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 TPOT (ms)",
            "value": 22.64,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 throughput (tok/s)",
            "value": 1720.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 Total Tput (tok/s)",
            "value": 3440.79,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TTFT (ms)",
            "value": 265.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 TPOT (ms)",
            "value": 35.74,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 throughput (tok/s)",
            "value": 311.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 Total Tput (tok/s)",
            "value": 621.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 TTFT (ms)",
            "value": 135.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 TPOT (ms)",
            "value": 24.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 throughput (tok/s)",
            "value": 44.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 Total Tput (tok/s)",
            "value": 50.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 TTFT (ms)",
            "value": 101.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 TPOT (ms)",
            "value": 22.27,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 throughput (tok/s)",
            "value": 578.91,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 Total Tput (tok/s)",
            "value": 651.53,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 TTFT (ms)",
            "value": 151.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 TPOT (ms)",
            "value": 26.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 throughput (tok/s)",
            "value": 86.57,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 Total Tput (tok/s)",
            "value": 97.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 TTFT (ms)",
            "value": 114.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 TPOT (ms)",
            "value": 22.92,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 throughput (tok/s)",
            "value": 1082.15,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 Total Tput (tok/s)",
            "value": 1216.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 TTFT (ms)",
            "value": 185.31,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 TPOT (ms)",
            "value": 28.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 throughput (tok/s)",
            "value": 165.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 Total Tput (tok/s)",
            "value": 186.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 TTFT (ms)",
            "value": 119.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 TPOT (ms)",
            "value": 23.78,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 throughput (tok/s)",
            "value": 303.97,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 Total Tput (tok/s)",
            "value": 342.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 TTFT (ms)",
            "value": 134.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 TPOT (ms)",
            "value": 25.59,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 1024/8192 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 throughput (tok/s)",
            "value": 42.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 Total Tput (tok/s)",
            "value": 378.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 TTFT (ms)",
            "value": 863.3,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 TPOT (ms)",
            "value": 22.73,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=1 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 throughput (tok/s)",
            "value": 1320.85,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 Total Tput (tok/s)",
            "value": 11924.72,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 TTFT (ms)",
            "value": 4197.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 TPOT (ms)",
            "value": 90.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 throughput (tok/s)",
            "value": 442.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 Total Tput (tok/s)",
            "value": 3993.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 TTFT (ms)",
            "value": 1191.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 TPOT (ms)",
            "value": 33.71,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=16 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 throughput (tok/s)",
            "value": 84.37,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 Total Tput (tok/s)",
            "value": 751.88,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 TTFT (ms)",
            "value": 770.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 TPOT (ms)",
            "value": 22.83,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=2 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 throughput (tok/s)",
            "value": 674.25,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 Total Tput (tok/s)",
            "value": 6029.36,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 TTFT (ms)",
            "value": 1955.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 TPOT (ms)",
            "value": 44.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=32 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 throughput (tok/s)",
            "value": 137.84,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 Total Tput (tok/s)",
            "value": 1239.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 TTFT (ms)",
            "value": 978.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 TPOT (ms)",
            "value": 27.32,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=4 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 throughput (tok/s)",
            "value": 999.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 Total Tput (tok/s)",
            "value": 9009.19,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 TTFT (ms)",
            "value": 2535.36,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 TPOT (ms)",
            "value": 60.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=64 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=8 throughput (tok/s)",
            "value": 265.87,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=8 Total Tput (tok/s)",
            "value": 2365.07,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=8 TTFT (ms)",
            "value": 940.15,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=8 TPOT (ms)",
            "value": 28.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "GLM-5-FP8 8192/1024 c=8 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=1 throughput (tok/s)",
            "value": 262.8,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=1 Total Tput (tok/s)",
            "value": 523.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=1 TTFT (ms)",
            "value": 48.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=1 TPOT (ms)",
            "value": 3.75,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=1 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 throughput (tok/s)",
            "value": 8654.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 Total Tput (tok/s)",
            "value": 17327.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 TTFT (ms)",
            "value": 190.24,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 TPOT (ms)",
            "value": 14.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=128 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=16 throughput (tok/s)",
            "value": 2392.26,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=16 Total Tput (tok/s)",
            "value": 4810,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=16 TTFT (ms)",
            "value": 65.62,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=16 TPOT (ms)",
            "value": 6.49,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=2 throughput (tok/s)",
            "value": 508.98,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=2 Total Tput (tok/s)",
            "value": 1008.66,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=2 TTFT (ms)",
            "value": 45.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=2 TPOT (ms)",
            "value": 3.87,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=2 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=256 throughput (tok/s)",
            "value": 12397.49,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=256 Total Tput (tok/s)",
            "value": 24783,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=256 TTFT (ms)",
            "value": 1619.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=256 TPOT (ms)",
            "value": 18.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=256 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=32 throughput (tok/s)",
            "value": 3702.6,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=32 Total Tput (tok/s)",
            "value": 7393.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=32 TTFT (ms)",
            "value": 78.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=32 TPOT (ms)",
            "value": 8.39,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 throughput (tok/s)",
            "value": 908.51,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 Total Tput (tok/s)",
            "value": 1825.9,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 TTFT (ms)",
            "value": 48.13,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 TPOT (ms)",
            "value": 4.22,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 throughput (tok/s)",
            "value": 5796.31,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 Total Tput (tok/s)",
            "value": 11595.22,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 TTFT (ms)",
            "value": 107.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 TPOT (ms)",
            "value": 10.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=8 throughput (tok/s)",
            "value": 1502.11,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=8 Total Tput (tok/s)",
            "value": 2993.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=8 TTFT (ms)",
            "value": 53.05,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=8 TPOT (ms)",
            "value": 5.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/1024 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=1 throughput (tok/s)",
            "value": 264.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=1 Total Tput (tok/s)",
            "value": 297.29,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=1 TTFT (ms)",
            "value": 47.47,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=1 TPOT (ms)",
            "value": 3.77,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=1 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=16 throughput (tok/s)",
            "value": 2555.89,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=16 Total Tput (tok/s)",
            "value": 2876.55,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=16 TTFT (ms)",
            "value": 59.94,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=16 TPOT (ms)",
            "value": 6.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=2 throughput (tok/s)",
            "value": 513.73,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=2 Total Tput (tok/s)",
            "value": 578.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=2 TTFT (ms)",
            "value": 43.5,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=2 TPOT (ms)",
            "value": 3.86,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=2 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=32 throughput (tok/s)",
            "value": 4055.76,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=32 Total Tput (tok/s)",
            "value": 4559.32,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=32 TTFT (ms)",
            "value": 71.45,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=32 TPOT (ms)",
            "value": 7.7,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=4 throughput (tok/s)",
            "value": 942.47,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=4 Total Tput (tok/s)",
            "value": 1062.74,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=4 TTFT (ms)",
            "value": 47.72,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=4 TPOT (ms)",
            "value": 4.17,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=8 throughput (tok/s)",
            "value": 1581.48,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=8 Total Tput (tok/s)",
            "value": 1779.7,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=8 TTFT (ms)",
            "value": 52.16,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=8 TPOT (ms)",
            "value": 4.93,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 1024/8192 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=1 throughput (tok/s)",
            "value": 253.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=1 Total Tput (tok/s)",
            "value": 2272.46,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=1 TTFT (ms)",
            "value": 127.35,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=1 TPOT (ms)",
            "value": 3.81,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=1 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=128 throughput (tok/s)",
            "value": 4711.54,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=128 Total Tput (tok/s)",
            "value": 42536.2,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=128 TTFT (ms)",
            "value": 897.76,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=128 TPOT (ms)",
            "value": 25.79,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=128 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=16 throughput (tok/s)",
            "value": 1994.01,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=16 Total Tput (tok/s)",
            "value": 17995.43,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=16 TTFT (ms)",
            "value": 234.66,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=16 TPOT (ms)",
            "value": 7.55,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=16 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=2 throughput (tok/s)",
            "value": 486.93,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=2 Total Tput (tok/s)",
            "value": 4339.23,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=2 TTFT (ms)",
            "value": 125.91,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=2 TPOT (ms)",
            "value": 3.96,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=2 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=32 throughput (tok/s)",
            "value": 2807.06,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=32 Total Tput (tok/s)",
            "value": 25101.67,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=32 TTFT (ms)",
            "value": 327.25,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=32 TPOT (ms)",
            "value": 10.8,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=32 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=4 throughput (tok/s)",
            "value": 842.45,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=4 Total Tput (tok/s)",
            "value": 7573.12,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=4 TTFT (ms)",
            "value": 156.11,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=4 TPOT (ms)",
            "value": 4.48,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=4 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=64 throughput (tok/s)",
            "value": 3901.1,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=64 Total Tput (tok/s)",
            "value": 35163.61,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=64 TTFT (ms)",
            "value": 537.21,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=64 TPOT (ms)",
            "value": 15.57,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=64 _gpu_count",
            "value": 1,
            "unit": ""
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=8 throughput (tok/s)",
            "value": 1354.86,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=8 Total Tput (tok/s)",
            "value": 12052.14,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=8 TTFT (ms)",
            "value": 188.43,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=8 TPOT (ms)",
            "value": 5.61,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23258004061"
          },
          {
            "name": "gpt-oss-120b 8192/1024 c=8 _gpu_count",
            "value": 1,
            "unit": ""
          }
        ]
      }
    ]
  }
}