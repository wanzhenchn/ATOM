window.BENCHMARK_DATA = {
  "lastUpdate": 1773614186936,
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
        "benches": [
          {
            "name": "Meta-Llama-3-8B-Instruct 1024/1024 c=128 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090100375"
          },
          {
            "name": "Meta-Llama-3-8B-Instruct 1024/1024 c=128 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090100375"
          },
          {
            "name": "Meta-Llama-3-8B-Instruct 1024/1024 c=128 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090100375"
          },
          {
            "name": "Meta-Llama-3-8B-Instruct 1024/1024 c=128 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090100375"
          },
          {
            "name": "Meta-Llama-3-8B-Instruct 1024/1024 c=128 _gpu_count",
            "value": 8,
            "unit": ""
          },
          {
            "name": "Meta-Llama-3-8B-Instruct 1024/1024 c=64 throughput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090100375"
          },
          {
            "name": "Meta-Llama-3-8B-Instruct 1024/1024 c=64 Total Tput (tok/s)",
            "value": 0,
            "unit": "tok/s",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090100375"
          },
          {
            "name": "Meta-Llama-3-8B-Instruct 1024/1024 c=64 TTFT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090100375"
          },
          {
            "name": "Meta-Llama-3-8B-Instruct 1024/1024 c=64 TPOT (ms)",
            "value": 0,
            "unit": "ms",
            "extra": "Run: https://github.com/ROCm/ATOM/actions/runs/23090100375"
          },
          {
            "name": "Meta-Llama-3-8B-Instruct 1024/1024 c=64 _gpu_count",
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
      }
    ]
  }
}