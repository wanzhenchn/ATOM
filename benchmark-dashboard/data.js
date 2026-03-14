window.BENCHMARK_DATA = {
  "lastUpdate": 1773488649608,
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
      }
    ]
  }
}