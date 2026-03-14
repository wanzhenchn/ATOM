# Implementation Plan: Benchmark Dashboard + Auto Regression Analysis

## Task Type
- [x] Backend (CI/CD pipeline, Python scripts)

## Design Principle

**最小化自建代码，最大化复用现有工具。** 用成熟的开源 Action 替代自建看板和数据存储，减少长期维护压力。

## Architecture

```
Nightly Benchmark
      │
      ▼
┌──────────────────────────────┐
│  Run benchmarks (matrix)     │  existing jobs + MTP3 variant (via matrix)
│  DeepSeek × {non-MTP, MTP3}  │
│  gpt-oss-120b               │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  summarize.py (enhanced)     │  existing script + JSON output + exit code
└──────────┬───────────────────┘
      ┌────┴────┐
      │         │
      ▼         ▼
  Always     Regression only
      │         │
      ▼         ▼
┌──────────┐ ┌────────────────────────────┐
│ benchmark│ │ Re-run regressed configs   │
│ -action  │ │ with --enable-profiler     │
│ → chart  │ │ Upload traces → GH Issue   │
│ → alert  │ └────────────────────────────┘
└──────────┘
```

## What We Build vs What We Reuse

| 组件 | 方案 | 维护量 |
|------|------|--------|
| ~~Dashboard HTML/JS~~ | **`github-action-benchmark`** — 自动生成 Chart.js 看板到 gh-pages | **零** |
| ~~update_dashboard.py~~ | **`github-action-benchmark`** — 自动管理历史 JSON 数据 | **零** |
| ~~analyze_trace.py~~ | **不自建解析器** — 上传 trace artifact + 创建 Issue 附链接，人工分析 | **零** |
| summarize.py 增强 | 加 `--output-json` + 退出码 | **小** (30 行改动) |
| Workflow 修改 | 加 MTP matrix variant + regression job + benchmark-action step | **中** |

净减少：**3 个新文件 → 0 个新文件**，只修改 2 个现有文件。

## Implementation Steps

### Step 1: Add MTP3 Coverage via Matrix Variant

**File:** `.github/workflows/atom-benchmark.yaml`
**Operation:** Modify existing `deepseek-r1-0528-benchmark` job

Convert to matrix with variant dimension:

```yaml
strategy:
  fail-fast: false
  matrix:
    config: ${{ fromJson(needs.parse-param-lists.outputs.matrix_json) }}
    variant:
      - name: ""
        args: "--kv_cache_dtype fp8 -tp 8"
        suffix: ""
      - name: " MTP3"
        args: "--kv_cache_dtype fp8 -tp 8 --method mtp --num-speculative-tokens 3"
        suffix: "-mtp3"
env:
  ARGS: ${{ matrix.variant.args }}
  RESULT_FILENAME: deepseek-r1-0528${{ matrix.variant.suffix }}-${{ matrix.config.input_length }}-...
```

Risk control — skip low-concurrency MTP to limit CI time:
```yaml
exclude:
  - variant: { suffix: "-mtp3" }
    config: { concurrency: 1 }
  - variant: { suffix: "-mtp3" }
    config: { concurrency: 2 }
```

Single job definition covers both variants. Adding/removing = one line in matrix.

### Step 2: Enhance `summarize.py` — JSON Output + Exit Code

**File:** `.github/scripts/summarize.py`
**Operation:** Modify (~30 lines)

Changes:
1. Add `--output-json <path>` CLI arg
2. `print_regression_report()` returns regression list with config details
3. Write JSON with `all_results` + `regressions` list
4. `sys.exit(1)` when regression detected

Output format:
```json
{
  "timestamp": "2026-03-14T17:00:00Z",
  "commit": "abc123",
  "regression_count": 1,
  "regressions": [
    {
      "model": "DeepSeek-R1-0528", "isl": 1024, "osl": 1024, "conc": 128,
      "output_throughput": {"current": 4200, "baseline": 4500, "pct": -6.7}
    }
  ]
}
```

### Step 3: Add `github-action-benchmark` for Dashboard

**File:** `.github/workflows/atom-benchmark.yaml`
**Operation:** Modify (add steps to `summarize-benchmark-result` job)

```yaml
- name: Transform results for benchmark-action
  run: |
    # Convert ATOM JSON results to customBiggerIsBetter format
    python3 -c "
    import json, glob
    entries = []
    for f in glob.glob('*.json'):
        d = json.load(open(f))
        if 'output_throughput' not in d: continue
        model = d.get('model_id','').split('/')[-1]
        isl = d.get('random_input_len', 0)
        osl = d.get('random_output_len', 0)
        conc = d.get('max_concurrency', 0)
        label = f'{model} {isl}/{osl} c={conc}'
        entries.append({'name': f'{label} throughput', 'unit': 'tok/s',
                        'value': d['output_throughput']})
        entries.append({'name': f'{label} TPOT', 'unit': 'ms',
                        'value': d['mean_tpot_ms']})
    json.dump(entries, open('benchmark-results.json','w'))
    "

- name: Store benchmark result
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: customBiggerIsBetter
    output-file-path: benchmark-results.json
    gh-pages-branch: gh-pages
    benchmark-data-dir-path: benchmark-dashboard
    auto-push: true
    alert-threshold: "80%"
    comment-on-alert: true
    fail-on-alert: false
    max-items-in-chart: 90
```

This gives us for free:
- **Trend charts** on `https://<org>.github.io/ATOM/benchmark-dashboard`
- **Auto-alert** when metric drops below 80% of previous (comment on commit)
- **Historical data** managed automatically (90 data points retained)
- **Zero custom frontend code**

### Step 4: Add Regression Re-run with Profiler

**File:** `.github/workflows/atom-benchmark.yaml`
**Operation:** Modify (add new job)

```yaml
regression-analysis:
  needs: [summarize-benchmark-result]
  if: needs.summarize-benchmark-result.outputs.has_regression == 'true'
  runs-on: atom-mi355-8gpu.predownload
  steps:
    - name: Checkout
      uses: actions/checkout@v4

    - name: Download regression report
      uses: actions/download-artifact@v4
      with:
        name: regression-report

    - name: Re-run regressed configs with profiler
      run: |
        # Parse regressed configs from JSON
        # For each: launch server → start_profile → benchmark (fewer prompts) → stop_profile
        # Collect traces from /app/trace/

    - name: Upload traces
      uses: actions/upload-artifact@v4
      with:
        name: regression-traces-${{ github.run_id }}
        path: profiler-traces/

    - name: Create GitHub Issue
      if: success()
      uses: actions/github-script@v7
      with:
        script: |
          const report = require('./regression_report.json');
          const body = `## Performance Regression Detected\n\n` +
            `**Commit:** ${context.sha}\n` +
            `**Run:** ${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}\n\n` +
            `### Regressed Configurations\n\n` +
            report.regressions.map(r =>
              `- **${r.model}** ${r.isl}/${r.osl} c=${r.conc}: ` +
              `throughput ${r.output_throughput.pct.toFixed(1)}%`
            ).join('\n') +
            `\n\n### Profiler Traces\n` +
            `Download from the workflow artifacts above.\n` +
            `\n\ncc @${context.actor}`;
          await github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: `[Perf Regression] ${report.regressions.length} config(s) regressed`,
            body: body,
            labels: ['performance', 'regression']
          });
```

No custom trace parser — traces are uploaded as artifacts, team downloads and analyzes in Chrome trace viewer or Perfetto. When mature enough, can add analysis later.

## Key Files

| File | Operation | Change Size |
|------|-----------|-------------|
| `.github/workflows/atom-benchmark.yaml` | Modify | ~80 lines (matrix variant + benchmark-action + regression job) |
| `.github/scripts/summarize.py` | Modify | ~30 lines (JSON output + exit code) |

**No new files.** Everything is workflow config + minor script enhancement.

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| MTP3 doubles benchmark matrix size | `exclude` low-concurrency MTP combos; MTP jobs run in parallel |
| `github-action-benchmark` doesn't match exact needs | It's the most popular solution (4k+ stars, actively maintained); `customBiggerIsBetter` format is flexible enough |
| Profiler re-run adds CI time | Only triggered on regression (rare); uses fewer prompts |
| gh-pages data grows | `max-items-in-chart: 90` built-in retention |
| Trace analysis requires human effort | Acceptable initially; automate later only if regression frequency justifies it |

## Execution Order

```
Step 1 (MTP matrix variant) ──┐
Step 2 (summarize.py enhance) ┘  parallel, no dependency
         │
         ▼
Step 3 (benchmark-action) ──┐
Step 4 (regression job)     ┘  both modify workflow, do together
```

Total estimated effort: **~110 lines of changes across 2 existing files**.
