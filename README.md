# TitanRL rollout scheduling study

中文学习资料见 [TitanRL 学习手册](docs/index.md)，涵盖 rollout 调度、策略滞后和吞吐测量。

## Question

When does overlapping rollout generation with policy updates improve end-to-end throughput, and what policy lag does that overlap introduce?

The experiment recipes, fixed-cap patch, and throughput analyzer are ready for a smoke test. The local checks used synthetic telemetry. GPU measurements are pending.

## Upstream baseline

Start with [TorchTitan v0.4.0](https://github.com/pytorch/torchtitan/releases/tag/v0.4.0). The [TitanRL architecture](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/README.md#architecture) separates generation and training and links them through a rollout work buffer. The upstream documentation identifies `target_offpolicy_steps=0` as the synchronous control. A positive setting enables overlap. The controller also exposes `windowed_fifo_batches`, which can bound how far slow rollout groups exceed the target. Keep this setting fixed for the first comparison.

## First experiment

- Start from the public [DAPO Math recipe](https://github.com/pytorch/torchtitan/tree/v0.4.0/torchtitan/rl/examples/dapo_math). Identify the smallest supported public config and its device requirements before scheduling a run.
- Compare `target_offpolicy_steps=0` with one supported positive setting. Keep the model, dataset, hardware, trainer/generator placement, batch size, generation limits, and seed fixed.
- Record post-warm-up wall time per update, generated tokens per second, and the distribution of policy lag in optimizer steps. Confirm the exact metric names and semantics in the versioned code before collecting results.
- Report reward as a sanity check. Treat a short run as a systems measurement and leave conclusions about learning quality for a longer controlled experiment.
- If compatible GPUs are unavailable, inspect existing CPU coverage for rollout scheduling and build or run a small scheduling reproduction. Label any synthetic measurements as such.

## Smoke test

The local `smoke_recipe.py` wraps the upstream Qwen3-0.6B alphabet-sort recipe. Both configurations run ten optimizer steps on six CUDA GPUs with one turn per sample and validation disabled. `sync_smoke` uses a rollout target of 0 and `async_smoke` uses 1. The stock controller derives a vLLM sequence cap of 64 and 128 respectively with the default topology. This run checks launch and logging for the stock settings. A follow-up comparison with a fixed cap is needed to isolate scheduling.

From a checkout of TorchTitan at `45e33cd413b64cb37a7840cfd3c9765d3c639ae1` (`v0.4.0`), follow the upstream RL environment setup and place the public `Qwen/Qwen3-0.6B` checkpoint in `assets/hf/Qwen3-0.6B`. Set `TITANRL_LAB` to the absolute path of this directory. Each invocation creates a fresh output directory because the generator exporter appends JSONL. [TitanRL setup](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/README.md#prerequisites)

```bash
(
  set -e
  : "${TITANRL_LAB:?Set TITANRL_LAB to the local experiment directory}"
  mkdir -p outputs/rl
  run_dir=$(mktemp -d "$PWD/outputs/rl/rollout-smoke.XXXXXX")
  printf 'Smoke-test output: %s\n' "$run_dir"
  for arm in sync_smoke async_smoke; do
    PYTHONPATH="$PWD:$TITANRL_LAB${PYTHONPATH:+:$PYTHONPATH}" \
      OTEL_METRICS_EXPORTER=jsonl VLLM_LOG_STATS_INTERVAL=1 \
      python -m torchtitan.rl.train --module smoke_recipe --config "$arm" \
        --output-dir "$run_dir/$arm"
  done
)
```

Check the console for ten optimizer steps and the `timing/step/total` and `train_batch/policy_age` metrics. Confirm that `vllm_metrics/` contains multiple snapshots of `vllm.generation_tokens` with increasing timestamps and the same counter start time. Use a longer run if ten steps leave too few post-warm-up snapshots.

### Compare generated tokens per second

Set `RUN_DIR` to the printed output directory containing `sync_smoke/` and `async_smoke/`. Choose the same snapshot offsets for both arms after checking that both measurement windows fall inside active training after warm-up. The command below compares the third through eighth snapshots in each arm.

```bash
python "$TITANRL_LAB/compare_generation.py" "$RUN_DIR" \
  --skip-snapshots 2 --intervals 5
```

The script expects one generator log per arm, as in the smoke recipe, and reports the cumulative token delta divided by elapsed time from that arm's export timestamps. Use console metrics for trainer step time and policy age. The ten-step run primarily checks instrumentation, and useful throughput estimates require a longer steady-state window.

### Hold the generator sequence cap fixed

Apply the local patch from the pinned TorchTitan checkout. The patch adds an optional cap before the controller hands the value to the generator, so the scheduler and CUDA graph configuration receive the same value. The fixed-cap recipes use 128 in both arms, matching the stock asynchronous setting.

```bash
git apply --check "$TITANRL_LAB/fixed-generator-cap.patch" &&
  git apply "$TITANRL_LAB/fixed-generator-cap.patch"
```

Run the same launch loop with `sync_fixed_cap async_fixed_cap` as the two recipe names. Check that startup logs report `max_num_seqs=128` in both arms. Add `--fixed-cap` to the analyzer command to read these output directories.

The patch targets public TorchTitan source. See the [upstream license](https://github.com/pytorch/torchtitan/blob/v0.4.0/LICENSE).

## Work log

### 2026-11-09

- Verified the public v0.4.0 release and the synchronous configuration knob in the versioned TitanRL documentation.
- Selected rollout scheduling as the first comparison because the same experiment measures throughput and a correctness-relevant side effect, policy lag.
- Next: trace the chosen recipe to the controller and verify which logged fields measure lag and throughput. No GPU run or benchmark results yet.

### 2026-11-10

- Traced the v0.4.0 controller from [buffer sizing](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L156-L218) to [batch consumption](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L1100-L1133). With `P` prompt groups per step and target `S`, the buffer has `(S + 1) * P` active slots and one rollout worker per slot. Actual lag is measured at consumption.
- Verified the [metric definitions](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/metrics.py). `timing/step/total` covers the controller's trainer step, including batch wait. `perf/trainer/tokens_per_second_full_step` counts valid **training** tokens, not generated tokens. `train_batch/policy_age` and `train_batch/policy_age_max` summarize the trainer policy version minus each sample's stored minimum version. See the 2026-11-11 note for the v0.4.0 multi-turn caveat. `train_batch/pct_samples_over_target_age` uses each run's own target, so compare mean and max across runs and use a common age threshold if collecting per-sample data.
- Found an experiment-design issue in [v0.4.0 actor setup](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L575-L609). The controller derives the generator's `max_num_seqs` from rollout concurrency and validation size. Varying the buffer target can therefore also change vLLM's sequence cap and CUDA graph capture sizes. Record the effective cap for each arm. Report the stock comparison as an end-to-end configuration effect, and hold the cap fixed in a separate comparison if isolating the effect of scheduling.
- Identified two public recipes. [DAPO Math](https://github.com/pytorch/torchtitan/tree/v0.4.0/torchtitan/rl/examples/dapo_math) uses eight GPUs and 150 steps. [Qwen3-0.6B alphabet sort](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan_recipes/rl/alphabet_sort.py#L102-L171) is a smaller 10-step smoke test with two trainer GPUs and four generator GPUs. Reserve the short recipe for validating launch and metrics.
- No new release since v0.4.0. Upstream main merged [#5944](https://github.com/pytorch/torchtitan/pull/5944), which changes how the generator cap is derived, and [#5228](https://github.com/pytorch/torchtitan/pull/5228), which lets periodic validation run alongside training. Keep the baseline pinned to v0.4.0 so those changes do not enter the first comparison.
- Next: trace completion-token accounting and confirm which validation settings keep the smoke test and measurement window comparable. Findings above are from reading public source. No GPU run yet.

### 2026-11-11

- Found a measurement bug relevant to the experiment. In [v0.4.0's sample builder](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/trainer/training_sample_builder.py#L350-L398), a multi-turn segment keeps the opening turn's minimum policy version, even if a later turn uses an older one. The stored age can therefore be too low. [#5954](https://github.com/pytorch/torchtitan/pull/5954) fixes the minimum across turns on main. For the first v0.4.0 smoke test, set the alphabet-sort dataset's `max_turns=1` in both arms. The [dataset defaults to three turns](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/examples/alphabet_sort/data.py#L58-L74). For a subsequent multi-turn experiment, use a baseline that includes the fix and record the exact commit.
- In v0.4.0, `validation.steps=0` skips both the blocking start and end validation and makes the validation contribution to the derived generator cap zero. Hold this at zero in both throughput arms. Run the held-out reward check separately with identical validation settings for both arms. [Controller validation](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L760-L817) · [Validation sample selection](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/rollout/rollouter.py#L145-L154)
- For generator throughput, use the delta of the cumulative `vllm.generation_tokens` counter divided by elapsed wall time across a shared post-warm-up interval. Enable `generator.engine.stat_logger` and use `OTEL_METRICS_EXPORTER=jsonl` to capture it locally. Validate that the smoke run exports enough snapshots, and count each distinct generator engine stream once. `rollout/output_tokens` summarizes completion length per turn, and the default JSONL rollout recorder samples extreme rewards, so neither gives a total generated-token count. [vLLM observability](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/observability/metrics/README.md) · [Rollout recorder](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/rollout/recorder.py)
- No new release. Main also added an [SGLang engine](https://github.com/pytorch/torchtitan/pull/5812). Keep vLLM fixed across both arms to avoid mixing an engine comparison with a scheduling comparison.
- Next: turn the single-turn, validation-off smoke-test settings into a reproducible recipe and verify the exported counter schema. The findings above are from public source review. No GPU run yet.

### 2026-11-12

- Added the two local smoke-test configurations and launch commands. They use the same public recipe, disable W&B and validation, restrict the dataset to single-turn prompts, and enable local vLLM token telemetry.
- Verified the local recipe's Python syntax and exercised both configuration functions with stand-ins for the upstream config objects. Used the real OpenTelemetry SDK on CPU to generate two synthetic snapshots and verified the cumulative counter schema and a known increment. The full TorchTitan launch and actual generator exports still require CUDA.
- No new upstream release. Reviewed [#5982](https://github.com/pytorch/torchtitan/pull/5982), which corrects generator weight loading for an FSDP-sharded output head in stream-ordered loading. The smoke test uses the dense Qwen3 tensor-parallel generator and does not exercise that configuration.
- Next: build a small local analyzer for paired JSONL snapshots and verify it rejects resets and missing streams before processing GPU results.

### 2026-11-13

- Implemented `compare_generation.py` for the single-generator smoke recipe. A CPU check using synthetic exports from the OpenTelemetry SDK recovered the known rates of 30 and 45 tokens/s. The script rejected a reset, insufficient samples, and missing or duplicate generator logs. These values validate the analyzer and are synthetic.
- No new release. Upstream main merged [#5990](https://github.com/pytorch/torchtitan/pull/5990), which lets rollout callers send token payloads directly to generators after reserving a slot with the router. This changes generation overhead and is a separate variable from buffer depth. Both smoke arms use the pinned v0.4.0 routing path.
- Next: trace how to hold the vLLM sequence cap constant in both arms and verify that the chosen override reaches both the scheduler and graph capture settings. CUDA launch and telemetry are still pending.

### 2026-11-14

- Traced the controller's `max_num_seqs` to both `EngineArgs` and the CUDA graph builder. The upstream `extra_engine_args` helper rejects a second value for this key. Added a local patch at the controller boundary and two opt-in recipes that fix the cap at 128. [Controller handoff](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/controller/controller.py#L596-L635) · [vLLM engine options](https://github.com/pytorch/torchtitan/blob/v0.4.0/torchtitan/rl/generator/vllm/engine.py#L280-L313)
- The patch applies to the pinned public source. A local check of the cap calculation reproduced 64 and 128 for the stock pair and 128 for both fixed arms. The analyzer accepted both fixed-cap run names with synthetic JSONL. Runtime verification of the resolved vLLM settings still needs a CUDA run.
- No new release in the review window. Main merged an [opt-in weight transfer path](https://github.com/pytorch/torchtitan/pull/5997). Weight transfer time can affect the benefit of overlap, so both arms use the same v0.4.0 transport.
- Weekly summary. The experiment has paired run recipes, a fixed-cap control, and a local throughput analyzer tested with synthetic logs. Next week, run the paired six-GPU smoke test when suitable CUDA resources are available and inspect the cap, token counter, step time, and policy age before planning a longer measurement.
