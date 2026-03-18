# Protocol Stable Baseline (2026-03-18)

## Git Baseline
- Commit: `eaf9873e26caf41f4942f47dbd9b9ec9db719230`
- Branch: `protocol_stable_baseline_20260318`
- Tag: `protocol_stable_baseline_20260318`

## Checkpoints
- SFT: `artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_new/global_step_11`
- RL: `artifacts/checkpoints/rl/etth1_ot_qwen3_1_7b_rl_gpu012_eval5_test_reward/global_step_20`

## Runtime/Profile Snapshot
- Profile: `examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh`
- Reward mode: `strict_raw` (in `recipe/time_series_forecast/reward.py`)
- Debug files:
  - `logs/debug/ts_chain_debug_eval5_test.jsonl`
  - `logs/debug/turn3_generation_debug_eval5_test.jsonl`

## Stability Metrics (chain: reward_compute)
- Total samples: `13888`
- `pred_len == 96`: `13679` (`98.4951%`)
- `pred_len in {94,95}`: `105` (`0.7560%`)
- `missing_answer_close_tag`: `0`
- `was_clipped == true`: `0`

## Success-Sample Error Distribution (`pred_len==96`, `format_failure_reason==ok`)
- Count: `13679`
- `raw_mse`: mean `2.4644`, p50 `1.6788`, p90 `5.0204`, p95 `7.5211`, max `43.6159`
- `raw_mae`: mean `1.2269`, p50 `1.0463`, p90 `2.0193`, p95 `2.4647`, max `6.1220`

## Reproduce Command (as used)
```bash
DEBUG_CHAIN=1 \
TS_CHAIN_DEBUG_FILE=$PWD/logs/debug/ts_chain_debug_eval5_test.jsonl \
MODEL_PATH=/data/linyujie/Cast-R1-TS-main/Cast-R1-TS-main/artifacts/checkpoints/sft/time_series_forecast_sft_teacher200_new/global_step_11/huggingface \
PROFILE_PATH=examples/time_series_forecast/configs/etth1_ot_qwen3_gpu012.sh \
RUN_MODE=train \
RL_EXP_NAME=etth1_ot_qwen3_1_7b_rl_gpu012_eval5_test_reward \
bash examples/time_series_forecast/run_qwen3-1.7B.sh \
  trainer.total_training_steps=20 \
  trainer.test_freq=5 \
  trainer.resume_mode=disable
```