import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def print_format_fail_samples(records: list[dict[str, Any]], top_k: int) -> None:
    reward_compute = [r for r in records if r.get("stage") == "reward_compute"]
    fails = [r for r in reward_compute if float(r.get("format_score", 0.0)) < 0]

    print(f"[format_fail] total={len(fails)} / reward_compute={len(reward_compute)}")
    for idx, rec in enumerate(fails[:top_k], start=1):
        print(f"\n--- fail sample {idx}/{min(top_k, len(fails))} ---")
        print(f"ts: {rec.get('ts')}")
        print(f"sample_uid: {rec.get('sample_uid')}")
        print(f"reason: {rec.get('format_failure_reason')}")
        print(f"pred_len={rec.get('pred_len')} gt_len={rec.get('gt_len')}")
        print(f"format_score={rec.get('format_score')} length_score={rec.get('length_score')} mse_score={rec.get('mse_score')} final_score={rec.get('final_score')}")
        print(f"raw_model_output_head: {rec.get('raw_model_output_head')}")
        print(f"parsed_answer_text_head: {rec.get('parsed_answer_text_head')}")


def summarize_reward_chain(records: list[dict[str, Any]]) -> None:
    manager_out = [r for r in records if r.get("stage") == "reward_manager_output"]
    trainer_in = [r for r in records if r.get("stage") == "trainer_reward_input"]
    trainer_extract = [r for r in records if r.get("stage") == "trainer_reward_extract"]
    trainer_tensor = [r for r in records if r.get("stage") == "trainer_reward_tensor"]

    print("\n[reward_chain_summary]")
    if manager_out:
        vals = [float(r.get("reward_score", 0.0)) for r in manager_out]
        neg = sum(v < 0 for v in vals)
        print(f"reward_manager_output: count={len(vals)} min={min(vals):.4f} max={max(vals):.4f} mean={sum(vals)/len(vals):.4f} neg_count={neg}")
    else:
        print("reward_manager_output: none")

    if trainer_in:
        last = trainer_in[-1]
        print(
            "trainer_reward_input(last): "
            f"reward_min={last.get('reward_min')} reward_max={last.get('reward_max')} "
            f"reward_mean={last.get('reward_mean')} has_negative={last.get('has_negative')}"
        )
    else:
        print("trainer_reward_input: none")

    if trainer_extract:
        last = trainer_extract[-1]
        print(
            "trainer_reward_extract(last): "
            f"seq_score_min={last.get('seq_score_min')} seq_score_max={last.get('seq_score_max')} "
            f"seq_score_mean={last.get('seq_score_mean')} seq_negative_count={last.get('seq_negative_count')}"
        )
    else:
        print("trainer_reward_extract: none")

    if trainer_tensor:
        last = trainer_tensor[-1]
        print(
            "trainer_reward_tensor(last): "
            f"score_seq_min={last.get('score_seq_min')} score_seq_max={last.get('score_seq_max')} "
            f"score_seq_mean={last.get('score_seq_mean')} score_seq_negative_count={last.get('score_seq_negative_count')} "
            f"reward_seq_min={last.get('reward_seq_min')} reward_seq_max={last.get('reward_seq_max')} "
            f"reward_seq_mean={last.get('reward_seq_mean')} reward_seq_negative_count={last.get('reward_seq_negative_count')}"
        )
    else:
        print("trainer_reward_tensor: none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ts_chain_debug.jsonl")
    parser.add_argument("--debug-file", type=str, default="/tmp/ts_chain_debug.jsonl")
    parser.add_argument("--top-k-fail", type=int, default=10)
    args = parser.parse_args()

    debug_file = Path(args.debug_file)
    if not debug_file.exists():
        raise FileNotFoundError(f"Debug file not found: {debug_file}")

    records = load_jsonl(debug_file)
    print(f"Loaded records: {len(records)} from {debug_file}")
    print_format_fail_samples(records, top_k=args.top_k_fail)
    summarize_reward_chain(records)


if __name__ == "__main__":
    main()
