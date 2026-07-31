"""Summarize the primary HellaSwag and MMLU accuracies from lm-eval JSON files."""

import argparse
import json
from pathlib import Path


def metric(results, task, name="acc,none"):
    value = results.get(task, {}).get(name)
    return None if value is None else 100.0 * value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload.get("results", {})
        rows.append(
            (
                payload.get("edited_run", {}).get("run_name", path.stem),
                metric(results, "hellaswag"),
                metric(results, "mmlu"),
            )
        )

    print("| Method | HellaSwag Acc. | MMLU Acc. |")
    print("|---|---:|---:|")
    for name, hellaswag, mmlu in rows:
        hellaswag_text = "-" if hellaswag is None else f"{hellaswag:.2f}"
        mmlu_text = "-" if mmlu is None else f"{mmlu:.2f}"
        print(f"| {name} | {hellaswag_text} | {mmlu_text} |")


if __name__ == "__main__":
    main()
