"""Evaluate the available t438 projector pair with the archived probe/protocol.

This is a separate candidate experiment, not a historical-projector replay.
The archived evaluator, probe, thresholds, prompts and task-specific L remain fixed.
"""

import argparse
import json
from pathlib import Path

import historical_replay as replay


replay.OUT = replay.ROOT / "exp_results" / "revision_t438_old_probe_20261102"
TASKS = replay.TASKS
PROJECTORS = {
    "svd_positive_proj": (
        "server_checkpoint_backup/t438_svd_halueval_2000_k998/no_mean_sub_uu_positive.pt",
        "4b3fbf3d3874734e12d2ae98336dc0ca9fe63f10960c4aff867869c4e0dc2cff",
    ),
    "svd_negative_proj": (
        "server_checkpoint_backup/t438_svd_halueval_2000_k998/no_mean_sub_uu_negative.pt",
        "ee08f577aabc8f6903d9ac9bdb4250343fb2d2be2cfe8dbf7d47cc9601df1567",
    ),
    "gevd_positive_proj": (
        "server_checkpoint_backup/t438_gevd_adaptive_bottomk_k32_n8/no_mean_sub_uu_positive.pt",
        "3202d7d2be2ae2bbf1cc98d56fdd86e00cc976e9fff1806ac91e7c3d5d659751",
    ),
    "gevd_negative_proj": (
        "server_checkpoint_backup/t438_gevd_adaptive_bottomk_k32_n8/no_mean_sub_uu_negative.pt",
        "b044719fe6f655814677e5f5264631f18a0e35c872ffad5c6e75de0e1caae343",
    ),
}
original_resolved_args = replay.resolved_args


def candidate_args(task, model, smoke):
    args = original_resolved_args(task, model, smoke)
    for key, (relative, _) in PROJECTORS.items():
        args[key] = str(replay.ROOT / relative)
    return args


replay.resolved_args = candidate_args


def checked_preflight(tasks, model):
    evidence = replay.preflight(tasks, model)
    for key, (_, expected) in PROJECTORS.items():
        check = evidence["checks"].setdefault(key, {})
        actual = check.get("sha256")
        check["expected_sha256"] = expected
        if actual != expected:
            evidence["errors"].append("t438 candidate SHA256 mismatch: " + key)
    evidence["errors"] = sorted(set(evidence["errors"]))
    replay.save(replay.OUT / "candidate_preflight.json", evidence)
    print(json.dumps({"errors": evidence["errors"], "assets": evidence["checks"]}, indent=2), flush=True)
    if evidence["errors"]:
        raise SystemExit("Candidate preflight failed; no model run started.")


def pilot_report():
    truth = replay.load(replay.output("truthfulqa", True))
    bbq = replay.load(replay.output("bbq_disambig", True))
    old = replay.load(replay.CONFIG / "archived_fixture.json")
    truth_scores = truth["model_scores"]
    bbq_rows = bbq["results"]
    report = {
        "label": "t438 projectors + archived probe/protocol; not historical projector replay",
        "truthfulqa_first10": {
            "n": len(truth_scores),
            "MC1_percent": 100 * sum(row["MC1"] for row in truth_scores) / len(truth_scores),
            "archived_first10_MC1_matches": sum(
                row["MC1"] == old_row["MC1"]
                for row, old_row in zip(truth_scores, old["truthfulqa"]["metrics"])
            ),
            "questions_match_archived": [x["question"] for x in truth["question"]] == old["truthfulqa"]["questions"],
        },
        "bbq_disambig_first10": {
            "n": len(bbq_rows),
            "predictions_match_archived": sum(
                row["pred_idx"] == pred
                for row, pred in zip(bbq_rows, old["bbq_disambig"]["pred_idx"])
            ),
            "keys_match_archived": [[x["category"], x["example_id"]] for x in bbq_rows] == old["bbq_disambig"]["keys"],
        },
    }
    replay.save(replay.OUT / "candidate_pilot_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def full_report():
    report = {"label": "t438 projectors + archived probe/protocol; new candidate experiment"}
    for task in TASKS:
        path = replay.output(task, False)
        if not path.is_file():
            continue
        data = replay.load(path)
        if task == "truthfulqa":
            report[task] = {
                "n": len(data["question"]),
                "MC1_percent": 100 * data["total_mc1"],
                "MC2_percent": 100 * data["total_mc2"],
                "MC3_percent": 100 * data["total_mc3"],
                "router_stats": data.get("router_stats", {}),
            }
        else:
            s = data["summary"]
            report[task] = {
                "n": s["total_examples"],
                "accuracy_percent": s["accuracy_percent"],
                "unknown_answer_rate_percent": s["unknown_answer_rate_percent"],
                "router_stats": s.get("router_stats", {}),
            }
            if task == "bbq_disambig":
                report[task]["bias_score_percent"] = s["bias_score_percent"]
                report[task]["stereotypical_response_rate_percent"] = s["stereotypical_response_rate_percent"]
    replay.save(replay.OUT / "candidate_full_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("check", "pilot", "full"), default="check")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--model-path", type=Path, default=replay.MODEL_DEFAULT)
    args = parser.parse_args()
    tasks = tuple(args.tasks)
    if len(set(tasks)) != len(tasks):
        raise ValueError("Duplicate task")
    required = ("truthfulqa", "bbq_disambig") if args.phase == "pilot" else tasks
    checked_preflight(required, args.model_path)
    if args.phase == "check":
        print("Candidate preflight passed. Next: --phase pilot", flush=True)
    elif args.phase == "pilot":
        for task in required:
            replay.run_one(task, args.model_path, True)
        pilot_report()
    else:
        if not (replay.OUT / "candidate_pilot_report.json").is_file():
            raise RuntimeError("Full candidate run requires a completed first-ten pilot")
        for task in tasks:
            replay.run_one(task, args.model_path, False)
            full_report()
        print("CANDIDATE FULL RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
