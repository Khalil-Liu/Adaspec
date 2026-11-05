"""Run matched BBQ policy controls with the archived evaluator and t438 assets.

Upload next to candidate_t438_old_probe.py on the server. Dry-run by default.
The probability thresholds force all candidate forwards to one chosen action.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import candidate_t438_old_probe as candidate


replay = candidate.replay
OUT = candidate.replay.OUT / "matched_policies"
POLICIES = {
    "bypass": (0.0, 1.0),  # sigmoid probabilities are within [0, 1]; strict comparisons
    "svd": (2.0, 3.0),
    "gevd": (-2.0, -1.0),
}
DEFAULT = ("bypass_ambig", "bypass_disambig")
ALL = tuple(f"{policy}_{mode}" for policy in POLICIES for mode in ("ambig", "disambig"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plan(name):
    policy, mode = name.split("_", 1)
    source = replay.OUT / "full" / f"bbq_{mode}.planned_args.json"
    args = load(source)
    args["output_path"] = str(OUT / (name + ".json"))
    args["lower_threshold"], args["upper_threshold"] = POLICIES[policy]
    return args


def check_assets(names):
    modes = {"bbq_" + name.split("_", 1)[1] for name in names}
    model = Path(plan(names[0])["model_name"])
    evidence = replay.preflight(tuple(sorted(modes)), model)
    for key, (_, expected) in candidate.PROJECTORS.items():
        actual = evidence["checks"].get(key, {}).get("sha256")
        if actual != expected:
            evidence["errors"].append("t438 projector SHA256 mismatch: " + key)
    save(OUT / "preflight.json", evidence)
    if evidence["errors"]:
        raise RuntimeError("Preflight errors: " + "; ".join(evidence["errors"]))
    if "bbq_ambig" in modes:
        replay.stage_ambiguous_input()
    print("Preflight passed; source, data, probe, and projector hashes match.", flush=True)


def run_one(name):
    args = plan(name)
    target = Path(args["output_path"])
    expected_actions = 5246 * 3
    policy = name.split("_", 1)[0]
    if target.exists():
        data = load(target)
        stats = data["summary"]["router_stats"]
        if len(data["results"]) != 5246 or stats[policy] != expected_actions:
            raise RuntimeError("Existing output is incomplete or unexpected: " + str(target))
        print("SKIP verified existing", name, flush=True)
        return
    if target.with_suffix(".planned_args.json").exists() or target.with_suffix(".log").exists():
        raise RuntimeError("A previous attempt exists; inspect it before retrying: " + name)
    save(target.with_suffix(".planned_args.json"), args)
    cmd = replay.command("bbq_" + name.split("_", 1)[1], args)
    print("RUN", name, "->", target, flush=True)
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    with target.with_suffix(".log").open("w", encoding="utf-8") as log:
        status = subprocess.run(cmd, cwd=replay.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if status.returncode:
        raise RuntimeError(f"{name} exited {status.returncode}; inspect {target.with_suffix('.log')}")
    data = load(target)
    stats = data["summary"]["router_stats"]
    if len(data["results"]) != 5246 or stats[policy] != expected_actions:
        raise RuntimeError("Forced policy or row count mismatch: " + name)
    print("DONE", name, "A", data["summary"]["accuracy_percent"], "U", data["summary"]["unknown_answer_rate_percent"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=ALL, default=list(DEFAULT))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.tasks or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("Specify unique tasks")
    for name in args.tasks:
        p = plan(name)
        print("PLAN", name, "thresholds", p["lower_threshold"], p["upper_threshold"], "output", p["output_path"])
    if not args.execute:
        print("Dry run only. Add --execute to run the selected tasks.")
        return
    check_assets(args.tasks)
    for name in args.tasks:
        run_one(name)
    print("MATCHED BBQ POLICY RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
