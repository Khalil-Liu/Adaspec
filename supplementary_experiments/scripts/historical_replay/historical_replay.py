"""Replay the archived AdaSpec TruthfulQA and BBQ runs with the old framework.

Run from revision/historical_replay_20261102 on the server. Dry check is the
default. A 10-item smoke comparison must pass before the full run can start.
Existing historical outputs are never overwritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "historical_replay_configs"
OUT = ROOT / "exp_results" / "revision_historical_replay_20261102"
MODEL_DEFAULT = Path(os.environ.get("MODEL", "/path/to/Llama-2-7b-chat-hf"))
TASKS = ("truthfulqa", "bbq_disambig", "bbq_ambig")
SOURCE_HASHES = {
    "src/benchmark_evaluation/truthfulqa_eval_probe_router_abstain.py": "1c727ea10283a2ef3071ad501088854712ca2f39dce49f8245053f05e3e706d4",
    "src/benchmark_evaluation/bbq_bias_metrics_eval_probe_router_abstain.py": "c22b4f0a828084e411ebca1bcd4fb422f40308204271408f423fa5c4373c9ea9",
    "src/decoding_algorithm/inference_probe_router_abstain.py": (
        "f36338959dc4e17a0b2393b557b8f3c32fc1fc2104b992aadcb0c7c85c3496ae",
        "1d626747983725625d2fa2caad3b744aaff4412a809be13523a35ee8ffa9f1ab",
    ),
}
DATA_HASHES = {
    "data/truthfulqa/TruthfulQA.csv": "967b82fb1fb6274e4971c6c80caa9d04f844c512b1033c146f26c78270cd384b",
    "data/BBQ/disambig/All-test-disambig.jsonl": "2fb0942d7a1f5d7fe9b7c83b1c89e611c629812987e86c011f62d2952913f6e9",
    "data/BBQ/supplemental/additional_metadata.csv": "f36708416b0e7adb81b47ad1926f9c39c2beb611702b73b01e06c9b6c9ffbd3d",
    "data/BBQ/ambig/All-test-ambig.jsonl": "b2d2c5ff8ecec96eb0482e43f40f791bb02020f32acda1ff9e6e5a12c67634de",
}
PROBE_HASH = "83d21cfcd3afc9d64c3b0d8bf1fe2a55f246634a7934e84e1de21ddd233d4ad9"
PATH_KEYS = ("svd_positive_proj", "svd_negative_proj", "gevd_positive_proj", "gevd_negative_proj", "probe_path")
COMMON_KEYS = (
    "model_name", "dataset_name", "num_gpus", "amateur_model_nums_gpus",
    "max_gpu_memory", "device", "data_path", "output_path", "is_chat",
    "max_new_tokens", "repetition_penalty", "relative_top", "relative_top_value",
    "svd_positive_proj", "svd_negative_proj", "gevd_positive_proj",
    "gevd_negative_proj", "probe_path", "lower_threshold", "upper_threshold",
    "apply_sea_layers", "L", "combine_sea_embeddings", "feature_function",
)
SPECIAL_FLAGS = {
    "max_gpu_memory": "--max_gpu_memory",
    "L": "--L",
    "top_p": "--top_p",
    "top_k": "--top_k",
    "repetition_penalty": "--repetition_penalty",
    "relative_top": "--relative_top",
    "relative_top_value": "--relative_top_value",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def old_path(value):
    prefix = "/workspace/sea-llm/"
    if value.startswith(prefix):
        return str(ROOT / value[len(prefix):])
    if value.startswith("/workspace/"):
        raise ValueError("Unrecognized archived asset path: " + value)
    return str(ROOT / value)


def archived(task):
    name = "truthfulqa_adaspec_args.json" if task == "truthfulqa" else "bbq_adaspec_args.json"
    return load(CONFIG / name)


def runner(task):
    name = "truthfulqa_eval_probe_router_abstain.py" if task == "truthfulqa" else "bbq_bias_metrics_eval_probe_router_abstain.py"
    return ROOT / "src" / "benchmark_evaluation" / name


def output(task, smoke):
    directory = OUT / ("smoke" if smoke else "full")
    return directory / (task + ".json")


def resolved_args(task, model, smoke):
    old = archived(task)
    current = {k: old[k] for k in COMMON_KEYS if k in old}
    current["model_name"] = str(model)
    current["data_path"] = str(ROOT / ("data/truthfulqa" if task == "truthfulqa" else "data/BBQ"))
    current["output_path"] = str(output(task, smoke))
    for key in PATH_KEYS:
        current[key] = old_path(old[key])
    if task != "truthfulqa":
        current["bbq_mode"] = "ambig" if task == "bbq_ambig" else "disambig"
        current["additional_metadata_path"] = str(ROOT / "data/BBQ/supplemental/additional_metadata.csv")
    if task == "truthfulqa":
        for key in ("top_p", "top_k", "temperature"):
            current[key] = old[key]
    if smoke:
        current["debug"] = True
    return current


def command(task, args):
    cmd = [sys.executable, "-u", str(runner(task))]
    for key, value in args.items():
        if value is None or value is False:
            continue
        flag = SPECIAL_FLAGS.get(key, "--" + key.replace("_", "-"))
        if value is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def preflight(tasks, model):
    errors = []
    evidence = {"time_utc": datetime.now(timezone.utc).isoformat(), "model": str(model), "checks": {}}
    for name, expected in SOURCE_HASHES.items():
        path = ROOT / name
        actual = digest(path) if path.is_file() else None
        allowed = expected if isinstance(expected, tuple) else (expected,)
        evidence["checks"][name] = {"allowed": allowed, "actual": actual}
        if actual not in allowed:
            errors.append("Old framework source missing or changed: " + name)
    if not (model / "config.json").is_file():
        errors.append("Model config missing: " + str(model / "config.json"))
    required_data = set()
    if "truthfulqa" in tasks:
        required_data.add("data/truthfulqa/TruthfulQA.csv")
    if any(t.startswith("bbq") for t in tasks):
        required_data.update(("data/BBQ/disambig/All-test-disambig.jsonl", "data/BBQ/supplemental/additional_metadata.csv"))
    if "bbq_ambig" in tasks:
        required_data.add("data/BBQ/ambig/All-test-ambig.jsonl")
    for name in sorted(required_data):
        path = ROOT / name
        actual = digest(path) if path.is_file() else None
        evidence["checks"][name] = {"expected": DATA_HASHES[name], "actual": actual}
        if actual != DATA_HASHES[name]:
            errors.append("Dataset missing or SHA256 differs: " + name)
    assets = {}
    for task in tasks:
        args = resolved_args(task, model, False)
        for key in PATH_KEYS:
            path = Path(args[key])
            assets[key] = path
            if not path.is_file():
                errors.append("Archived " + key + " missing: " + str(path))
    for key, path in assets.items():
        if path.is_file():
            evidence["checks"][key] = {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
    probe = assets.get("probe_path")
    if probe and probe.is_file() and digest(probe) != PROBE_HASH:
        errors.append("Old probe SHA256 differs from archived local probe: " + str(probe))
    if "truthfulqa" in tasks and "bbq_disambig" in tasks:
        truth, bbq = resolved_args("truthfulqa", model, False), resolved_args("bbq_disambig", model, False)
        for key in PATH_KEYS:
            if truth[key] != bbq[key]:
                errors.append("Archived TruthfulQA/BBQ asset mismatch: " + key)
    evidence["errors"] = sorted(set(errors))
    return evidence


def stage_ambiguous_input():
    source = ROOT / "data/BBQ/ambig/All-test-ambig.jsonl"
    target = ROOT / "data/BBQ/disambig/All-test-ambig.jsonl"
    if target.exists() or target.is_symlink():
        if digest(target) != DATA_HASHES["data/BBQ/ambig/All-test-ambig.jsonl"]:
            raise RuntimeError("Old evaluator's ambiguous input path exists with different bytes: " + str(target))
        return
    target.symlink_to(source)
    print("Linked old evaluator's ambiguous input path:", target, flush=True)


def run_one(task, model, smoke):
    target = output(task, smoke)
    if target.exists() or Path(str(target) + "-args.json").exists():
        raise FileExistsError("Preserving existing replay result: " + str(target))
    if task == "bbq_ambig":
        stage_ambiguous_input()
    target.parent.mkdir(parents=True, exist_ok=True)
    args = resolved_args(task, model, smoke)
    cmd = command(task, args)
    save(target.with_suffix(".planned_args.json"), args)
    print("RUN", task, "->", target, flush=True)
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "1")
    with target.with_suffix(".log").open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{task} exited {result.returncode}; inspect {target.with_suffix('.log')}")
    if not target.is_file():
        raise RuntimeError("Runner exited without result: " + str(target))
    print("DONE", task, target.stat().st_size, "bytes", flush=True)


def compare_smoke():
    fixture = load(CONFIG / "archived_fixture.json")
    truth = load(output("truthfulqa", True))
    bbq = load(output("bbq_disambig", True))
    report = {"truthfulqa": {}, "bbq_disambig": {}, "pass": False}
    questions = [x["question"] for x in truth["question"]]
    report["truthfulqa"]["questions_match"] = questions == fixture["truthfulqa"]["questions"]
    metric_diffs = {}
    for key in ("MC1", "MC2", "MC3"):
        actual = [x[key] for x in truth["model_scores"]]
        reference = [x[key] for x in fixture["truthfulqa"]["metrics"]]
        metric_diffs[key] = max(abs(a - b) for a, b in zip(actual, reference)) if len(actual) == len(reference) == 10 else None
    report["truthfulqa"]["max_abs_metric_differences"] = metric_diffs
    results = bbq["results"]
    report["bbq_disambig"]["keys_match"] = [[x["category"], x["example_id"]] for x in results] == fixture["bbq_disambig"]["keys"]
    report["bbq_disambig"]["predictions_match"] = [x["pred_idx"] for x in results] == fixture["bbq_disambig"]["pred_idx"]
    report["pass"] = (
        report["truthfulqa"]["questions_match"]
        and metric_diffs["MC1"] == 0
        and metric_diffs["MC3"] == 0
        and metric_diffs["MC2"] is not None and metric_diffs["MC2"] < 0.02
        and report["bbq_disambig"]["keys_match"]
        and report["bbq_disambig"]["predictions_match"]
    )
    save(OUT / "smoke_comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass"]:
        raise RuntimeError("Historical smoke mismatch; do not start full runs until investigated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("check", "smoke", "full"), default="check")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--model-path", type=Path, default=MODEL_DEFAULT)
    args = parser.parse_args()
    tasks = tuple(args.tasks)
    if len(set(tasks)) != len(tasks):
        raise ValueError("Duplicate task")
    required = ("truthfulqa", "bbq_disambig") if args.phase == "smoke" else tasks
    evidence = preflight(required, args.model_path)
    save(OUT / "historical_preflight.json", evidence)
    print(json.dumps({"model": evidence["model"], "errors": evidence["errors"], "assets": evidence["checks"]}, indent=2), flush=True)
    if evidence["errors"]:
        raise SystemExit("Preflight failed. Do not substitute t438 files for historical assets.")
    if args.phase == "check":
        print("Historical preflight passed. Next: --phase smoke", flush=True)
        return
    if args.phase == "smoke":
        for task in required:
            run_one(task, args.model_path, True)
        compare_smoke()
        return
    smoke = OUT / "smoke_comparison.json"
    if not smoke.is_file() or not load(smoke).get("pass"):
        raise RuntimeError("Full run requires a passed historical 10-item smoke comparison")
    for task in tasks:
        run_one(task, args.model_path, False)
    print("HISTORICAL REPLAY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
