#!/usr/bin/env python3
"""Resume the current controlled validation without mixing configurations.

Place this file beside run.py on the server. Dry-run by default; --execute runs
the remaining stages. It never reruns an existing model-evaluation stage.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PATH_KEYS = (
    "model_path", "halueval", "truthfulqa", "bbq_disambig", "bbq_ambig",
    "bbq_metadata", "svd_positive", "svd_negative", "gevd_positive",
    "gevd_negative", "projector_indices_json",
)
PROJECTORS = ("svd_positive", "svd_negative", "gevd_positive", "gevd_negative")
DATASETS = ("halueval", "truthfulqa", "bbq_disambig", "bbq_ambig", "bbq_metadata")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolved(path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def normalized_config(config):
    result = dict(config)
    for key in PATH_KEYS:
        if result.get(key):
            result[key] = str(resolved(result[key]))
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_alignment(config, out, verify_hashes):
    if not (HERE / "run.py").is_file():
        raise RuntimeError("Place batch_remaining.py beside revision/validation_20261101/run.py")
    snapshot_path = out / "config_snapshot.json"
    truth_path = out / "truthfulqa.json"
    for path in (snapshot_path, truth_path, out / "trained_router.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required completed stage is missing: {path}")
    expected = normalized_config(config)
    snapshot = normalized_config(read_json(snapshot_path))
    truth = read_json(truth_path)
    truth_config = normalized_config(truth["config"])
    for source_name, actual in (("config_snapshot.json", snapshot), ("truthfulqa.json", truth_config)):
        if actual != expected:
            differing = [key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)]
            raise RuntimeError(f"Configuration mismatch in {source_name}: {sorted(differing)}. Use a new output_dir for a changed protocol.")
    if truth.get("smoke_limit") or len(truth["rows"]) != 817:
        raise RuntimeError("Expected the completed full 817-question TruthfulQA run")
    model = resolved(config["model_path"])
    if not model.is_dir() or not (model / "config.json").is_file():
        raise FileNotFoundError(f"Model directory/config missing: {model}")
    preflight = read_json(out / "preflight.json") if (out / "preflight.json").is_file() else {}
    hashes = {"existing_truthfulqa_json": sha256(truth_path), "verified_inputs": {}}
    if verify_hashes:
        for key in DATASETS + PROJECTORS:
            path = resolved(config[key])
            if not path.is_file():
                if key == "bbq_ambig":
                    continue
                raise FileNotFoundError(f"Configured input missing: {key}: {path}")
            digest = sha256(path)
            old = preflight.get("dataset_sha256", {}).get(key) or preflight.get("projector_sha256", {}).get(key)
            if old and digest != old:
                raise RuntimeError(f"Input bytes changed since preflight: {key}: {path}")
            hashes["verified_inputs"][key] = digest
    return truth, hashes


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def bypass_diagnostic(truth, source_hash, seed=42, replicates=10000):
    rows = truth["rows"]
    n = len(rows)
    skipped = []
    acted = []
    delta = []
    base_scores = []
    no_bypass_scores = []
    actual_scores = []
    for row in rows:
        actions = row["actions"]
        scores = row["scores"]
        base = scores[actions["backbone"]]
        counterfactual = scores[actions["router_no_bypass"]]
        actual = scores[actions["adaspec"]]
        target = skipped if actions["adaspec"] == 0 else acted
        target.append((base["MC1"], counterfactual["MC1"]))
        base_scores.append(base)
        no_bypass_scores.append(counterfactual)
        actual_scores.append(actual)
        delta.append([actual[k] - counterfactual[k] for k in ("MC1", "MC2", "MC3")])

    def outcome(group):
        return {
            "n": len(group),
            "harm_if_forced_edit": sum(edit < base for base, edit in group),
            "help_if_forced_edit": sum(edit > base for base, edit in group),
            "unchanged_if_forced_edit": sum(edit == base for base, edit in group),
        }

    rng = random.Random(seed)
    means = [sum(row[j] for row in delta) / n * 100 for j in range(3)]
    draws = [[], [], []]
    for _ in range(replicates):
        total = [0.0, 0.0, 0.0]
        for _ in range(n):
            row = delta[rng.randrange(n)]
            for j in range(3):
                total[j] += row[j]
        for j in range(3):
            draws[j].append(total[j] / n * 100)

    # Random abstention at exactly the observed bypass count, against the
    # same no-bypass decisions. This is exploratory, not a tuned policy.
    gains = [base_scores[i]["MC1"] - no_bypass_scores[i]["MC1"] for i in range(n)]
    no_bypass_total = sum(s["MC1"] for s in no_bypass_scores)
    actual_total = sum(s["MC1"] for s in actual_scores)
    random_totals = []
    for _ in range(replicates):
        chosen = rng.sample(range(n), len(skipped))
        random_totals.append((no_bypass_total + sum(gains[i] for i in chosen)) / n * 100)
    random_mean = sum(random_totals) / len(random_totals)

    return {
        "source_sha256": source_hash,
        "scope": "exploratory held-out counterfactual diagnostic; no new model inference",
        "n_questions": n,
        "bypass": outcome(skipped),
        "edited": outcome(acted),
        "adaspec_minus_no_bypass": {
            key: {"delta_pp": means[j], "paired_bootstrap_95CI_pp": [quantile(draws[j], .025), quantile(draws[j], .975)]}
            for j, key in enumerate(("MC1", "MC2", "MC3"))
        },
        "random_bypass_same_coverage_MC1": {
            "replicates": replicates,
            "seed": seed,
            "observed_adaspec_pct": actual_total / n * 100,
            "random_mean_pct": random_mean,
            "random_95_interval_pct": [quantile(random_totals, .025), quantile(random_totals, .975)],
            "fraction_random_at_least_observed": sum(x >= actual_total / n * 100 for x in random_totals) / replicates,
        },
    }


def backup(path):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(path.name + ".bak_" + stamp)
    if target.exists():
        raise FileExistsError(target)
    shutil.move(str(path), str(target))
    print("Archived stale summary:", target, flush=True)


def run_stage(task, config_path, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, "-u", str(HERE / "run.py"), "--config", str(config_path), "--tasks", task]
    print("RUN:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"Stage {task} failed with exit code {result.returncode}; inspect its log before resuming")


def write_report(out, available, diagnostic, ambiguous_missing):
    lines = [
        "# Controlled validation batch report", "",
        "Protocol: current t438 configuration; separate from the historical manuscript run.",
        "", "## Completed evaluation outputs", "",
    ]
    for name in available:
        summary = out / (name + "_summary.json")
        if not summary.is_file():
            continue
        data = read_json(summary)
        lines.append(f"### {name} (n = {len(read_json(out / (name + '.json'))['rows'])})")
        lines.append("")
        if name == "truthfulqa":
            lines.extend(["| Policy | MC1 | MC2 | MC3 |", "|---|---:|---:|---:|"])
            for policy in ("backbone", "fixed_svd", "fixed_gevd", "best_fixed_validation", "random_binary", "router_no_bypass", "adaspec", "direct_threeway"):
                m = data[policy]["metrics"]
                lines.append(f"| {policy} | {m['MC1']*100:.2f} | {m['MC2']*100:.2f} | {m['MC3']*100:.2f} |")
            preference = data["heldout_editor_preference"]
            lines.extend(["", f"Editor-preference subset: n={preference['n']}, accuracy={preference['accuracy']*100:.2f}%, training-majority baseline={preference['majority_training_baseline_accuracy']*100:.2f}%."])
        else:
            lines.extend(["| Policy | Acc | Unk | signed BS | SR |", "|---|---:|---:|---:|---:|"])
            for policy in ("backbone", "fixed_svd", "fixed_gevd", "router_no_bypass", "adaspec"):
                m = data[policy]["metrics"]
                def fmt(key):
                    return "undefined" if m.get(key) is None else f"{m[key]*100:.2f}"
                lines.append(f"| {policy} | {fmt('MC1')} | {fmt('Unk')} | {fmt('BS')} | {fmt('SR')} |")
        lines.append("")

    lines.extend(["## Bypass diagnostic on TruthfulQA", ""])
    for group in ("bypass", "edited"):
        d = diagnostic[group]
        lines.append(f"- {group}: n={d['n']}; forced no-bypass would harm {d['harm_if_forced_edit']}, help {d['help_if_forced_edit']}, leave unchanged {d['unchanged_if_forced_edit']} MC1 outcomes.")
    for metric, result in diagnostic["adaspec_minus_no_bypass"].items():
        lo, hi = result["paired_bootstrap_95CI_pp"]
        lines.append(f"- AdaSpec minus no bypass {metric}: {result['delta_pp']:+.3f} pp; paired 95% CI [{lo:+.3f}, {hi:+.3f}] pp.")
    rb = diagnostic["random_bypass_same_coverage_MC1"]
    lines.append(f"- Random bypass at the same coverage: mean MC1 {rb['random_mean_pct']:.2f}%; observed AdaSpec {rb['observed_adaspec_pct']:.2f}%; random 95% interval [{rb['random_95_interval_pct'][0]:.2f}, {rb['random_95_interval_pct'][1]:.2f}]%. Exploratory comparison.")

    stats = read_json(out / "paired_statistics.json")
    lines.extend(["", "## Paired statistics against validation-selected fixed editor", ""])
    for name, value in stats.items():
        lines.append(f"### {name}")
        lines.append("")
        for metric, result in value["policies"]["adaspec"].items():
            interval = result["paired_delta_95CI_pp"]
            ci = "undefined" if interval is None else f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"
            delta = "undefined" if result["delta_pp"] is None else f"{result['delta_pp']:+.3f} pp"
            lines.append(f"- {metric}: {delta}; paired 95% CI {ci}; Holm p={result.get('p_Holm_three_metrics', 'not tested')}.")
        lines.append("")
    if ambiguous_missing:
        lines.append("BBQ ambiguous input is missing; this batch is partial for R3.6. Rerun the same script after supplying that dataset.\n")
    lines.append("The bypass diagnostic is exploratory and does not by itself establish probe robustness.\n")
    (out / "batch_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config_router_union.json")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--execute", action="store_true", help="Run remaining stages; default only prints the plan")
    parser.add_argument("--include-bbq", action="store_true", help="Also run missing BBQ inference under this same t438 protocol")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    out = resolved(config["output_dir"])
    truth, hashes = check_alignment(config, out, verify_hashes=args.execute)
    pending = []
    for task in ("bbq_disambig", "bbq_ambig"):
        raw = out / (task + ".json")
        summary = out / (task + "_summary.json")
        source = resolved(config[task])
        if raw.exists() != summary.exists():
            raise RuntimeError(f"Partial stage output for {task}; inspect {raw} and {summary} before resuming")
        if raw.exists():
            print("SKIP completed:", task)
        elif source.is_file() and args.include_bbq:
            pending.append(task)
        elif source.is_file():
            print("DEFER GPU stage (add --include-bbq):", task)
        else:
            print("SKIP missing input:", task, source)
    print("Planned GPU stages:", pending or "none")
    print("Planned CPU stages: bypass diagnostic, paired statistics refresh if needed, combined summary refresh")
    if not args.execute:
        print("Dry run only. Add --execute to run the plan.")
        return

    for task in pending:
        run_stage(task, config_path, args.gpu)

    # Re-read the source after inference; it is intentionally not regenerated.
    report = bypass_diagnostic(truth, hashes["existing_truthfulqa_json"], config["seed"])
    write_json(out / "bypass_diagnostic.json", report)
    print("Saved:", out / "bypass_diagnostic.json", flush=True)

    available = [name for name in ("truthfulqa", "bbq_disambig", "bbq_ambig") if (out / (name + ".json")).is_file()]
    stats_path = out / "paired_statistics.json"
    existing = read_json(stats_path) if stats_path.is_file() else {}
    if set(existing) != set(available):
        if stats_path.exists():
            backup(stats_path)
        run_stage("statistics", config_path, args.gpu)
    else:
        print("SKIP current: statistics")

    combined_path = out / "combined_summary.json"
    summary_files = (
        "router_test_report.json", "truthfulqa_summary.json",
        "bbq_disambig_summary.json", "bbq_ambig_summary.json",
    )
    expected = {
        name: read_json(out / name) if (out / name).is_file() else {"status": "not produced"}
        for name in summary_files
    }
    if not combined_path.is_file() or read_json(combined_path) != expected:
        if combined_path.exists():
            backup(combined_path)
        run_stage("summarize", config_path, args.gpu)
    else:
        print("SKIP current: summarize")

    manifest = {
        "config_path": str(config_path),
        "output_dir": str(out),
        "config_sha256": sha256(config_path),
        "alignment": "config snapshot and complete TruthfulQA record match current config",
        "hashes": hashes,
        "completed": [name for name in available if (out / (name + "_summary.json")).is_file()],
        "ambiguous_missing": not resolved(config["bbq_ambig"]).is_file(),
        "note": "This validates one controlled t438 protocol; it is not a reproduction of the manuscript's historical protocol.",
    }
    write_json(out / "batch_manifest.json", manifest)
    write_report(out, available, report, manifest["ambiguous_missing"])
    bundle = out / "review_bundle.zip"
    included = [
        config_path, out / "config_snapshot.json", out / "preflight.json",
        out / "router_pool.json", out / "training_eligibility.json",
        out / "trained_router.json", out / "router_test_report.json",
        out / "paired_statistics.json", out / "bypass_diagnostic.json",
        out / "combined_summary.json", out / "batch_manifest.json",
        out / "batch_report.md",
    ]
    for name in available:
        included.extend((out / (name + ".json"), out / (name + "_summary.json")))
    included.extend(sorted(out.glob("log_*.txt")))
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in included:
            if path.is_file():
                archive.write(path, arcname=path.name)
    print("BATCH COMPLETE. Read:", out / "batch_report.md", "Download:", bundle, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("BATCH STOPPED:", exc, file=sys.stderr, flush=True)
        raise SystemExit(1)
