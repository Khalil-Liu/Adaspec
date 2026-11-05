"""Recompute Table 3 paired statistics from the saved SEA/AdaSpec records."""

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEA = ROOT / "outputs" / "table3_paired_truthfulqa" / "sea.json"
ADASPEC = ROOT / "outputs" / "table3_paired_truthfulqa" / "adaspec.json"
OUTPUT = ROOT / "outputs" / "table3_paired_truthfulqa" / "paired_statistics.json"
B = 10_000
SEED = 42


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def interval(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def exact_mcnemar(adaspec_only, sea_only):
    n = adaspec_only + sea_only
    if n == 0:
        return 1.0
    smaller = min(adaspec_only, sea_only)
    lower_tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2 * lower_tail)


def main():
    sea = load(SEA)
    adaspec = load(ADASPEC)
    assert sea["question"] == adaspec["question"]
    metrics = ["MC1", "MC2", "MC3"]
    a = np.array([[row[m] for m in metrics] for row in sea["model_scores"]])
    b = np.array([[row[m] for m in metrics] for row in adaspec["model_scores"]])
    assert len(a) == 817
    delta = b - a
    rng = np.random.default_rng(SEED)

    boot = []
    for _ in range(B // 500):
        indices = rng.integers(0, len(a), size=(500, len(a)))
        boot.append(delta[indices].mean(axis=1))
    boot = np.concatenate(boot)

    results = {}
    raw_p = []
    for j, metric in enumerate(metrics):
        observed = abs(delta[:, j].mean())
        if metric == "MC1":
            adaspec_only = int(((a[:, j] == 0) & (b[:, j] == 1)).sum())
            sea_only = int(((a[:, j] == 1) & (b[:, j] == 0)).sum())
            p_value = exact_mcnemar(adaspec_only, sea_only)
            test = "exact two-sided McNemar"
        else:
            exceeds = 0
            for _ in range(B // 500):
                signs = rng.choice([-1, 1], size=(500, len(a)))
                exceeds += int((np.abs((signs * delta[:, j]).mean(axis=1)) >= observed - 1e-15).sum())
            p_value = (exceeds + 1) / (B + 1)
            test = "paired two-sided sign-flip Monte Carlo with plus-one correction"
        raw_p.append(p_value)
        results[metric] = {
            "SEA_percent": float(a[:, j].mean() * 100),
            "AdaSpec_percent": float(b[:, j].mean() * 100),
            "delta_pp": float(delta[:, j].mean() * 100),
            "paired_delta_95CI_pp": interval(boot[:, j] * 100),
            "p_raw": p_value,
            "test": test,
        }

    for metric, adjusted in zip(metrics, holm(raw_p)):
        results[metric]["p_Holm_three_metrics"] = float(adjusted)

    report = {
        "n": len(a),
        "bootstrap_replicates": B,
        "seed": SEED,
        "resampling_unit": "question",
        "metrics": results,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
