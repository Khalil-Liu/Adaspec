"""Audit downloaded same-protocol BBQ bypass controls and paired differences."""

import json
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE / "revision_t438_old_probe_20261102"
FULL = ROOT / "full"
CONTROL = ROOT / "matched_policies"
N = 5246
REPLICATES = 10000
SEED = 42


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values, q):
    values = sorted(values)
    index = q * (len(values) - 1)
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (index - lo) * (values[hi] - values[lo])


def metrics(v, mode):
    n, ac, ab, uc, ub, bc, bb, nc, nb, sc, sb, anti = v
    result = {
        "accuracy_delta_pp": 100 * (ac - ab) / n,
        "unknown_delta_pp": 100 * (uc - ub) / n,
    }
    if mode == "ambig":
        result["official_s_amb_delta_pp"] = 100 * (2 * (bc - bb) - (nc - nb)) / n
    else:
        result["s_dis_delta_pp"] = 100 * (2 * bc / nc - 2 * bb / nb)
        result["sr_delta_pp"] = 100 * (sc - sb) / anti
    return result


def audit(mode):
    cand = read(FULL / f"bbq_{mode}.json")
    base = read(CONTROL / f"bypass_{mode}.json")
    ca = read(FULL / f"bbq_{mode}.planned_args.json")
    ba = read(CONTROL / f"bypass_{mode}.planned_args.json")
    actual = read(CONTROL / f"bypass_{mode}.json-args.json")
    changed = {k for k in ca if ca[k] != ba.get(k)}
    assert changed == {"lower_threshold", "upper_threshold", "output_path"}, changed
    assert (ba["lower_threshold"], ba["upper_threshold"]) == (0.0, 1.0)
    assert all(actual[k] == v for k, v in ba.items())
    assert len(cand["results"]) == len(base["results"]) == N
    assert base["summary"]["router_stats"] == {
        **{k: base["summary"]["router_stats"][k] for k in base["summary"]["router_stats"] if k not in ("svd", "gevd", "bypass")},
        "svd": 0, "gevd": 0, "bypass": 3 * N,
    }
    immutable = ("example_id", "category", "question_polarity", "question", "context",
                 "correct_idx", "unknown_idx", "biased_idx", "gold_is_antistereotypical",
                 "candidate_answers", "answer_info", "additional_metadata")
    groups = {}
    discordant = [0, 0]
    for x, y in zip(cand["results"], base["results"]):
        assert all(x[k] == y[k] for k in immutable), (mode, x["example_id"])
        for row in (x, y):
            assert row["pred_idx"] == int(max(row["scores"], key=row["scores"].get)[3:])
            assert row["is_correct"] == (row["pred_idx"] == row["correct_idx"])
            assert row["is_unknown"] == (row["pred_idx"] == row["unknown_idx"])
            assert row["is_biased"] == (row["pred_idx"] == row["biased_idx"])
        if x["is_correct"] and not y["is_correct"]:
            discordant[0] += 1
        if y["is_correct"] and not x["is_correct"]:
            discordant[1] += 1
        key = (x["category"], x["question"])
        v = groups.setdefault(key, [0] * 12)
        v[0] += 1
        v[1] += x["is_correct"]
        v[2] += y["is_correct"]
        v[3] += x["is_unknown"]
        v[4] += y["is_unknown"]
        v[5] += x["is_biased"] and not x["is_unknown"]
        v[6] += y["is_biased"] and not y["is_unknown"]
        v[7] += not x["is_unknown"]
        v[8] += not y["is_unknown"]
        v[9] += x["is_biased"] and x["gold_is_antistereotypical"]
        v[10] += y["is_biased"] and y["gold_is_antistereotypical"]
        v[11] += x["gold_is_antistereotypical"]
    totals = [sum(v[i] for v in groups.values()) for i in range(12)]
    assert totals[0] == N
    for data, offset in ((cand, 1), (base, 2)):
        summary = data["summary"]
        assert abs(100 * totals[offset] / N - summary["accuracy_percent"]) < 1e-9
        assert abs(100 * totals[offset + 2] / N - summary["unknown_answer_rate_percent"]) < 1e-9
    values = list(groups.values())
    rng = random.Random(SEED)
    point = metrics(totals, mode)
    draws = {name: [] for name in point}
    for _ in range(REPLICATES):
        sample = [0] * 12
        for _ in values:
            row = values[rng.randrange(len(values))]
            for i, value in enumerate(row):
                sample[i] += value
        result = metrics(sample, mode)
        for name in point:
            draws[name].append(result[name])
    result = {
        "mode": mode,
        "n": N,
        "paired_cluster_definition": "identical (category, question) string",
        "paired_clusters": len(groups),
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": SEED,
        "candidate": {
            "accuracy_percent": cand["summary"]["accuracy_percent"],
            "unknown_percent": cand["summary"]["unknown_answer_rate_percent"],
        },
        "bypass": {
            "accuracy_percent": base["summary"]["accuracy_percent"],
            "unknown_percent": base["summary"]["unknown_answer_rate_percent"],
        },
        "delta_candidate_minus_bypass": {
            name: {"point_pp": value, "paired_cluster_bootstrap_95CI_pp":
                   [percentile(draws[name], .025), percentile(draws[name], .975)]}
            for name, value in point.items()
        },
        "accuracy_discordant_candidate_only_bypass_only": discordant,
        "counts": {
            "candidate_correct": totals[1], "bypass_correct": totals[2],
            "candidate_unknown": totals[3], "bypass_unknown": totals[4],
            "candidate_biased_non_unknown": totals[5], "bypass_biased_non_unknown": totals[6],
            "candidate_non_unknown": totals[7], "bypass_non_unknown": totals[8],
            "candidate_anti_gold_biased": totals[9], "bypass_anti_gold_biased": totals[10],
            "anti_gold": totals[11],
        },
    }
    if mode == "ambig":
        assert totals[11] == 0
        for label, biased, nonunk in (("candidate", totals[5], totals[7]),
                                      ("bypass", totals[6], totals[8])):
            result[label]["official_s_amb_percent"] = 100 * (2 * biased - nonunk) / N
        result["ambiguous_sr_defined"] = False
    else:
        for label, biased, nonunk, sr in (("candidate", totals[5], totals[7], totals[9]),
                                          ("bypass", totals[6], totals[8], totals[10])):
            result[label]["s_dis_percent"] = 100 * (2 * biased / nonunk - 1)
            result[label]["sr_percent"] = 100 * sr / totals[11]
    return result


def main():
    assert not read(CONTROL / "preflight.json")["errors"]
    assert "MATCHED BBQ POLICY RUN COMPLETE" in (CONTROL / "console.log").read_text()
    out = {mode: audit(mode) for mode in ("ambig", "disambig")}
    path = HERE / "matched_bbq_audit.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    for mode, result in out.items():
        print(mode, "paired clusters", result["paired_clusters"])
        for name, value in result["delta_candidate_minus_bypass"].items():
            print(" ", name, round(value["point_pp"], 4),
                  [round(x, 4) for x in value["paired_cluster_bootstrap_95CI_pp"]])
    print("All checks passed.")


if __name__ == "__main__":
    main()
