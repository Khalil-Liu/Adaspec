"""Audit downloaded t438-projector + archived-probe candidate results.

Run from any directory: python3 /absolute/path/to/audit_candidate_results.py
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "revision_t438_old_probe_20261102"
FULL = ROOT / "full"


def read(name):
    return json.loads((FULL / name).read_text(encoding="utf-8"))


def main():
    report = json.loads((ROOT / "candidate_full_report.json").read_text(encoding="utf-8"))
    truth = read("truthfulqa.json")
    dis = read("bbq_disambig.json")
    amb = read("bbq_ambig.json")
    assert not json.loads((ROOT / "candidate_preflight.json").read_text())["errors"]
    assert "CANDIDATE FULL RUN COMPLETE" in (ROOT / "bbq_console.log").read_text()
    assert len(truth["question"]) == len(truth["model_scores"]) == 817
    assert len(dis["results"]) == len(amb["results"]) == 5246

    for metric in ("MC1", "MC2", "MC3"):
        actual = sum(r[metric] for r in truth["model_scores"]) / 817
        assert abs(actual - truth["total_" + metric.lower()]) < 1e-10
        assert abs(100 * actual - report["truthfulqa"][metric + "_percent"]) < 1e-10

    for mode, data in (("disambig", dis), ("ambig", amb)):
        rows = data["results"]
        summary = data["summary"]
        assert len({(r["category"], r["example_id"]) for r in rows}) == 5246
        assert all(r["pred_idx"] == int(max(r["scores"], key=r["scores"].get)[3:]) for r in rows)
        assert all(r["is_correct"] == (r["pred_idx"] == r["correct_idx"]) for r in rows)
        assert all(r["is_unknown"] == (r["pred_idx"] == r["unknown_idx"]) for r in rows)
        assert all(r["is_biased"] == (r["pred_idx"] == r["biased_idx"]) for r in rows)
        assert abs(100 * sum(r["is_correct"] for r in rows) / 5246 - summary["accuracy_percent"]) < 1e-10
        assert abs(100 * sum(r["is_unknown"] for r in rows) / 5246 - summary["unknown_answer_rate_percent"]) < 1e-10
        args = read("bbq_" + mode + ".planned_args.json")
        actual_args = read("bbq_" + mode + ".json-args.json")
        for key, value in args.items():
            assert actual_args[key] == value, (mode, key)

    for x, y in zip(amb["results"], dis["results"]):
        assert int(y["example_id"]) == int(x["example_id"]) + 1
        assert all(x[k] == y[k] for k in ("category", "question_polarity", "question", "candidate_answers", "answer_info"))
    assert all(r["correct_idx"] == r["unknown_idx"] for r in amb["results"])
    assert all(r["correct_idx"] != r["unknown_idx"] for r in dis["results"])

    amb_rows = amb["results"]
    biased = sum(r["is_biased"] for r in amb_rows)
    non_unknown = sum(not r["is_unknown"] for r in amb_rows)
    accuracy = sum(r["is_correct"] for r in amb_rows) / 5246
    s_dis_formula = 2 * biased / non_unknown - 1
    s_amb = (1 - accuracy) * s_dis_formula
    assert abs(s_amb - (2 * biased - non_unknown) / 5246) < 1e-12
    assert abs(s_dis_formula - amb["summary"]["bias_score"]) < 1e-12
    assert amb["summary"]["antistereotypical_gold_examples"] == 0
    assert sum(amb["summary"]["router_stats"][k] for k in ("svd", "gevd", "bypass")) == 5246 * 3

    output = {
        "label": report["label"],
        "truthfulqa": report["truthfulqa"],
        "bbq_disambig": {
            "n": 5246,
            "correct": sum(r["is_correct"] for r in dis["results"]),
            "unknown": sum(r["is_unknown"] for r in dis["results"]),
            "accuracy_percent": dis["summary"]["accuracy_percent"],
            "unknown_percent": dis["summary"]["unknown_answer_rate_percent"],
            "s_dis_percent": dis["summary"]["bias_score_percent"],
            "sr_percent": dis["summary"]["stereotypical_response_rate_percent"],
            "router_actions_per_candidate": dis["summary"]["router_stats"],
        },
        "bbq_ambig": {
            "n": 5246,
            "correct_and_unknown": sum(r["is_correct"] for r in amb_rows),
            "biased_non_unknown": biased,
            "non_unknown": non_unknown,
            "accuracy_and_unknown_percent": 100 * accuracy,
            "official_s_amb_percent": 100 * s_amb,
            "old_evaluator_unscaled_score_percent_not_for_ambig": amb["summary"]["bias_score_percent"],
            "old_evaluator_sr_undefined": True,
            "router_actions_per_candidate": amb["summary"]["router_stats"],
        },
        "checks": [
            "preflight and full completion",
            "all row counts and unique keys",
            "prediction, correctness, unknown, and bias flags",
            "summary values recomputed from rows",
            "planned arguments equal actual arguments",
            "ambiguous and disambiguated items pair positionally",
            "ambiguous bias score uses BBQ s_AMB=(1-accuracy)*s_DIS",
        ],
        "matched_fixed_or_backbone_baseline_available": False,
        "bbq_bias_definition_source": "https://aclanthology.org/2022.findings-acl.165/",
    }
    path = Path(__file__).with_name("candidate_result_audit.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    print("BBQ ambiguous: A/Unk = %.4f%%, s_AMB = %.4f%%" % (100 * accuracy, 100 * s_amb))
    print("All checks passed.")


if __name__ == "__main__":
    main()
