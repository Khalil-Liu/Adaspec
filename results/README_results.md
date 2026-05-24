# Included Result Summaries

This folder contains compact result summaries copied from the local experiment archive.

## Cross-Backbone AdaSpec

Files:

- `cross_backbone/adaspec_cross_backbone_summary.json`
- `cross_backbone/adaspec_cross_backbone_summary.csv`

The rows correspond to the AdaSpec lines in the cross-backbone table:

| Model | TruthfulQA MC1 | TruthfulQA MC2 | BBQ Acc | BBQ Unk | BBQ SR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-2-7B | 43.21 | 59.87 | 45.20 | 14.81 | 43.34 |
| Gemma-7B | 35.37 | 53.93 | 43.19 | 26.76 | 33.15 |
| Mistral-7B | 56.30 | 72.41 | 94.82 | 0.78 | 4.25 |

## TruthfulQA Open-Ended Generation

Files:

- `openended/truthfulqa_openended_adaspec_summary.json`
- `openended/judged_probe_nochat_summary.original.json`

Open-ended AdaSpec result:

| True | Info | True*Info |
| ---: | ---: | ---: |
| 62.18 | 76.74 | 40.88 |

Large per-example result files are not copied here; this directory keeps only table-level summaries and source-path provenance.
