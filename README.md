# AdaSpec Anonymous Evaluation Release

This anonymous repository contains the evaluation-time code, compact result summaries, and released probe-router weights for reproducing the reported AdaSpec evaluation results. Training scripts are intentionally omitted during review and will be released after the review process.

## Included Files

- `truthqa_eval.py`: TruthfulQA evaluation script with `SVD / GEVD / BYPASS` routing.
- `bbq_eval.py`: BBQ evaluation script with `SVD / GEVD / BYPASS` routing.
- `src/decoding_algorithm/`: inference-time SEA and probe-router decoding implementations.
- `weights/probe_router/probe_router.pt`: final lightweight probe-router weights.
- `results/`: compact summaries for TruthfulQA, BBQ, cross-backbone, and open-ended evaluations.
- `data/truthfulqa/` and `data/BBQ/`: evaluation data used by the provided scripts.
- `projectors/`: placeholder directory for SVD/GEVD projector weights.

The anonymous review release does not include these training scripts:

```text
svd_train.py
gevd_train.py
probe_feature.py
probe_train.py
```

## Environment

```bash
pip install -r requirements.txt
```

The base language model must be downloaded separately and passed with `--model-name`.

## Weights

The released probe-router weights are included here:

```text
weights/probe_router/probe_router.pt
```

The SVD/GEVD projector files are large and should be provided through Git LFS, GitHub Releases, or an anonymous artifact link during review. Put them at the following paths, or replace the command-line arguments with your own absolute paths:

```text
projectors/svd/no_mean_sub_uu_positive.pt
projectors/svd/no_mean_sub_uu_negative.pt
projectors/gevd/no_mean_sub_uu_positive.pt
projectors/gevd/no_mean_sub_uu_negative.pt
```

## Default Configuration

```text
feature_source = mlp_output
pooling = mean_tokens
probe_layer = 30
edit_layers = last-L
L = 2
lower_threshold = 0.40
upper_threshold = 0.60
```

## Main Results

### Llama-2-7B

TruthfulQA:

```text
MC1 = 43.21
MC2 = 59.87
MC3 = 30.66
```

BBQ:

```text
Accuracy = 45.20
Unknown answer rate = 14.81
Stereotypical response rate = 43.34
Bias score = 7.72
```

### Cross-Backbone Results

| Model | TruthfulQA MC1 | TruthfulQA MC2 | TruthfulQA MC3 | BBQ Acc | BBQ Unk | BBQ SR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-2-7B | 43.21 | 59.87 | 30.66 | 45.20 | 14.81 | 43.34 |
| Gemma-7B | 35.37 | 53.93 | 26.80 | 43.19 | 26.76 | 33.15 |
| Mistral-7B | 56.30 | 72.41 | 45.19 | 94.82 | 0.78 | 4.25 |

### TruthfulQA Open-Ended Generation

```text
True = 62.18
Info = 76.74
True * Info = 40.88
Number of examples = 817
```

## Example: TruthfulQA Evaluation

```bash
python truthqa_eval.py \
  --model-name /path/to/Llama-2-7b-chat-hf \
  --data-path data/truthfulqa \
  --output-path results/truthfulqa/adaspec_eval.json \
  --is-chat \
  --svd-positive-proj projectors/svd/no_mean_sub_uu_positive.pt \
  --svd-negative-proj projectors/svd/no_mean_sub_uu_negative.pt \
  --gevd-positive-proj projectors/gevd/no_mean_sub_uu_positive.pt \
  --gevd-negative-proj projectors/gevd/no_mean_sub_uu_negative.pt \
  --probe-path weights/probe_router/probe_router.pt \
  --apply-sea-layers last-L \
  --L 2 \
  --combine-sea-embeddings l2_norm \
  --lower-threshold 0.40 \
  --upper-threshold 0.60
```

## Example: BBQ Evaluation

```bash
python bbq_eval.py \
  --model-name /path/to/Llama-2-7b-chat-hf \
  --data-path data/BBQ \
  --output-path results/bbq/adaspec_eval.json \
  --is-chat \
  --svd-positive-proj projectors/svd/no_mean_sub_uu_positive.pt \
  --svd-negative-proj projectors/svd/no_mean_sub_uu_negative.pt \
  --gevd-positive-proj projectors/gevd/no_mean_sub_uu_positive.pt \
  --gevd-negative-proj projectors/gevd/no_mean_sub_uu_negative.pt \
  --probe-path weights/probe_router/probe_router.pt \
  --apply-sea-layers last-L \
  --L 2 \
  --combine-sea-embeddings l2_norm \
  --lower-threshold 0.40 \
  --upper-threshold 0.60
```
