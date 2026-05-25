# AdaSpec Anonymous Evaluation Release

This anonymous repository currently provides evaluation-time code, compact result summaries, and released probe-router weights for reproducing the reported AdaSpec evaluation results. More implementation details, including full training scripts, configuration files, and extended reproduction resources, are coming soon and will be released after the review process.

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
