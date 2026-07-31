# AdaSpec: Adaptive Spectral Editing for Large Language Model Alignment

AdaSpec is an inference-time LLM alignment framework. It adaptively routes each input to an SVD editor, a GEVD editor, or bypass. The two editors are complementary: SVD preserves dominant representation structure, while GEVD emphasizes class-discriminative spectral directions.

## Updates

- (2026-07-31) Released inference, evaluation, result artifacts, and manuscript visualizations.
- Training scripts, configurations, and extended reproduction resources are coming soon after the review process.

## Highlights

AdaSpec makes spectral activation editing adaptive to the input rather than applying one fixed editor to every prompt. It combines two complementary spectral editors: the SVD editor preserves dominant representation directions learned from the aligned demonstrations, while the GEVD editor emphasizes directions that discriminate aligned and misaligned representations. A lightweight probe estimates the preferred editor from pre-edit hidden states. High-confidence predictions are routed to the selected editor, whereas uncertain cases bypass editing, reducing unnecessary interventions. The complete procedure operates at inference time and keeps the backbone frozen.

![AdaSpec overview](https://raw.githubusercontent.com/Khalil-Liu/Adaspec/main/assets/method.jpg)

## Results

### Table 1: TruthfulQA Comparison with State-of-the-Art Methods

Higher is better for all metrics. `True*Info` is the joint TruthfulQA open-ended score.

![Table 1: TruthfulQA comparison](https://raw.githubusercontent.com/Khalil-Liu/Adaspec/main/assets/table1.png)

### Table 2: Generality Across Backbones

![Table 2: Generality across backbones](https://raw.githubusercontent.com/Khalil-Liu/Adaspec/main/assets/table2.png)

<p align="center">
  <img src="https://raw.githubusercontent.com/Khalil-Liu/Adaspec/main/assets/ablation.png" width="47%" alt="Ablation study">
  <img src="https://raw.githubusercontent.com/Khalil-Liu/Adaspec/main/assets/router_threshold.png" width="47%" alt="Router threshold analysis">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Khalil-Liu/Adaspec/main/assets/operator_visualization.png" width="72%" alt="SVD and GEVD operator visualization">
</p>

### Table 3: Post-Editing Performance on Control Tasks

![Table 3: Control-task results](https://raw.githubusercontent.com/Khalil-Liu/Adaspec/main/assets/table3.png)

### Table 4: Qualitative Editor Complementarity

![Table 4: Qualitative editor cases](https://raw.githubusercontent.com/Khalil-Liu/Adaspec/main/assets/table4.png)

### Table 5: Computational and Storage Overhead

![Table 5: Resource overhead](https://raw.githubusercontent.com/Khalil-Liu/Adaspec/main/assets/table5.png)

The resource measurements use one NVIDIA GeForce RTX 5090 on TruthfulQA. State and storage exclude the shared frozen Llama-2-7B backbone.

## Installation

The released experiments use Python 3.10 and Transformers 4.38.2. Linux with an NVIDIA GPU is recommended.

```bash
conda create -n adaspec-py310 python=3.10 -y
conda activate adaspec-py310

# Choose a PyTorch CUDA wheel that matches your driver.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The default backbone is the gated Hugging Face model `meta-llama/Llama-2-7b-chat-hf`. Request access and either run `huggingface-cli login` or pass a local model path through `--model-name` or `MODEL`.

## Required Artifacts

The evaluation release contains the probe checkpoint, but projector matrices are not stored in Git. Place them as follows, or override the shell variables used below.

```text
projectors/
  svd/no_mean_sub_uu_positive.pt
  svd/no_mean_sub_uu_negative.pt
  gevd/no_mean_sub_uu_positive.pt
  gevd/no_mean_sub_uu_negative.pt
weights/probe_router/probe_router.pt
```

```bash
export MODEL=/path/to/Llama-2-7b-chat-hf
export SVD_POS=projectors/svd/no_mean_sub_uu_positive.pt
export SVD_NEG=projectors/svd/no_mean_sub_uu_negative.pt
export GEVD_POS=projectors/gevd/no_mean_sub_uu_positive.pt
export GEVD_NEG=projectors/gevd/no_mean_sub_uu_negative.pt
export PROBE=weights/probe_router/probe_router.pt
```

## Training Code

This repository is an evaluation and inference release. HaluEval activation extraction, SVD/GEVD projector training, probe-feature extraction, and probe-training code are intentionally not included in the current repository. Full training scripts, configurations, and extended reproduction resources are coming soon after the review process.

## Evaluate TruthfulQA

```bash
python truthqa_eval.py \
  --model-name "$MODEL" --data-path data/truthfulqa \
  --output-path results/truthfulqa/adaspec_l2.json --is-chat \
  --svd-positive-proj "$SVD_POS" --svd-negative-proj "$SVD_NEG" \
  --gevd-positive-proj "$GEVD_POS" --gevd-negative-proj "$GEVD_NEG" \
  --probe-path "$PROBE" --lower-threshold 0.40 --upper-threshold 0.60 \
  --apply-sea-layers last-L --L 2 --combine-sea-embeddings l2_norm
```

## Evaluate BBQ

```bash
python bbq_eval.py \
  --model-name "$MODEL" --data-path data/BBQ \
  --output-path results/bbq/adaspec_l2.json --is-chat \
  --svd-positive-proj "$SVD_POS" --svd-negative-proj "$SVD_NEG" \
  --gevd-positive-proj "$GEVD_POS" --gevd-negative-proj "$GEVD_NEG" \
  --probe-path "$PROBE" --lower-threshold 0.40 --upper-threshold 0.60 \
  --apply-sea-layers last-L --L 2 --combine-sea-embeddings l2_norm
```

## Evaluate General Capabilities

The released control-task results cover four benchmarks: HellaSwag, MMLU, ToxiGen, and MathQA. `run_control_tasks.sh` runs methods sequentially on one GPU and writes JSON results plus `summary.md` under `results/control_tasks/`.

### HellaSwag

```bash
METHODS="base svd_l2 gevd_l2 adaspec" \
TASKS="hellaswag" \
BATCH_SIZE="auto:4" \
bash run_control_tasks.sh
```

### MMLU

```bash
METHODS="base svd_l2 gevd_l2 adaspec" \
TASKS="mmlu" \
BATCH_SIZE="auto:4" \
bash run_control_tasks.sh
```

### ToxiGen

```bash
METHODS="base svd_l2 gevd_l2 adaspec" \
TASKS="toxigen" \
BATCH_SIZE="auto:4" \
bash run_control_tasks.sh
```

### MathQA

MathQA requires the official `MathQA.zip` to be unpacked under `data/control_tasks/mathqa/` and is then evaluated with the included local task definition:

```bash
mkdir -p data/control_tasks/mathqa
curl -fL --retry 5 --retry-delay 5 \
  -o data/control_tasks/mathqa/MathQA.zip \
  https://math-qa.github.io/math-QA/data/MathQA.zip
unzip -oq data/control_tasks/mathqa/MathQA.zip -d data/control_tasks/mathqa

INCLUDE_PATH=local_tasks \
METHODS="base svd_l2 gevd_l2 adaspec" \
TASKS="mathqa_local" \
BATCH_SIZE="auto:4" \
bash run_control_tasks.sh
```

## Repository Contents

### Evaluation Entry Points

- `truthqa_eval.py`: evaluates Llama-2-style chat models on TruthfulQA multiple-choice questions and reports MC1, MC2, MC3, and mean model-forward time. It supports the base model, a fixed SEA editor, or AdaSpec routing.
- `bbq_eval.py`: evaluates the same editing modes on BBQ and reports accuracy, unknown-answer rate, and stereotype rate for the requested split.
- `evaluate_edited_lm_eval.py`: runs a base, fixed-editor, or AdaSpec-edited model through `lm-evaluation-harness`. It is the shared evaluator for HellaSwag, MMLU, ToxiGen, GSM8K, and other harness tasks.
- `run_control_tasks.sh`: sequentially runs the requested methods on one GPU, writes one JSON record per method, and invokes the result summarizer. Set `METHODS`, `TASKS`, `BATCH_SIZE`, `NUM_FEWSHOT`, or `LIMIT` as environment variables to control a run.
- `summarize_control_task_results.py`: reads the per-method harness JSON files and prints the HellaSwag and MMLU accuracy table used in the README.
- `local_tasks/mathqa_local/`: local `lm-evaluation-harness` task definition for the official MathQA archive; it evaluates multiple-choice accuracy after `MathQA.zip` is unpacked under `data/control_tasks/mathqa/`.

### Inference Components

- `src/decoding_algorithm/inference.py`: fixed spectral editing with the SVD or GEVD projection pair.
- `src/decoding_algorithm/inference_probe_router_abstain.py`: AdaSpec inference wrapper. It obtains the probe score, routes to SVD or GEVD when confident, and bypasses editing in the uncertainty interval.

### Figures and Result Artifacts

- `assets/method.jpg`: overview of the two-editor construction and confidence-aware routing procedure.
- `assets/ablation.png`: final radar visualization of the ablation results reported in the paper.
- `assets/router_threshold.png`: routing-threshold analysis, illustrating the effect of the confidence interval used for SVD/GEVD/bypass decisions.
- `assets/operator_visualization.png`: induced positive and negative spectral operators for SVD and GEVD, visualizing their distinct structures.
- `results/truthfulqa/` and `results/bbq/`: final benchmark JSON files, compact summaries, and exact command arguments.
- `results/control_tasks/`: per-method JSON outputs and compact general-capability metric summaries.
- `results/resource_overhead/`: raw latency, peak-memory, and editor-storage measurements together with the final cost summary.

The released `assets/` directory contains the final manuscript figure files. The corresponding plotting scripts are not part of the current evaluation and inference release.

## Citation

```bibtex
@article{adaspec2026,
  title={AdaSpec: Adaptive Spectral Editing for Large Language Model Alignment},
  author={Anonymous Authors},
  year={2026}
}
```

## Acknowledgments

AdaSpec extends the [SEA-LLM](https://github.com/liuhs666/sea-llm) implementation of *SEA: Spectral Editing of Activations for Large Language Model Alignment*. In particular, the model-loading, activation-editing, and baseline evaluation utilities under `src/` were used as the engineering foundation and were extended here with the GEVD editor and confidence-aware probe routing.

The general-capability evaluation interface is built on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [Hugging Face Transformers](https://github.com/huggingface/transformers). We also thank the authors of [TruthfulQA](https://github.com/sylinrl/TruthfulQA) and [BBQ](https://github.com/nyu-mll/BBQ) for releasing their benchmarks and evaluation resources.
