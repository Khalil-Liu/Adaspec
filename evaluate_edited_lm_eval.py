"""Run lm-eval-harness tasks on a base, SEA, or AdaSpec edited model."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from decoding_algorithm.inference import Inference
from decoding_algorithm.inference_probe_router_abstain import InferenceProbeRouterAbstain


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("base", "sea", "adaspec"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated lm-eval task names.")
    parser.add_argument(
        "--include-path",
        action="append",
        default=[],
        help="Directory containing local lm-eval task YAML files; may be repeated.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", default="auto:4")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-fewshot", type=int)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=int, default=80)

    parser.add_argument("--sea-positive-proj")
    parser.add_argument("--sea-negative-proj")
    parser.add_argument("--svd-positive-proj")
    parser.add_argument("--svd-negative-proj")
    parser.add_argument("--gevd-positive-proj")
    parser.add_argument("--gevd-negative-proj")
    parser.add_argument("--probe-path")
    parser.add_argument("--lower-threshold", type=float, default=0.4)
    parser.add_argument("--upper-threshold", type=float, default=0.6)
    parser.add_argument("--apply-sea-layers", default="last-L")
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--combine-sea-embeddings", default="l2_norm")
    return parser.parse_args()


def require_args(args, names):
    missing = [f"--{name.replace('_', '-')}" for name in names if not getattr(args, name)]
    if missing:
        raise ValueError(f"{args.method} requires: {', '.join(missing)}")


def build_inference(args):
    common = dict(
        model_name=args.model_name,
        lora_name=None,
        dataset_name="lm_eval",
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        amateur_model_name=None,
        num_gpus=args.num_gpus,
    )
    if args.method == "base":
        return Inference(**common)
    if args.method == "sea":
        require_args(args, ("sea_positive_proj", "sea_negative_proj"))
        return Inference(
            **common,
            sea=True,
            positive_proj=args.sea_positive_proj,
            negative_proj=args.sea_negative_proj,
            apply_sea_layers=args.apply_sea_layers,
            L=args.L,
            combine_sea_embeddings=args.combine_sea_embeddings,
        )

    require_args(
        args,
        (
            "svd_positive_proj",
            "svd_negative_proj",
            "gevd_positive_proj",
            "gevd_negative_proj",
            "probe_path",
        ),
    )
    return InferenceProbeRouterAbstain(
        **common,
        sea_probe_router=True,
        svd_positive_proj=args.svd_positive_proj,
        svd_negative_proj=args.svd_negative_proj,
        gevd_positive_proj=args.gevd_positive_proj,
        gevd_negative_proj=args.gevd_negative_proj,
        probe_path=args.probe_path,
        lower_threshold=args.lower_threshold,
        upper_threshold=args.upper_threshold,
        apply_sea_layers=args.apply_sea_layers,
        L=args.L,
        combine_sea_embeddings=args.combine_sea_embeddings,
    )


def main():
    args = parse_args()
    try:
        import datasets
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
    except ImportError as error:
        raise RuntimeError(
            "lm-eval-harness is required. Install it with: pip install 'lm_eval[hf]==0.4.5'"
        ) from error

    # MathQA still uses its maintained Hugging Face dataset loading script.
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    print(f"Loading {args.run_name} for lm-eval tasks: {args.tasks}", flush=True)
    edited = build_inference(args)
    lm = HFLM(
        pretrained=edited.model,
        tokenizer=edited.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    evaluation_args = {
        "model": lm,
        "tasks": [task.strip() for task in args.tasks.split(",") if task.strip()],
        "task_manager": TaskManager(include_path=args.include_path),
        "batch_size": args.batch_size,
        "device": args.device,
        "limit": args.limit,
        "log_samples": True,
    }
    if args.num_fewshot is not None:
        evaluation_args["num_fewshot"] = args.num_fewshot
    results = simple_evaluate(**evaluation_args)
    results["edited_run"] = {
        "run_name": args.run_name,
        "method": args.method,
        "model_name": args.model_name,
        "tasks": args.tasks,
        "limit": args.limit,
    }
    if args.method == "adaspec":
        results["edited_run"]["router_stats"] = edited.summarize_router_stats()

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(json.dumps(results["results"], indent=2, default=str), flush=True)
    print(f"Saved lm-eval results to: {output}", flush=True)


if __name__ == "__main__":
    main()
