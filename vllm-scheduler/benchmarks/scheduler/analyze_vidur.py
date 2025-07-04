import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from rich import print
from rich.progress import track

def extract_info_from_filename(filename: str) -> dict[str, str]:
    path = Path(filename)
    stem = path.stem
    model_prefix, other_info_str = stem.split("b_")
    model_name = f"{model_prefix}b"
    max_num_seqs, max_num_batched_tokens, preemption_policy, scheduler_type = other_info_str.split("_")
    return dict(
        model=model_name,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        preemption_policy=preemption_policy,
        scheduler_type=scheduler_type,
    )

def get_bulk_metrics(metric_list: list[float | int]) -> dict[str, float | int]:
    return {
        "min": np.min(metric_list).item(),
        "median": np.median(metric_list).item(),
        "mean": np.mean(metric_list).item(),
        "max": np.max(metric_list).item(),
        "sum": np.sum(metric_list).item(),
        "std": np.std(metric_list).item(),
    }

def process_metrics_file(metrics_file: str):
    with open(metrics_file, "rb") as f:
        item = pickle.load(f)

    batch_start_time = item["batch_start_time"]
    total_time = batch_start_time[-1] - batch_start_time[0]

    model_forward_time = item["model_forward_time"][1:]
    model_forward_metrics = get_bulk_metrics(model_forward_time)
    model_execution_time = item["model_execution_time"][1:]
    model_execution_metrics = get_bulk_metrics(model_execution_time)
    gpu_cache_usage = item["gpu_cache_usage"][1:]
    gpu_cache_usage_metrics = get_bulk_metrics(gpu_cache_usage)


    results = {
        "info": extract_info_from_filename(metrics_file),
        "total_time": total_time,
        "batch_start_time": batch_start_time,
        "cache_block_size": item["cache_block_size"],
        "total_num_tokens": item["total_num_tokens"],
        "requests_per_second": item["requests_per_second"],
        "tokens_per_second": item["tokens_per_second"],
        "output_tokens_per_second": item["output_tokens_per_second"],
        "model_forward_time": model_forward_metrics,
        "model_execution_time": model_execution_metrics,
        "gpu_cache_usage": gpu_cache_usage_metrics,
    }
    print(f"Processed metrics: {results}")

    # Save results to a file
    output_file = metrics_file.replace(".pkl", ".json")
    with open(output_file, "w+", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_file}")


def main(args):
    metrics_files = Path(args.metrics_folder).glob("*.pkl")
    for metrics_file in track(metrics_files, description="Processing metrics files"):
        process_metrics_file(str(metrics_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics-folder",
        type=str,
        required=True,
        help="Path to the metrics file",
    )

    args = parser.parse_args()

    main(args)