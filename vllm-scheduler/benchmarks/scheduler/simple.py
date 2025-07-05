"""Benchmark offline inference throughput."""

import argparse
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from vllm import ModelRegistry, LLM, SamplingParams
from vllm.core.scheduler import PrecomputedSchedule, Scheduler, StepTracker
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

from custom_llama import DiasLlamaForCausalLM
from engine_step_tracker import (
    track_request_changes_and_create_batch_schedule,
    extract_scheduler_state,
    print_batch_info,
    RequestStateTracker,
    BatchInfo,
)

logging.basicConfig(level=logging.INFO)
logger = init_logger(__name__)


@dataclass
class TraceRequest:
    """Represents a request from the trace file."""

    request_id: int
    arrival_time: float
    num_prefill_tokens: int
    num_decode_tokens: int


def parse_trace_file(csv_path: str, top_n: int = -1) -> List[TraceRequest]:
    """Parse the trace CSV file into TraceRequest objects."""
    df = pd.read_csv(csv_path)
    if top_n > 0:
        df = df.head(top_n)
    trace_requests = sorted(
        (
            TraceRequest(
                request_id=i,
                arrival_time=float(row["arrival_time"]),
                num_prefill_tokens=int(row["num_prefill_tokens"]),
                num_decode_tokens=int(row["num_decode_tokens"]),
            )
            for i, row in enumerate(df.to_dict(orient="records"))
        ),
        key=lambda x: x.arrival_time,
    )

    logger.info(f"Loaded {len(trace_requests)} trace requests")
    logger.info(
        f"Time span: {trace_requests[0].arrival_time:.2f}s to"
        f" {trace_requests[-1].arrival_time:.2f}s"
    )

    return trace_requests


def run_vllm(
    requests: List[TraceRequest],
    engine_args: EngineArgs,
    use_srf_preemption: bool = False,
) -> tuple[float, float]:
    # Register the custom model
    ModelRegistry.register_model("LlamaForCausalLM", DiasLlamaForCausalLM)

    # Initialize the LLM engine
    llm = LLM(**asdict(engine_args))

    # Get the tokenizer
    tokenizer = llm.get_tokenizer()

    # Initialize the precomputed schedule
    precomputed_schedule = PrecomputedSchedule()

    # Scheduler
    scheduler: Scheduler = llm.llm_engine.scheduler[0]
    if use_srf_preemption:
        scheduler.set_use_srf_preemption(use_srf_preemption)

    # Force disable infermax schedule
    scheduler.set_use_infermax_schedule(False)

    # Add the requests to the engine
    for i, request in enumerate(requests):
        sampling_param = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=request.num_decode_tokens,
        )
        prompt_token_ids = torch.randint(
            tokenizer.vocab_size, size=(request.num_prefill_tokens,)
        ).tolist()
        llm.llm_engine.add_request(
            request_id=f"seq{i}",
            prompt={"prompt_token_ids": prompt_token_ids},
            params=sampling_param,
            arrival_time=request.arrival_time,
        )

    # Create progress bar
    processed_requests = 0

    # Initialize request tracker for state changes
    request_tracker: Dict[str, RequestStateTracker] = {}

    # Step tracker
    step_tracker = StepTracker()
    step_tracker.set_step(0)

    # Progress bar
    pbar = tqdm(total=len(requests), desc="Processing requests")

    # Start timer
    start = time.perf_counter()

    # Profile each step with real-time tracking and incremental saving
    while llm.llm_engine.has_unfinished_requests():
        step_count = step_tracker.get_current_step()
        logger.info(f"\n--- Step {step_count} ---")

        # Get number of running sequences
        num_running_seqs = len(scheduler.running)
        logger.info(f"Number of running sequences: {num_running_seqs}")

        # Preemption count
        prev_preemption_count = scheduler.preemption_count

        # Engine step for processing requests
        step_start_time = time.perf_counter()
        outputs: list[RequestOutput] = llm.llm_engine.step()
        step_end_time = time.perf_counter()

        num_finished_requests = len([output for output in outputs if output.finished])

        # Determine stage (Step_1, Step_2, etc.)
        stage = f"Step_{step_count}"

        # Extract scheduler state after the step
        post_step_state = extract_scheduler_state(scheduler)

        # Create batch schedule from scheduler state using new tracking logic
        track_result = track_request_changes_and_create_batch_schedule(
            post_step_state,
            step_count,
            prev_preemption_count,
            request_tracker,
            engine_args.enable_chunked_prefill,
        )
        batch_schedule = track_result["batch_schedule"]
        total_kv_cache_tokens = track_result["total_kv_cache_tokens"]
        total_processed_tokens = track_result["total_processed_tokens"]
        preemption_in_step = track_result["preemption_in_step"]
        acc_preemption_count = track_result["acc_preemption_count"]

        # Create batch info
        batch_info = BatchInfo(
            step_id=step_count,
            stage=stage,
            batch_size=post_step_state["num_running"],
            timestamp=precomputed_schedule.batch_start_time[-1],
            duration=step_end_time - step_start_time,
            num_running_requests=post_step_state["num_running"],
            num_waiting_requests=post_step_state["num_waiting"],
            num_finished_requests=num_finished_requests,
            finished_request_ids=post_step_state["finished_req_ids"],
            batch_schedule=batch_schedule,
            total_kv_cache_tokens=total_kv_cache_tokens,
            total_processed_tokens=total_processed_tokens,
            preemption_in_step=preemption_in_step,
            acc_preemption_count=acc_preemption_count,
        )

        # Update the precomputed schedule
        precomputed_schedule.batch_info.append(asdict(batch_info))

        # Print batch information
        logger.info(f"Stage {batch_info.stage} finished")
        print_batch_info(batch_info)
        logger.info(f"Step {step_count} saved incrementally")

        # Update processed requests count
        step_tracker.update_step()
        processed_requests += num_finished_requests
        pbar.update(num_finished_requests)

        # Check if all requests are finished
        if not llm.llm_engine.has_unfinished_requests():
            logger.info("\nAll requests completed!")
            break

    # Stop timer
    end = time.perf_counter()

    # Update the schedule
    precomputed_schedule.batch_start_time.append(end)

    pbar.close()

    return end - start, scheduler.preemption_count


def main(args: argparse.Namespace):
    # Set seed
    torch.manual_seed(args.seed)

    # Initialize the precomputed schedule
    precomputed_schedule = PrecomputedSchedule()

    # Metrics containers
    precomputed_schedule.all_reduce_time = [0]
    precomputed_schedule.batch_info = []
    precomputed_schedule.batch_start_time = []
    precomputed_schedule.gpu_cache_usage = {}
    precomputed_schedule.model_execution_time = {}
    precomputed_schedule.model_forward_time = {}
    precomputed_schedule.decoders_forward_time = {}

    # Parse the trace file
    requests = parse_trace_file(args.trace_file)

    # Run the VLLM engine
    elapsed_time, preemption_count = run_vllm(
        requests,
        EngineArgs.from_cli_args(args),
        use_srf_preemption=args.use_srf_preemption,
    )

    total_num_tokens = sum(
        request.num_prefill_tokens + request.num_decode_tokens for request in requests
    )
    total_output_tokens = sum(request.num_decode_tokens for request in requests)

    try:
        precomputed_schedule.gpu_cache_usage.append(
            precomputed_schedule.num_total_gpu_cache
        )
    except:
        logger.error("Failed to get the number of total GPU cache")

    results = {
        "elapsed_time": elapsed_time,
        "num_requests": len(requests),
        "total_preemption_count": preemption_count,
        "total_num_tokens": total_num_tokens,
        "requests_per_second": len(requests) / elapsed_time,
        "tokens_per_second": total_num_tokens / elapsed_time,
        "output_tokens_per_second": total_output_tokens / elapsed_time,
        "batch_start_time": precomputed_schedule.batch_start_time,
        "gpu_cache_usage": precomputed_schedule.gpu_cache_usage,
        "model_execution_time": precomputed_schedule.model_execution_time,
        "model_forward_time": precomputed_schedule.model_forward_time,
        "available_kv_cache_memory": precomputed_schedule.available_kv_cache_memory,
        "cache_block_size": precomputed_schedule.cache_block_size,
        "num_total_gpu_cache": precomputed_schedule.num_total_gpu_cache,
        "batch_info": precomputed_schedule.batch_info,
        "decoders_forward_time": precomputed_schedule.decoders_forward_time,
    }

    with open(args.output_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")

    parser.add_argument(
        "--trace-file",
        type=str,
        default="trace.csv",
        help="Path to the trace file.",
    )
    parser.add_argument(
        "--use-srf-preemption",
        action="store_true",
        default=False,
        help="Use SRF preemption.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results.pkl",
        help="Path to save the throughput results in pickle format.",
    )

    parser = EngineArgs.add_cli_args(parser)

    args = parser.parse_args()

    main(args)
