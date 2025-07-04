"""Benchmark offline inference throughput."""
import argparse
import pickle
import time
from dataclasses import asdict, dataclass
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm


from vllm import LLM, SamplingParams
from vllm.core.scheduler import PrecomputedSchedule, Scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

logger = init_logger(__name__)


@dataclass
class TraceRequest:
    """Represents a request from the trace file."""

    request_id: int
    arrival_time: float
    num_prefill_tokens: int
    num_decode_tokens: int

def parse_trace_file(csv_path: str) -> List[TraceRequest]:
    """Parse the trace CSV file into TraceRequest objects."""
    df = pd.read_csv(csv_path)
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
        f"Time span: {trace_requests[0].arrival_time:.2f}s to {trace_requests[-1].arrival_time:.2f}s"
    )

    return trace_requests

def run_vllm(
    requests: List[TraceRequest],
    engine_args: EngineArgs,
    use_srf_preemption: bool = False,
) -> float:
    # Initialize the LLM engine
    llm = LLM(**asdict(engine_args))

    # Get the tokenizer
    tokenizer = llm.get_tokenizer()

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
    pbar = tqdm(total=len(requests), desc="Processing requests")

    # Start timer
    start = time.perf_counter()

    # Profile each step with real-time tracking and incremental saving
    while llm.llm_engine.has_unfinished_requests():
        # Step the engine
        outputs: list[RequestOutput] = llm.llm_engine.step()

        # Get num of running requests
        num_running_requests = len(scheduler.running)
        num_waiting_requests = len(scheduler.waiting)

        # Count the number of finished requests
        num_finished_requests = len([output for output in outputs if output.finished])

        # Update processed requests count
        processed_requests += num_finished_requests

        # Update progress bar
        pbar.update(num_finished_requests)
        logger.info(
            f"Batch status: {dict(
                num_running_requests=num_running_requests,
                num_waiting_requests=num_waiting_requests,
                processed_requests=processed_requests,
                preemption_count=scheduler.preemption_count,
            )}"
        )
        pbar.refresh()
    
    # Stop timer
    end = time.perf_counter()

    # Update the schedule
    precomputed_schedule = PrecomputedSchedule()
    precomputed_schedule.batch_start_time.append(time.time())

    pbar.close()

    return end - start


def main(args: argparse.Namespace):
    # Set seed
    torch.manual_seed(args.seed)

    # Initialize the precomputed schedule
    precomputed_schedule = PrecomputedSchedule()

    # Metrics containers
    precomputed_schedule.batch_start_time = []
    precomputed_schedule.gpu_cache_usage = []
    precomputed_schedule.model_execution_time = []
    precomputed_schedule.model_forward_time = []
    precomputed_schedule.all_reduce_time = [0]
    precomputed_schedule.attn_time = []
    precomputed_schedule.attn_profile_log = []

    # Parse the trace file
    requests = parse_trace_file(args.trace_file)

    # Run the VLLM engine
    elapsed_time = run_vllm(
        requests,
        EngineArgs.from_cli_args(args),
        use_srf_preemption=args.use_srf_preemption,
    )

    total_num_tokens = sum(
        request.num_prefill_tokens + request.num_decode_tokens
        for request in requests
    )
    total_output_tokens = sum(
        request.num_decode_tokens
        for request in requests
    )

    try:
        precomputed_schedule.gpu_cache_usage.append(precomputed_schedule.num_total_gpu_cache)
    except:
        logger.error("Failed to get the number of total GPU cache")

    results = {
        "elapsed_time": elapsed_time,
        "num_requests": len(requests),
        "total_num_tokens": total_num_tokens,
        "requests_per_second": len(requests) / elapsed_time,
        "tokens_per_second": total_num_tokens / elapsed_time,
        'output_tokens_per_second': total_output_tokens / elapsed_time,
        "batch_start_time": precomputed_schedule.batch_start_time,
        "gpu_cache_usage": precomputed_schedule.gpu_cache_usage,
        'model_execution_time': precomputed_schedule.model_execution_time,
        'model_forward_time': precomputed_schedule.model_forward_time,
        'available_kv_cache_memory': precomputed_schedule.available_kv_cache_memory,
        'cache_block_size': precomputed_schedule.cache_block_size,
        'num_total_gpu_cache': precomputed_schedule.num_total_gpu_cache,
        'attn_time': precomputed_schedule.attn_time,
        'attn_profile_log': precomputed_schedule.attn_profile_log,
    }

    with open(args.output_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")

    parser.add_argument(
        "--trace-file",
        type=str,
        default="trace.csv",
        help='Path to the trace file.',
    )
    parser.add_argument(
        "--use-srf-preemption",
        action="store_true",
        default=False,
        help='Use SRF preemption.',
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default="results.pkl",
        help='Path to save the throughput results in pickle format.',
    )

    parser = EngineArgs.add_cli_args(parser)

    args = parser.parse_args()

    main(args)
