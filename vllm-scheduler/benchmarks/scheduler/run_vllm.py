"""Benchmark offline inference throughput."""
import argparse
import dataclasses
import json
import random
import time
from typing import List, Optional, Tuple

import torch
import uvloop
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args)
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser, merge_async_iterators
from vllm.core.scheduler import PrecomputedSchedule

from generate_precomputed_schedule import generate_precomputed_schedule, generate_precomputed_schedule_from_vidur, generate_synthetic_schedule

def run_vllm(
    requests: List[Tuple[str, int, int]],
    n: int,
    engine_args: EngineArgs,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(**dataclasses.asdict(engine_args))

    # Add the requests to the engine.
    prompts: List[str] = []
    sampling_params: List[SamplingParams] = []
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
            ))

    start = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=True)
    end = time.perf_counter()
    precomputed_schedule = PrecomputedSchedule()
    precomputed_schedule.batch_start_time.append(time.time())
    return end - start


def generate_requests(tokenizer, schedule):
    prev_input_len = 0
    requests = []
    for input_len, output_len in schedule['requests']:
        if prev_input_len != input_len:
            for i in range(-10, 10):
                prompt = "hi" * (input_len + i)
                tokenized_prompt = tokenizer(prompt).input_ids
                if len(tokenized_prompt) == input_len:
                    break
            assert(len(tokenized_prompt) == input_len)
        requests.append([prompt, input_len, output_len])
        prev_input_len = input_len
    return requests


def main(args: argparse.Namespace):

    random.seed(args.seed)

    precomputed_schedule = PrecomputedSchedule()
    if args.vidur:
        schedule = generate_precomputed_schedule_from_vidur(args.schedule)
    elif args.synthetic:
        if args.prefill_chunk_size == 0:
            args.prefill_chunk_size = args.I            
        schedule = generate_synthetic_schedule(args.I, args.O, args.B, args.prefill_chunk_size)
        args.enable_chunked_prefill = schedule['chunked_prefill']
        print('chunked_prefill', args.enable_chunked_prefill)
    else:
        schedule = generate_precomputed_schedule(args.schedule, int(args.k))
        args.enable_chunked_prefill = schedule['chunked_prefill']

        #print(schedule)
    precomputed_schedule.c = schedule['c']
    precomputed_schedule.p = schedule['p']
    precomputed_schedule.evicted = schedule['evicted']
    precomputed_schedule.batch_start_time = []
    precomputed_schedule.gpu_cache_usage = []
    precomputed_schedule.model_execution_time = []
    precomputed_schedule.model_forward_time = []
    precomputed_schedule.all_reduce_time = [0]
    precomputed_schedule.attn_time = []
    precomputed_schedule.attn_profile_log = []
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
        
    requests = generate_requests(tokenizer, schedule)

    elapsed_time = run_vllm(requests, args.n,
                            EngineArgs.from_cli_args(args))

    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    total_output_tokens = sum(output_len for _, _, output_len in requests)

    precomputed_schedule.gpu_cache_usage.append(precomputed_schedule.num_total_gpu_cache)

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

    if args.synthetic:
        with open(args.schedule + f'.vllm.json', "w") as f:
            json.dump(results, f, indent=4)
    else:
        if args.k == 1:
            with open(args.schedule + '.vllm.json', "w") as f:
                json.dump(results, f, indent=4)
        else:
            with open(args.schedule + f'.vllm.{args.k}.json', "w") as f:
                json.dump(results, f, indent=4)       

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")

    parser.add_argument("--k", type=int, default=1)

    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--schedule",
                        type=str,
                        default='./schedule.pkl',
                        help='Path to the schedule')
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument("--async-engine",
                        action='store_true',
                        default=False,
                        help="Use vLLM async engine rather than LLM class.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    parser.add_argument("--vidur", action='store_true', default=False)
    parser.add_argument("--synthetic", action='store_true', default=False)

    parser.add_argument("--prefill-chunk-size", type=int, default=0)
    parser.add_argument("--I", type=int, default=0)
    parser.add_argument("--O", type=int, default=0)
    parser.add_argument("--B", type=int, default=0)


    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)
