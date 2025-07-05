from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Literal

from vllm.core.scheduler import Scheduler
from vllm.sequence import SequenceStatus
from vllm.logger import init_logger

logger = init_logger(__name__)

@dataclass
class RequestStateTracker:
    """Tracks the state of requests across steps."""

    request_id: str
    last_computed_tokens: int
    last_output_tokens: int
    last_prompt_tokens: int
    phase: Literal["PREFILL", "DECODE"]
    step_added: int  # Step when first added to running
    preempted: bool = False


@dataclass
class BatchInfo:
    """Information about a batch processing step."""

    step_id: int
    timestamp: float
    duration: float
    stage: str  # "Step_1", "Step_2", etc.

    # Scheduler state
    num_running_requests: int
    num_waiting_requests: int
    num_finished_requests: int

    # Request details
    running_requests: List[Dict[str, Any]]
    waiting_requests: List[Dict[str, Any]]
    finished_request_ids: List[str]

    # Batch schedule information - updated format
    # (request_id, c, m, phase) where:
    # - c = # tokens to process (1 for decode, num_tokens for prefill)
    # - m = # KVs to read (num_computed_tokens for decode, 0 for prefill)
    # - phase = "PREFILL" or "DECODE"
    # - preempted = True if the request is preempted
    batch_schedule: List[Tuple[str, int, int, str, bool]]

    # Batch size for this step
    batch_size: int

    # Total number of KV cache tokens
    total_kv_cache_tokens: int

    # Total number of processed tokens
    total_processed_tokens: int

    # Preemption in this step
    preemption_in_step: int

    # Total preemption count
    acc_preemption_count: int


def extract_scheduler_state(scheduler: Scheduler) -> Dict[str, Any]:
    """Extract current state from vLLM engine scheduler."""

    # Extract running requests information
    running_requests = []
    for seq_group in scheduler.running:
        running_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        if len(running_seqs) == 0:
            continue
        # Get first sequence since we only have one sequence per request
        seq = running_seqs[0]
        num_computed_tokens = seq.data.get_num_computed_tokens()
        num_tokens = seq.get_len()
        remaining_tokens = num_tokens - num_computed_tokens
        total_tokens = num_tokens
        num_prefill_tokens = seq.get_num_prefill_tokens()
        num_decode_tokens = seq.get_num_decode_tokens()
        running_requests.append(
            {
                "request_id": seq_group.request_id,
                "num_computed_tokens": num_computed_tokens,
                "num_tokens": num_tokens,
                "remaining_tokens": remaining_tokens,
                "total_tokens": total_tokens,
                "status": str(seq.status),
                "arrival_time": getattr(seq_group, "arrival_time", None),
                "prompt_token_ids_len": len(seq.data.prompt_token_ids),
                "output_token_ids_len": len(seq.data.output_token_ids),
                "is_prefill": seq_group.is_prefill(),
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
            }
        )

    # Extract waiting requests information
    waiting_requests = []
    for seq_group in scheduler.waiting:
        # Get first sequence since we only have one sequence per request
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        waiting_requests.append(
            {
                "request_id": seq_group.request_id,
                "num_tokens": seq.get_len(),
                "status": str(seq.status),
                "arrival_time": getattr(seq_group, "arrival_time", None),
                "is_prefill": seq_group.is_prefill(),
            }
        )

    return {
        "running_requests": running_requests,
        "waiting_requests": waiting_requests,
        "num_running": len(scheduler.running),
        "num_waiting": len(scheduler.waiting),
        "finished_req_ids": scheduler._finished_requests_ids,
        "preemption_count": scheduler.preemption_count,
    }


def track_request_changes_and_create_batch_schedule(
    scheduler_state: Dict[str, Any],
    step_count: int,
    prev_preemption_count: int,
    request_tracker: Dict[str, RequestStateTracker],
    enable_chunked_prefill: bool = False,
) -> Dict[str, Any]:
    """Track request changes and create batch schedule based on state changes.

    Returns:
        List of (request_id, c, m, phase) where:
        - c = # tokens to process (1 for decode, num_tokens for prefill)
        - m = # KVs to read (num_computed_tokens for decode, 0 for prefill)
        - phase = "PREFILL" or "DECODE"
    """
    batch_schedule = []

    # Track current running requests
    current_running_ids = set()

    # Total number of tokens processed
    total_kv_cache_tokens = 0

    # Total number of output tokens
    total_processed_tokens = 0

    # Update preemption count
    preemption_in_step = scheduler_state["preemption_count"] - prev_preemption_count
    acc_preemption_count = scheduler_state["preemption_count"]

    # Process running requests
    for req_info in scheduler_state["running_requests"]:
        request_id = req_info["request_id"]
        current_running_ids.add(request_id)

        num_computed_tokens = req_info["num_computed_tokens"]
        num_tokens = req_info["num_tokens"]
        prompt_token_ids_len = req_info["prompt_token_ids_len"]
        output_token_ids_len = req_info["output_token_ids_len"]
        is_prefill = req_info["is_prefill"]
        num_prefill_tokens = req_info["num_prefill_tokens"]

        if request_id not in request_tracker:
            # This is a PREFILL request (could be chunked or full)
            if enable_chunked_prefill:
                # For chunked prefill, we need to determine if this is a chunk
                # or the full prefill. Check if computed tokens equals prompt tokens
                if num_computed_tokens == prompt_token_ids_len:
                    # This is a full prefill (not chunked yet)
                    c = num_tokens  # prompt + output tokens
                else:
                    # This is a chunked prefill
                    c = num_computed_tokens  # tokens to process in this chunk
            else:
                # Non-chunked prefill
                c = num_tokens  # prompt + output tokens

            # There's always 0 kv cache tokens are read for initial step of prefill
            m = 0
            phase = "PREFILL"

            # Add to tracker
            request_tracker[request_id] = RequestStateTracker(
                request_id=request_id,
                last_computed_tokens=num_computed_tokens,
                last_output_tokens=output_token_ids_len,
                last_prompt_tokens=prompt_token_ids_len,
                phase=phase,
                step_added=step_count,
            )

            batch_schedule.append((request_id, c, m, phase, False))
            total_kv_cache_tokens += m
            total_processed_tokens += c
        else:
            # Request already tracked - check for changes
            tracker = request_tracker[request_id]

            # Check if it's a chunked prefill step
            if num_computed_tokens <= num_prefill_tokens:
                assert enable_chunked_prefill or (
                    not enable_chunked_prefill
                    and num_computed_tokens == num_prefill_tokens
                ), (
                    "Such circumstances should not happen. Only chunked prefill "
                    "should be enabled when is_prefill is True."
                )
                c = num_computed_tokens - tracker.last_computed_tokens
                m = tracker.last_computed_tokens
                phase = "PREFILL"
                if c > 0:
                    batch_schedule.append((request_id, c, m, phase, False))
                    total_kv_cache_tokens += m
                    total_processed_tokens += c
                tracker.last_computed_tokens = num_computed_tokens
                tracker.last_output_tokens = output_token_ids_len
                tracker.phase = phase
                request_tracker[request_id] = tracker
            else:
                # This is a decode step
                assert not is_prefill, "is_prefill should be False for decode step"
                c = 1
                m = num_computed_tokens
                phase = "DECODE"
                batch_schedule.append((request_id, c, m, phase, False))
                total_kv_cache_tokens += m
                total_processed_tokens += c
                tracker.last_computed_tokens = num_computed_tokens
                tracker.last_output_tokens = output_token_ids_len
                tracker.phase = phase

    # Waiting request to check if they are preempted
    local_preemption_count = 0
    for req_info in scheduler_state["waiting_requests"]:
        request_id = req_info["request_id"]
        if request_id in request_tracker:
            tracker = request_tracker[request_id]
            tracker.preempted = True
            c = 0
            m = tracker.last_computed_tokens
            phase = tracker.phase
            batch_schedule.append((request_id, c, m, phase, tracker.preempted))
            # Remove from tracker since it's preempted
            del request_tracker[request_id]
            local_preemption_count += 1
    logger.info(f"Local preemption count: {local_preemption_count} for step {step_count}")

    # Process finished requests
    for request_id in scheduler_state["finished_req_ids"]:
        if request_id in request_tracker:
            # This request finished - it's a DECODE request
            tracker = request_tracker[request_id]
            c = 1  # process 1 token
            m = (
                tracker.last_computed_tokens + 1
            )  # read all computed tokens + 1 for final token
            phase = "DECODE"

            batch_schedule.append((request_id, c, m, phase, False))
            total_kv_cache_tokens += m
            total_processed_tokens += c
            # Remove from tracker since it's finished
            del request_tracker[request_id]

    # Clean up tracker for requests no longer in running
    finished_ids = set(request_tracker.keys()) - current_running_ids
    for request_id in finished_ids:
        del request_tracker[request_id]

    return dict(
        batch_schedule=batch_schedule,
        total_kv_cache_tokens=total_kv_cache_tokens,
        total_processed_tokens=total_processed_tokens,
        preemption_in_step=preemption_in_step,
        acc_preemption_count=acc_preemption_count,
    )

def print_batch_info(batch_info: BatchInfo):
    """Print batch information in a formatted way."""
    logger.info(f"  Step {batch_info.step_id} ({batch_info.stage}):")
    logger.info(
        f"    Running: {batch_info.num_running_requests}, Waiting: {batch_info.num_waiting_requests}"
    )
    logger.info(f"    Finished: {batch_info.num_finished_requests}")
    logger.info(f"    Batch schedule: {batch_info.batch_schedule}")
    logger.info(f"    Batch size: {batch_info.batch_size}")
