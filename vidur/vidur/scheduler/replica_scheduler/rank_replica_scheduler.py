from math import ceil, log2

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from typing import List


class RankReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # sarathi config
        self._num_running_batches = 0
        self._preempted_requests = []
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

        self.num_completed_requests = 0
        self.num_processed_tokens = 0

    def _calculate_num_required_blocks(self, requests: List[Request]) -> int:
        if self._config.no_evict:
            return ceil(sum(ceil(request.total_tokens / self._config.block_size) for request in requests))
        else:
            return ceil(sum(ceil(request.num_prefill_tokens / self._config.block_size) for request in requests))

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            #XXX for eviction-free
            if self._config.no_evict:
                num_required_blocks = ceil(
                    request.total_tokens / self._config.block_size
                )
            else:
                num_required_tokens = request.num_prefill_tokens

                num_required_blocks = ceil(
                    num_required_tokens / self._config.block_size
                )
            #print('can_allocate', request._id, 'prefill', request.num_prefill_tokens, 'total', self._config.num_blocks, 'allocated', self._num_allocated_blocks, 'required', num_required_blocks, 'watermark', self._watermark_blocks)
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        #XXX for eviction-free
        if self._config.no_evict:
            #return True
            return self._config.num_blocks - self._num_allocated_blocks >= 1
            #return True
            #return self._config.num_blocks - self._num_allocated_blocks >= 1
            num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
            num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)
            if num_tokens_required == 0:
                return True
            else:
                return self._config.num_blocks - self._num_allocated_blocks >= 1
        else:
            # vllm requires at least one block to be available
            return (self._config.num_blocks - self._num_allocated_blocks) >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            #XXX for eviction-free
            if self._config.no_evict:
                num_required_blocks = ceil(
                    request.total_tokens / self._config.block_size
                )
            else:
                num_required_tokens = request.num_prefill_tokens
                num_required_blocks = ceil(
                    num_required_tokens / self._config.block_size
                )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)

        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return

        assert not self._config.no_evict, "Should not reach here for eviction-free"

        self.allocate(request.id, 1)

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
                self.num_completed_requests += 1
                self.num_processed_tokens += request.total_tokens
            else:
                self._preempted_requests.append(request)

        #print('MEMORY', self._num_allocated_blocks)

    def _get_request_next_num_tokens(
        self, request: Request, num_batch_tokens: int
    ) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return min(
                1,
                self._config.chunk_size - num_batch_tokens,
            )

        #XXX no chunked prefill
        if self._config.no_chunked_prefill:
            assert request.num_processed_tokens == 0
            if self._config.chunk_size < num_batch_tokens + request.num_prefill_tokens:
                return 0

        next_num_tokens = min(
            request.num_prefill_tokens - request.num_processed_tokens,
            self._config.chunk_size - num_batch_tokens,
        )

        next_num_tokens = max(0, next_num_tokens)

        return next_num_tokens

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        num_batch_tokens = 0

        #print('R_w', [r.id for r in self._request_queue])
        #print('R_r', [r.id for r in self._preempted_requests])

        # preempted requests could contain multiple requests which have
        # partial prefills completed, so we need to be careful
        if self._config.sorted_by == "I":
            all_requests = sorted(self._request_queue + self._preempted_requests, key=lambda req: (req._initial_num_prefill_tokens, req._id))
        elif self._config.sorted_by == "O":
            all_requests = sorted(self._request_queue + self._preempted_requests, key=lambda req: (req._initial_num_decode_tokens, req._id))
        elif self._config.sorted_by == "id":
            all_requests = sorted(self._request_queue + self._preempted_requests, key=lambda req: req._id)
        assert len(all_requests) == len(self._request_queue) + len(self._preempted_requests)

        preempted = []

        #for i in range(len(all_requests)):
        for request in all_requests:
            assert len(all_requests) == len(self._request_queue) + len(self._preempted_requests) + len(requests), f'{len(all_requests)} == {len(self._request_queue)} + {len(self._preempted_requests)} + {len(requests)}'

            if len(requests) == self._max_micro_batch_size:
                break
            next_num_tokens = self._get_request_next_num_tokens(
                request, num_batch_tokens
            )
            if next_num_tokens == 0:
                continue

            if request in self._request_queue:
                self._request_queue.remove(request)
            elif request in self._preempted_requests:
                self._preempted_requests.remove(request)

            #next_num_tokens = self._get_request_next_num_tokens(request)
            #if num_batch_tokens + next_num_tokens > self._config.max_tokens_in_batch:
            #    break

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue = [victim_request] + self._request_queue
                    #preempted = [victim_request] + preempted
                else:
                    # add request to R_w
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    #preempted = [request] + preempted
                    break
            else:
                # add request to batch
                self._allocate_request(request)
                #next_num_tokens = self._get_request_next_num_tokens(request)
                num_batch_tokens += next_num_tokens
                requests.append(request)
                num_tokens.append(next_num_tokens)
                assert num_batch_tokens <= self._config.chunk_size, f'{num_batch_tokens} <= {self._config.chunk_size}, new = {next_num_tokens}'

        #self._request_queue = preempted + self._request_queue
        assert len(all_requests) == len(self._request_queue) + len(self._preempted_requests) + len(requests), f'{len(all_requests)} == {len(self._request_queue)} + {len(self._preempted_requests)} + {len(requests)}'

        #assert(self._config.chunk_size <= 16384, f"chunk_size: {self._config.chunk_size}")
        #print(f"num_batch_tokens: {num_batch_tokens} chunk_size: {self._config.chunk_size}")
        #assert(num_batch_tokens <= self._config.chunk_size, f"num_batch_tokens: {num_batch_tokens} chunk_size: {self._config.chunk_size}")
        #assert(sum(num_tokens) <= self._config.chunk_size, f"sum(num_tokens): {sum(num_tokens)} chunk_size: {self._config.chunk_size}")

        # print('requests')
        # for request in requests:
        #     print(request._id, request._initial_num_prefill_tokens, request._initial_num_decode_tokens)

        if not requests:
            return

        assert num_batch_tokens <= self._config.chunk_size

        batch = Batch(self._replica_id, requests, num_tokens)
        #print('BATCH', batch._id, [r.id for r in requests])
        return batch
