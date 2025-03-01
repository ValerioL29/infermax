from math import ceil

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class SarathiReplicaScheduler(BaseReplicaScheduler):
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

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            #XXX for eviction-free
            if self._config.no_evict:
                num_required_blocks = ceil(
                    (request.total_tokens - 1) / self._config.block_size
                )
            else:
                num_required_blocks = ceil(
                    request.num_prefill_tokens / self._config.block_size
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
            return True
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
            return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            #XXX for eviction-free
            if self._config.no_evict:
                num_required_blocks = ceil(
                    (request.total_tokens - 1) / self._config.block_size
                )
            else:
                num_required_blocks = ceil(
                    request.num_prefill_tokens / self._config.block_size
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
            else:
                self._preempted_requests.append(request)

        #print('MEMORY', self._num_allocated_blocks)

    def _get_request_next_num_tokens(
        self, request: Request, batch_contains_prefill: bool, num_batch_tokens: int
    ) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1

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
        skipped_requests = []
        running_prefills = []
        contains_prefill = False
        num_batch_tokens = 0

        #print('R_w', [r.id for r in self._request_queue])
        #print('R_r', [r.id for r in self._preempted_requests])

        # preempted requests could contain multiple requests which have
        # partial prefills completed, so we need to be careful
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                #print('B_LIMIT', self._max_micro_batch_size)
                break

            request = self._preempted_requests.pop(0)

            if not request.is_prefill_complete:
                running_prefills.append(request)
                continue

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                #print('C_LIMIT', request._id)
                skipped_requests.append(request)
                continue

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue = [victim_request] + self._request_queue
                    #print('M_LIMIT VICTIM', victim_request._id, 'FOR', request.id)
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue = [request] + self._request_queue
                    #print('M_LIMIT SELF', request._id)
                    break
            else:
                self._allocate_request(request)
                assert request.is_prefill_complete
                num_batch_tokens += next_num_tokens
                requests.append(request)
                num_tokens.append(next_num_tokens)

        #XXX no hybrid batch
        if self._config.no_hybrid_batch and len(requests) > 0:
            # re-add the skipped requests, but make sure that we add them to the
            # front of the queue so that they are scheduled first and we maintain FIFO ordering
            self._preempted_requests = running_prefills + self._preempted_requests
            self._preempted_requests = sorted(
                self._preempted_requests, key=lambda req: req.arrived_at
            )
            return Batch(self._replica_id, requests, num_tokens)

        #XXX here, don't check _can_allocate_request as they have already been checked
        for request in running_prefills:
            assert not request.is_prefill_complete


            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                skipped_requests.append(request)
                #print('C_LIMIT', request._id)
                continue

            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        # re-add the skipped requests, but make sure that we add them to the
        # front of the queue so that they are scheduled first and we maintain FIFO ordering
        self._preempted_requests = skipped_requests + self._preempted_requests
        self._preempted_requests = sorted(
            self._preempted_requests, key=lambda req: req.arrived_at
        )
        skipped_requests = []

        while self._request_queue:
            if len(self._allocation_map) == self._config.batch_size_cap:
                #print('R_LIMIT', self._config.batch_size_cap)
                break

            if len(requests) == self._max_micro_batch_size:
                #print('B_LIMIT', self._max_micro_batch_size)
                break

            if not self._can_allocate_request(self._request_queue[0]):
                #print('M_LIMIT', self._request_queue[0].id)
                break

            next_num_tokens = self._get_request_next_num_tokens(
                self._request_queue[0], contains_prefill, num_batch_tokens
            )

            if next_num_tokens == 0:
                #print('C_LIMIT', self._request_queue[0].id, 'prefill', self._request_queue[0].num_prefill_tokens, 'processed', self._request_queue[0].num_processed_tokens, 'chunk_size', self._config.chunk_size, 'batch_tokens', num_batch_tokens)
                break

            request = self._request_queue.pop(0)

            self._allocate_request(request)

            # all new requests will have a prefill
            contains_prefill = True
            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)
            #print('ADD', request._id, '# tokens', next_num_tokens)

        if not requests:
            return

        batch = Batch(self._replica_id, requests, num_tokens)
        #print('BATCH', batch._id, [r.id for r in requests])
        return batch
