from math import ceil, log2

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from typing import List


class TestReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, is_decode_scheduler: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self._is_decode_scheduler = is_decode_scheduler
        self._decode_scheduler = None
        self.handoff_requests = []

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
        self.histogram = [[0,0] for i in range(100)]
        self.num_overloaded_tokens = 0
        self.overload_map = {}

    def set_decode_scheduler(self, scheduler: "TestReplicaScheduler") -> None:
        self._decode_scheduler = scheduler

    #XXX: not used
    def _calculate_num_required_blocks(self, requests: List[Request]) -> int:
        if self._config.no_evict:
            return ceil(sum(ceil((request.total_tokens - 1) / self._config.block_size) for request in requests))
        else:
            return ceil(sum(ceil(request.num_prefill_tokens / self._config.block_size) for request in requests))

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            #XXX for eviction-free
            if self._config.no_evict:
                num_required_blocks = ceil(
                    (request.total_tokens - 1) / self._config.block_size
                )
            else:

                #XXX: different from SARATHI
                if self._config.histogram:
                    histogram_key = int(log2(request._initial_num_prefill_tokens))
                    histogram_bucket = self.histogram[histogram_key]

                    if histogram_bucket[0] == 0:
                        num_required_tokens = request.num_prefill_tokens
                    else:
                        avg_processed_tokens = histogram_bucket[1] / histogram_bucket[0] # request.total_tokens
                        num_remaining_tokens = avg_processed_tokens - request.num_processed_tokens #request.total_tokens - self.num_processed_tokens
                        num_required_tokens = ceil(request.num_processed_tokens + num_remaining_tokens)  # request.num_prefill_tokens + request.num_decode_tokens # * 3 / 4
                    if request.num_prefill_tokens < self._config.histogram_threshold:
                        num_required_tokens = request.num_prefill_tokens # for short request, do as before, ignore future decoding steps
                    else:
                        num_required_tokens = max(request.num_prefill_tokens, num_required_tokens) # for long request, use max(avg total tokens, current prefill tokens)
                else:
                    num_required_tokens = request.num_prefill_tokens # do the same as SARATHI

                num_required_blocks = ceil(
                    num_required_tokens / self._config.block_size
                )
            #print('can_allocate', request._id, 'prefill', request.num_prefill_tokens, 'total', self._config.num_blocks, 'allocated', self._num_allocated_blocks, 'required', num_required_blocks, 'watermark', self._watermark_blocks)
            # allocate w/o overloading
            if (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            ):
                return True

            # check overloading
            if self._config.overload:
                num_available_blocks = (self._config.num_blocks - self._num_allocated_blocks - self._watermark_blocks)
                num_available_tokens = num_available_blocks * self._config.block_size
                num_overloaded_tokens = num_required_tokens - num_available_tokens
                assert num_overloaded_tokens >= 0
                #XXX overload only short ones
                if num_overloaded_tokens + self.num_overloaded_tokens < self._config.overload_threshold and request.num_prefill_tokens < self._config.overload_prefill_threshold:
                    # store # overloaded tokens and # blocks to allocate
                    self.overload_map[request.id] = [num_overloaded_tokens, num_available_blocks]
                    #request.set_overloaded_tokens(num_overloaded_tokens)
                    request._num_overloaded_tokens = num_overloaded_tokens
                    self.num_overloaded_tokens += num_overloaded_tokens
                    print('[can allocate first] request', request.id, 'requiring', num_required_tokens, 'overload', num_overloaded_tokens, '/', self.num_overloaded_tokens, 'tokens with allocating', num_available_blocks, 'blocks')
                    return True

            return False

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
            if self._config.overload and request.id in self.overload_map:
                # first try to claim more memory for the overloaded tokens
                num_available_blocks = (self._config.num_blocks - self._num_allocated_blocks)
                num_available_tokens = num_available_blocks * self._config.block_size
                num_overloaded_tokens = self.overload_map[request.id][0]
                #num_allocated_tokens = self.overload_map[request.id][1] * self._config.block_size
                #num_required_tokens = num_overloaded_tokens - num_available_tokens
                ret = False
                reason = ''
                if num_overloaded_tokens <= num_available_tokens:
                    # can allocate all
                    num_required_blocks = ceil(num_overloaded_tokens / self._config.block_size)
                    assert num_required_blocks <= num_available_blocks
                    #self.allocate(request.id, num_required_blocks)
                    self.overload_map[request.id][1] = num_required_blocks
                    ret = True
                    reason = 'ALLOC_ALL'
                elif num_available_tokens > 0:
                    # can allocate some
                    num_required_blocks = num_available_blocks
                    #self.allocate(request.id, num_required_blocks)
                    self.overload_map[request.id][1] = num_required_blocks
                    ret = True
                    reason = 'ALLOC_PART'
                else:
                    # cannot allocate more
                    self.overload_map[request.id][1] = 0
                    #if self.num_overloaded_tokens + 1 < self._config.overload_threshold:
                    ret = True
                    reason = 'OVERLOAD'
                    #else:
                    #    ret = False
                    #    reason = 'EVICT'
                print('[can allocate next] request', request.id, reason, 'overload', self.overload_map[request.id][0], '/', self.num_overloaded_tokens,
                    'tokens with allocated', self.overload_map[request.id][1], 'blocks where', num_available_blocks, 'available')
                return ret
            # vllm requires at least one block to be available
            if self._config.partial_evict_block != 1 and not request.is_prefill_complete:
                num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
                assert num_tokens_reserved > 0
                num_tokens_required = max(0, request.num_prefill_tokens - num_tokens_reserved)
                num_required_blocks = ceil(num_tokens_required / self._config.block_size)
                return (
                    self._config.num_blocks
                    - self._num_allocated_blocks
                    - num_required_blocks
                    >= self._watermark_blocks
                )
            else:
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
                if self._config.overload and request.id in self.overload_map:
                    num_required_blocks = self.overload_map[request.id][1]
                    print('[allocate first] request', request.id, num_required_blocks, 'blocks')
                else:
                    # if self.num_completed_requests > 1000:
                    #     avg_processed_tokens = self.num_processed_tokens / self.num_completed_requests
                    # else:
                    #     avg_processed_tokens = 0
                    # avg_processed_tokens = request.total_tokens
                    # num_remaining_tokens = avg_processed_tokens - request.num_processed_tokens #request.total_tokens - self.num_processed_tokens
                    # num_required_tokens = request.num_processed_tokens + num_remaining_tokens * 0.5  # request.num_prefill_tokens + request.num_decode_tokens # * 3 / 4
                    # num_required_tokens = max(request.num_prefill_tokens, num_required_tokens)
                    if self._is_decode_scheduler:
                        num_required_tokens = max(request.num_processed_tokens, request.num_prefill_tokens)
                    else:
                        num_required_tokens = request.num_prefill_tokens
                    num_required_blocks = ceil(
                        num_required_tokens / self._config.block_size
                    )
            self.allocate(request.id, num_required_blocks)
            if self._is_decode_scheduler and request.id == 18818:
                print('allocate R first time', request.id, num_required_blocks)
            return

        if self._config.overload and request.id in self.overload_map:
            num_required_blocks = self.overload_map[request.id][1]
            assert self.overload_map[request.id][0] > 0
            if num_required_blocks > 0:
                self.allocate(request.id, num_required_blocks)
                num_allocated_tokens = num_required_blocks * self._config.block_size
                self.overload_map[request.id][0] -= num_allocated_tokens
                request._num_overloaded_tokens -= num_allocated_tokens
                self.num_overloaded_tokens -= num_allocated_tokens
                assert self.overload_map[request.id][0] >= 0
                assert self.num_overloaded_tokens >= 0
            else:
                self.overload_map[request.id][0] += 1
                request._num_overloaded_tokens += 1
                self.num_overloaded_tokens += 1
                assert self.num_overloaded_tokens < self._config.overload_threshold
            print('[allocate next] request', request.id, 'overloading', self.overload_map[request.id][0], 'tokens and allocated new', num_required_blocks, 'blocks')
            if self.overload_map[request.id][0] == 0:
                self.free_overload(request.id)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)

        # decode can be here
        if self._config.partial_evict_block != 1: # and not request.is_prefill_complete:
            pass
            #print(f"[allocate next] request {request.id} num_tokens_required: {num_tokens_required} num_processed_tokens: {request.num_processed_tokens} num_tokens_reserved: {num_tokens_reserved} is_prefill_complete {request.is_prefill_complete}")
        else:
            assert (
                num_tokens_required == 0 or num_tokens_required == 1
            ), f"request {request.id} num_tokens_required: {num_tokens_required} num_processed_tokens: {request.num_processed_tokens} num_tokens_reserved: {num_tokens_reserved} num_prefill_tokens {request.num_prefill_tokens} is_prefill_complete {request.is_prefill_complete}"

        if num_tokens_required == 0:
            return

        assert not self._config.no_evict, "Should not reach here for eviction-free"

        if self._config.partial_evict_block != 1 and not request.is_prefill_complete:
            num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
            num_tokens_required = max(0, request.num_prefill_tokens - num_tokens_reserved)
            num_required_blocks = ceil(num_tokens_required / self._config.block_size)
            self.allocate(request.id, num_required_blocks)
        else:
            self.allocate(request.id, 1)
            #if self._is_decode_scheduler:
            if self._is_decode_scheduler and request.id == 18818:
                print('allocate R decode', request.id, 1)

    def free_overload(self, *request_ids: List[int]) -> None:
        for request_id in request_ids:
            if request_id in self.overload_map:
                num_tokens = self.overload_map.pop(request_id)[0]
                self.num_overloaded_tokens -= num_tokens

        assert self.num_overloaded_tokens >= 0

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        self.handoff_requests = []

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
                self.free_overload(request.id)
                if not self._is_decode_scheduler and self._decode_scheduler is not None:
                    self.handoff_requests.append(request)
                    if request.id == 18818:
                        print('add handoff', request.id)
                else:
                    #XXX: different from SARATHI
                    self.num_completed_requests += 1
                    self.num_processed_tokens += request.total_tokens
                    histogram_key = int(log2(request._initial_num_prefill_tokens)) # key = initial # prefill tokens
                    self.histogram[histogram_key][0] += 1 # count
                    self.histogram[histogram_key][1] += request.total_tokens # sum
            else:
                self._preempted_requests.append(request)

        #print('MEMORY', self._num_allocated_blocks)

    def _get_request_next_num_tokens(
        self, request: Request, batch_contains_prefill: bool, num_batch_tokens: int
    ) -> int:
        assert not request.completed

        #XXX consider overloading
        if False:
            if request.id in self.overload_map:
                #assert request.is_prefill_complete
                num_overloaded_tokens = self.overload_map[request.id][0]
                assert num_overloaded_tokens > 0
                print('[get next num tokens] request', request.id, 'overloaded', num_overloaded_tokens, 'tokens and needs to process', num_overloaded_tokens, 'tokens')
                return num_overloaded_tokens

        if request.id in self.overload_map:
            assert not request.is_prefill_complete

        if request.is_prefill_complete:
            assert request not in self.overload_map
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

        if self._config.sortI:
            #XXX: why sort by num_prefill_tokens, not all tokens?
            #XXX prefer long-m (m >= I)
            #self._preempted_requests = sorted(self._preempted_requests, key=lambda req: req.num_prefill_tokens, reverse=True)
            self._preempted_requests = sorted(self._preempted_requests, key=lambda req: req.num_processed_tokens, reverse=True)

        #XXX: same as SARATHI -- start
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                #print('B_LIMIT', self._max_micro_batch_size)
                break

            #XXX if so, isn't this scheduling long requests first???
            request = self._preempted_requests.pop(0) # longest

            if not request.is_prefill_complete:
                running_prefills.append(request)
                continue

            next_num_tokens = self._get_request_next_num_tokens(
                request, contains_prefill, num_batch_tokens
            )
            if self._is_decode_scheduler:
                assert next_num_tokens == 1
                if request.id == 18818:
                    print('tokens to process R', request.id, next_num_tokens)

            if next_num_tokens == 0:
                #print('C_LIMIT', request._id)
                skipped_requests.append(request)
                continue

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1) # shortest
                    #if self._config.sortI and len(self._preempted_requests) > 0:
                    #   assert victim_request.num_processed_tokens <= self._preempted_requests[0].num_processed_tokens
                    #XXX - evict small I requests first
                    num_remaining_tokens = self.free(victim_request.id, self._config.partial_evict_block)
                    #print('[free victim] request', victim_request.id, 'remaining', num_remaining_tokens)
                    self.free_overload(victim_request.id)
                    victim_request.restart(num_remaining_tokens) #XXX partial evict
                    if num_remaining_tokens > 0:
                        self._preempted_requests = [victim_request] + self._preempted_requests
                    else:
                        self._request_queue = [victim_request] + self._request_queue
                        #print('M_LIMIT VICTIM', victim_request._id, 'FOR', request.id)
                else:
                    num_remaining_tokens = self.free(request.id, self._config.partial_evict_block)
                    #print('[free self] request', request.id, 'remaining', num_remaining_tokens)
                    self.free_overload(request.id)
                    request.restart(num_remaining_tokens) #XXX partial evict
                    if num_remaining_tokens > 0:
                        self._preempted_requests = [request] + self._preempted_requests
                    else:
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
            return Batch(
                self._replica_id,
                requests,
                num_tokens,
                scheduler=self,
                is_decode=self._is_decode_scheduler,
            )

        #XXX here, don't check _can_allocate_request as they have already been checked

        #XXX: same as SARATHI -- end

        #if self._config.sortI:
        #    #XXX prefer short-I
        #    running_prefills = sorted(running_prefills, key=lambda req: req.num_prefill_tokens) #, reverse=True)
        #    #running_prefills = sorted(running_prefills, key=lambda req: req.num_processed_tokens)

        #XXX: same as SARATHI -- start

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

        #if self._config.sortI:
        #    #XXX prefer short-I
        #    self._request_queue = sorted(self._request_queue, key=lambda req: req.num_prefill_tokens, reverse=True)

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

        batch = Batch(
            self._replica_id,
            requests,
            num_tokens,
            scheduler=self,
            is_decode=self._is_decode_scheduler,
        )
        #print('BATCH', batch._id, [r.id for r in requests])
        return batch

        #XXX: same as SARATHI -- end
