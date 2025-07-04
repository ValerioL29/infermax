from typing import Tuple

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Request has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Request has not been completed yet")
        return func(self, *args, **kwargs)

    return wrapper


class Request(BaseEntity):
    def __init__(
        self,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        num_processed_tokens: int = 0,
        request_id: int = None,
        original_num_prefill_tokens: int = None,
        original_num_decode_tokens: int = None,
    ):
        self._id = request_id if request_id is not None else Request.generate_id()
        self._arrived_at = arrived_at
        self._num_prefill_tokens = num_prefill_tokens
        self._num_decode_tokens = num_decode_tokens
        self._num_processed_tokens = num_processed_tokens

        self._initial_num_decode_tokens = num_decode_tokens
        self._initial_num_prefill_tokens = num_prefill_tokens
        self._original_num_prefill_tokens = (
            original_num_prefill_tokens
            if original_num_prefill_tokens is not None
            else num_prefill_tokens
        )
        self._original_num_decode_tokens = (
            original_num_decode_tokens
            if original_num_decode_tokens is not None
            else num_decode_tokens
        )

        self._scheduled_at = 0
        self._execution_time = 0
        self._model_execution_time = 0
        self._scheduling_delay = 0
        self._preempted_time = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0

        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts = 0
        self._just_evicted = False #XXX just used scheduled?
        self._num_generated_tokens = 0
        self._prev_m = 0
        self._prev_prev_m = 0
        self._prev_B = num_prefill_tokens

        #XXX for TPOT
        self._TPOT = []

        self._num_overloaded_tokens = 0

    @property
    def csp_m(self):
        return self.num_processed_tokens - 1 if self.is_prefill_complete else self.num_processed_tokens

    @property
    def csp_B(self):
        return self.num_prefill_tokens

    #XXX always preempted after the batch finishes
    def csp_e(self):
        #if self._id == 24:
        #    print('csp_e 24', 'je?', self._just_evicted, self._num_prefill_tokens, self._num_decode_tokens, self._num_processed_tokens)
        #assert self.scheduled != self._evicted, f'rid {self.id}, s {self.scheduled}, e {self._evicted}, whole {self}'
        #return 1 if self.is_prefill_complete and self.preempted and not self.scheduled else 0
        if self._just_evicted:
            self._just_evicted = False
            return 1
        return 0

    def csp_delta_gen_z_c(self):
        #return 1 if self.is_prefill_complete else 0
        #return 1 if self.has_started_decode else 0
        delta = 1 if self._prev_m >= self._prev_B else 0
        z     = 1 if self.csp_m > self._prev_m else 0
        c     = max(0, self.csp_m - self._prev_m)
        gen   = 1 if (z > 0 and self.csp_m >= self.num_prefill_tokens) else 0 #if (delta > 0 and self._prev_m > self._prev_prev_m) else 0

        self._prev_prev_m = self._prev_m
        self._prev_m = self.csp_m #self.num_processed_tokens
        self._prev_B = self.num_prefill_tokens
        return delta, gen, z, c

    @property
    def TPOT(self):
        return self._TPOT

    @property
    def TTFT(self):
        return self._prefill_completed_at - self._arrived_at

    @property
    def size(self) -> Tuple[int, int]:
        return (self._num_prefill_tokens, self._num_decode_tokens)

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_scheduled
    def latest_stage_scheduled_at(self) -> float:
        return self._latest_stage_scheduled_at

    @property
    @check_scheduled
    def latest_stage_completed_at(self) -> float:
        return self._latest_stage_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduled_at(self) -> float:
        return self._latest_iteration_scheduled_at

    @property
    @check_scheduled
    def latest_iteration_completed_at(self) -> float:
        return self._latest_iteration_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduling_delay(self) -> float:
        return self._latest_iteration_scheduling_delay

    @property
    @check_scheduled
    def prefill_completed_at(self) -> float:
        return self._prefill_completed_at

    @property
    @check_scheduled
    def scheduling_delay(self) -> float:
        return self._scheduling_delay

    @property
    @check_scheduled
    def preempted_time(self) -> float:
        return self._preempted_time

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    @check_scheduled
    def e2e_time(self) -> float:
        return self._completed_at - self._arrived_at

    @property
    @check_scheduled
    def e2e_time_normalized(self) -> float:
        return self.e2e_time / self.num_decode_tokens

    @property
    @check_scheduled
    def execution_time(self) -> float:
        return self._execution_time

    @property
    @check_scheduled
    def execution_time_normalized(self) -> float:
        return self._execution_time / self.num_decode_tokens

    @property
    @check_scheduled
    def model_execution_time(self) -> float:
        return self._model_execution_time

    @property
    @check_scheduled
    def model_execution_time_normalized(self) -> float:
        return self._model_execution_time / self.num_decode_tokens

    @property
    def arrived_at(self) -> float:
        return self._arrived_at

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def pd_ratio(self) -> float:
        return self._num_prefill_tokens / self._num_decode_tokens

    @property
    def num_processed_tokens(self) -> int:
        return self._num_processed_tokens

    @property
    def original_num_prefill_tokens(self) -> int:
        return self._original_num_prefill_tokens

    @property
    def original_num_decode_tokens(self) -> int:
        return self._original_num_decode_tokens

    @property
    def total_tokens(self) -> int:
        return self._num_prefill_tokens + self._num_decode_tokens

    @property
    def num_processed_prefill_tokens(self) -> int:
        return min(self._num_processed_tokens, self._num_prefill_tokens)

    @property
    def num_processed_decode_tokens(self) -> int:
        return max(self._num_processed_tokens - self._num_prefill_tokens, 0)

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def preempted(self) -> bool:
        return self._preempted and not self._completed

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def num_restarts(self) -> int:
        return self._num_restarts

    @property
    def is_prefill_complete(self) -> bool:
        return self._is_prefill_complete

    @property
    def has_started_decode(self) -> bool:
        return self._num_processed_tokens > self._num_prefill_tokens + 1

    def on_batch_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_iteration_scheduled_at = time
        self._latest_iteration_scheduling_delay = (
            time - self._latest_iteration_completed_at
        )

        if self._scheduled:
            return

        if self._num_restarts > 0:
            self._scheduled = True
            #self._evicted = False
            return

        self._scheduled_at = time
        self._scheduling_delay = time - self._arrived_at
        self._scheduled = True

    def on_batch_end(
        self,
        time: float,
        num_tokens_processed: int,
    ) -> None:
        prev_latest = self._latest_iteration_completed_at
        self._num_processed_tokens += num_tokens_processed
        self._latest_iteration_completed_at = time
        assert self._num_processed_tokens <= self.total_tokens, f'request {self.id} processed {self._num_processed_tokens} total {self.total_tokens} prefill {self.num_prefill_tokens} decode {self.num_decode_tokens}'

        if self._num_processed_tokens == self._num_prefill_tokens:
            if self.id == 96:
                print('[on batch end] request', self.id, 'is_prefill_complete True')
            self._is_prefill_complete = True
            # we get one decode token when the prefill processing completes
            self._num_processed_tokens += 1

            # we must record the prefill completion time only in the first time
            # in the subsequent restarts, we keep adding the previously decoded
            # tokens to the prefill tokens - that is irrelevant to the original prefill
            if self._prefill_completed_at == 0:
                self._prefill_completed_at = time

        # check if request is completed
        if self._num_processed_tokens == self.total_tokens:
            self._completed_at = time
            self._completed = True
            logger.debug(f"Request {self._id} completed at {self._completed_at}")

        #if self._id == 331:
        #    print('process 331', self._num_prefill_tokens, self._num_decode_tokens, self._num_processed_tokens)

        #XXX for TPOT
        if self._is_prefill_complete and time > self._prefill_completed_at:
            self._TPOT.append(float(time - prev_latest))
            self._num_generated_tokens += 1
            #print('add TPOT for R', self._id, self._TPOT[-1])

        if self._completed:
            return

        # TODO: consider overloading, that not all processed tokens may be added to memory
        # case 1: if prefill = 300, overload 200, m = 100, then processed = 300 + 1 = 301 here,
        if self._num_overloaded_tokens > 0:
            prev_prefill_tokens = self._num_prefill_tokens
            prev_processed_tokens = self._num_processed_tokens
            prev_decode_tokens = self._num_decode_tokens
            if self._num_prefill_tokens < self._num_processed_tokens:
                #XXX: WRONG, that num_processed_tokens can be smaller than num_prefill_tokens, in case of chunked prefill!!
                total_tokens = self._num_prefill_tokens + self._num_decode_tokens
                self._num_prefill_tokens = self._num_processed_tokens
                self._num_decode_tokens = total_tokens - self._num_prefill_tokens
                assert self._is_prefill_complete
                phase = 'decode'
            else:
                phase = 'prefill'

            self._num_processed_tokens -= self._num_overloaded_tokens # case 1: -200
            if self._is_prefill_complete:
                self._num_processed_tokens -= 1 # case 1: -1, so 100 left
                if self._num_processed_tokens < self._num_prefill_tokens:
                    self._is_prefill_complete = False

            print('[on batch end] request', self.id, 'overloaded', self._num_overloaded_tokens, 'phase was', phase,
                'prefill', prev_prefill_tokens, '->', self._num_prefill_tokens,
                'decode', prev_decode_tokens, '->', self._num_decode_tokens,
                'processed', prev_processed_tokens, '->', self._num_processed_tokens)

    def on_batch_stage_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_stage_scheduled_at = time
        if self._latest_stage_completed_at == 0:
            self._preempted_time = 0
        else:
            self._preempted_time += time - self._latest_stage_completed_at
        self._preempted = False

    def on_batch_stage_end(
        self,
        time: float,
        execution_time: float,
        model_execution_time: float,
    ) -> None:
        self._execution_time += execution_time
        self._model_execution_time += model_execution_time
        self._latest_stage_completed_at = time
        self._preempted = True

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "arrived_at": self._arrived_at,
            "execution_time": self._execution_time,
            "model_execution_time": self._model_execution_time,
            "scheduled_at": self._scheduled_at,
            "scheduling_delay": self._scheduling_delay,
            "preempted_time": self._preempted_time,
            "completed_at": self._completed_at,
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "num_processed_tokens": self._num_processed_tokens,
            "scheduled": self._scheduled,
            "preempted": self._preempted,
            "completed": self._completed,
            "latest_stage_scheduled_at": self._latest_stage_scheduled_at,
            "latest_stage_completed_at": self._latest_stage_completed_at,
            "latest_iteration_scheduled_at": self._latest_iteration_scheduled_at,
            "latest_iteration_completed_at": self._latest_iteration_completed_at,
            "num_restarts": self._num_restarts,
        }

    def restart(self, num_remaining_tokens: int = 0):
        logger.debug(f"Restarting request {self._id} with remaining tokens {num_remaining_tokens}")

        # when we restart the request, we can process all the previously
        # decoded tokens in parallel (i.e., we can prefill all the tokens)
        #XXX
        #if self._id == 331:
        #    print('restart 331', self._num_prefill_tokens, self._num_decode_tokens, self._num_processed_tokens)

        prev_prefill_tokens = self._num_prefill_tokens
        prev_decode_tokens = self._num_decode_tokens
        prev_processed_tokens = self._num_processed_tokens

        #XXX: added to fix error
        if self._num_prefill_tokens < self._num_processed_tokens:
            #XXX: WRONG, that num_processed_tokens can be smaller than num_prefill_tokens, in case of chunked prefill!!
            total_tokens = self._num_prefill_tokens + self._num_decode_tokens
            self._num_prefill_tokens = self._num_processed_tokens
            self._num_decode_tokens = total_tokens - self._num_prefill_tokens

        self._num_processed_tokens = num_remaining_tokens
        if self._num_processed_tokens == 0:
            self._scheduled = False
            self._preempted = False
            self._completed = False
        if self._num_processed_tokens < self._num_prefill_tokens:
            self._is_prefill_complete = False

        if num_remaining_tokens > 0:
            pass
            print('[restart partial] request', self.id, 'remaining', num_remaining_tokens,
                'prefill', prev_prefill_tokens, '->', self._num_prefill_tokens,
                'decode', prev_decode_tokens, '->', self._num_decode_tokens,
                'processed', prev_processed_tokens, '->', self._num_processed_tokens,
                'is prefill complete', self._is_prefill_complete)


        self._num_restarts += 1
        #if self._id == 24:
        #    print('restart 24', self._num_prefill_tokens, self._num_decode_tokens, self._num_processed_tokens)
        self._just_evicted = True
