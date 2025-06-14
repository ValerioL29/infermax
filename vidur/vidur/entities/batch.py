from typing import List

from vidur.entities.base_entity import BaseEntity
from vidur.entities.request import Request
from vidur.logger import init_logger

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


class Batch(BaseEntity):
    def __init__(
        self,
        replica_id: int,
        requests: List[Request],
        num_tokens: List[int],
        scheduler=None,
        is_decode: bool = False,
    ) -> None:
        self._id = Batch.generate_id()
        self._replica_id = replica_id

        self._scheduler = scheduler
        self._is_decode = is_decode

        self._requests = requests
        self._num_tokens = num_tokens
        self._total_num_tokens = sum(num_tokens)
        self._num_prefill_tokens = sum(
            [
                (t if not r.is_prefill_complete else 0)
                for r, t in zip(self.requests, self._num_tokens)
            ]
        )
        self._num_prefill_tokens_square = sum(
            [
                (t ** 2 if not r.is_prefill_complete else 0)
                for r, t in zip(self.requests, self._num_tokens)
            ]
        )
        self._num_prefill_tokens_dot_kvs = sum(
            [
                (t * r.num_processed_tokens if not r.is_prefill_complete else 0)
                for r, t in zip(self.requests, self._num_tokens)
            ]
        )

        self._total_num_tokens_rounded = (self._total_num_tokens + 7) // 8 * 8
        #assert(self._total_num_tokens_rounded <= 16384, f"self._total_num_tokens: {self._total_num_tokens} self._total_num_tokens_rounded: {self._total_num_tokens_rounded}")

        self._scheduled_at = None
        self._completed_at = None
        self._scheduled = False
        self._completed = False

        self._num_prefill_requests = len(
            [
                r for r in self.requests if not r.is_prefill_complete
            ]
        )
        self._num_decode_requests = len(
            [
                r for r in self.requests if r.is_prefill_complete
            ]
        )
        assert (self._num_prefill_requests) + (self._num_decode_requests) == self.size
        self._num_prefill_kvs = sum(
            [
                r.num_processed_tokens for r in self.requests if not r.is_prefill_complete
            ]
        )
        self._num_decode_kvs = sum(
            [
                r.num_processed_tokens for r in self.requests if r.is_prefill_complete
            ]
        )

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def is_decode(self) -> bool:
        return self._is_decode

    @property
    def num_prefill_requests(self) -> int:
        return self._num_prefill_requests

    @property
    def num_decode_requests(self) -> int:
        return self._num_decode_requests

    @property
    def num_prefill_kvs(self) -> int:
        return self._num_prefill_kvs

    @property
    def num_decode_kvs(self) -> int:
        return self._num_decode_kvs

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def creation_time(self) -> float:
        return self._creation_time

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens

    @property
    def total_num_tokens(self) -> int:
        return self._total_num_tokens

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_prefill_tokens_square(self) -> int:
        return self._num_prefill_tokens_square

    @property
    def num_prefill_tokens_dot_kvs(self) -> int:
        return self._num_prefill_tokens_dot_kvs

    @property
    def num_decode_tokens(self) -> int:
        return self.total_num_tokens - self.num_prefill_tokens

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def size(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def request_ids(self) -> List[int]:
        return [request.id for request in self._requests]

    @property
    def all_requests_completed(self) -> bool:
        return all([request.completed for request in self._requests])

    def on_schedule(
        self,
        time: float,
    ) -> None:
        self._scheduled_at = time
        self._scheduled = True

        for request in self._requests:
            request.on_batch_schedule(time)

    def on_batch_end(self, time: float):
        self._completed = True
        self._completed_at = time

        for request, num_tokens in zip(self._requests, self._num_tokens):
            request.on_batch_end(time, num_tokens)

    @property
    def preempted_requests(self) -> List[Request]:
        return [request for request in self._requests if request.preempted]

    @property
    def completed_requests(self) -> List[Request]:
        return [request for request in self._requests if request.completed]

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "size": self.size,
            "replica_id": self._replica_id,
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
            "num_prefill_tokens": self.num_prefill_tokens,
            "num_decode_tokens": self.num_decode_tokens,
        }
