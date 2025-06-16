from typing import List

from vidur.entities import Batch, Request
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        self._batch.on_batch_end(self.time)
        replica_scheduler = self._batch.scheduler
        replica_scheduler.on_batch_end(self._batch)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent,
            #XXX for CSP
            replica_scheduler
        )

        events = [ReplicaScheduleEvent(self.time, replica_scheduler)]

        if scheduler.config.cluster_config.replica_scheduler_config.dual_schedulers:
            for req in getattr(replica_scheduler, "handoff_requests", []):
                decode_request = Request(
                    arrived_at=self.time,
                    #num_prefill_tokens=1,
                    num_prefill_tokens=req.num_prefill_tokens, #XXX not sure if this should be 1
                    num_decode_tokens=req.original_num_decode_tokens - 1,
                    num_processed_tokens=req.num_prefill_tokens + 1,
                    request_id=req.id,
                    original_num_prefill_tokens=req.original_num_prefill_tokens,
                    original_num_decode_tokens=req.original_num_decode_tokens,
                )
                decode_request._prefill_completed_at = req._prefill_completed_at
                decode_request._latest_stage_scheduled_at = req._latest_stage_scheduled_at
                decode_request._latest_stage_completed_at = req._latest_stage_completed_at
                decode_request._latest_iteration_scheduled_at = req._latest_iteration_scheduled_at
                decode_request._latest_iteration_completed_at = req._latest_iteration_completed_at
                decode_request._preempted = True
                decode_request._is_prefill_complete = True
                ds = scheduler.get_decode_scheduler(self._replica_id)
                ds.add_request(decode_request)
                events.append(ReplicaScheduleEvent(self.time, ds))
                #print('add D', req.id, 'at', self.time)

        return events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }
