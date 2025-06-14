from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)


class BaseGlobalScheduler(ABC):
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        self._config = config
        self._replicas = replicas

        self._num_replicas = len(self._replicas)
        self._dual_schedulers = (
            config.cluster_config.replica_scheduler_config.dual_schedulers
        )

        execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            replica_scheduler_config=config.cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
        )
        self._prefill_schedulers = {}
        self._decode_schedulers = {}
        for replica_id, replica in replicas.items():
            prefill = ReplicaSchedulerRegistry.get(
                config.cluster_config.replica_scheduler_config.get_type(),
                replica_config=config.cluster_config.replica_config,
                replica_scheduler_config=config.cluster_config.replica_scheduler_config,
                request_generator_config=config.request_generator_config,
                replica=replica,
                num_stages=replica.num_pipeline_stages,
                execution_time_predictor=execution_time_predictor,
            )
            if self._dual_schedulers:
                decode = ReplicaSchedulerRegistry.get(
                    config.cluster_config.replica_scheduler_config.get_type(),
                    replica_config=config.cluster_config.replica_config,
                    replica_scheduler_config=config.cluster_config.replica_scheduler_config,
                    request_generator_config=config.request_generator_config,
                    replica=replica,
                    num_stages=replica.num_pipeline_stages,
                    execution_time_predictor=execution_time_predictor,
                    is_decode_scheduler=True,
                )
                prefill.set_decode_scheduler(decode)
                self._decode_schedulers[replica_id] = decode
            else:
                self._decode_schedulers[replica_id] = prefill
            self._prefill_schedulers[replica_id] = prefill
        self._replica_schedulers = self._prefill_schedulers
        self._request_queue = []

    def sort_requests(self) -> None:
        self._request_queue.sort(key=lambda request: request._arrived_at)

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_scheduler(self, replica_id: int):
        return self._prefill_schedulers[replica_id]

    def get_decode_scheduler(self, replica_id: int):
        return self._decode_schedulers[replica_id]

    def get_replica_stage_scheduler(self, replica_id: int, stage_id: int, decode: bool = False):
        sched = self._decode_schedulers[replica_id] if decode else self._prefill_schedulers[replica_id]
        return sched.get_replica_stage_scheduler(stage_id)

    def is_empty(self) -> bool:
        if self._dual_schedulers:
            return len(self._request_queue) == 0 and all(
                ps.is_empty() and ds.is_empty()
                for ps, ds in zip(
                    self._prefill_schedulers.values(), self._decode_schedulers.values()
                )
            )
        else:
            return len(self._request_queue) == 0 and all(
                sched.is_empty() for sched in self._prefill_schedulers.values()
            )

    @property
    def config(self) -> SimulationConfig:
        return self._config

    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:
        pass
