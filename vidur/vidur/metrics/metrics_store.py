import os
from functools import reduce
from typing import Dict, List

import pandas as pd
import plotly_express as px
import wandb

from vidur.config import SimulationConfig
from vidur.entities import Batch, BatchStage, ExecutionTime, Request
from vidur.logger import init_logger
from vidur.metrics.cdf_sketch import CDFSketch
from vidur.metrics.constants import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    CpuOperationMetrics,
    OperationMetrics,
    RequestCompletionMetricsTimeSeries,
    RequestMetricsHistogram,
    RequestMetricsTimeDistributions,
    TokenCompletionMetricsTimeSeries,
    TokenMetricsTimeDistribution,
)
from vidur.metrics.data_series import DataSeries
from vidur.metrics.series_average_meter import SeriesAverageMeter
from vidur.utils.mfu_calculator import MFUCalculator

logger = init_logger(__name__)


def if_write_metrics(func):
    def wrapper(self, *args, **kwargs):
        if self._config.write_metrics:
            return func(self, *args, **kwargs)

    return wrapper


REQUEST_ID_STR = "Request Id"
COUNT_STR = "Count"
TIME_STR = "Time (sec)"
BATCH_ID_STR = "Batch Id"
MEMORY_USAGE_STR = "Memory Usage (%)"
BUSY_TIME_PERCENT = "Busy Time (%)"
UTILIZATION_STR = "Utilization (%)"
OPERATION_STR = "Operation"
TIME_STR_MS = "Time (ms)"


class MetricsStore:

    def __init__(self, simulation_config: SimulationConfig) -> None:
        self._simulation_config = simulation_config
        self._config = self._simulation_config.metrics_config
        self._last_request_arrived_at = None

        # copy config
        self._num_replicas = self._simulation_config.cluster_config.num_replicas
        self._num_pipeline_stages = (
            self._simulation_config.cluster_config.replica_config.num_pipeline_stages
        )

        # Initialise request metrics
        self._request_metrics_time_distributions: Dict[
            RequestMetricsTimeDistributions, DataSeries
        ] = {}
        for metric_name in RequestMetricsTimeDistributions:
            self._request_metrics_time_distributions[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        self._token_metrics_time_distribution: Dict[
            TokenMetricsTimeDistribution, DataSeries
        ] = {}
        for metric_name in TokenMetricsTimeDistribution:
            self._token_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        self._request_metrics_histogram: Dict[RequestMetricsHistogram, DataSeries] = {}
        for metric_name in RequestMetricsHistogram:
            self._request_metrics_histogram[metric_name] = DataSeries(
                REQUEST_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Initialise batch metrics
        self._batch_metrics_count_distribution: Dict[
            BatchMetricsCountDistribution, CDFSketch
        ] = {}
        self._batch_metrics_count_distribution_per_batch: Dict[
            BatchMetricsCountDistribution, DataSeries
        ] = {}
        for metric_name in BatchMetricsCountDistribution:
            self._batch_metrics_count_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._batch_metrics_count_distribution_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        self._batch_metrics_time_distribution: Dict[
            BatchMetricsTimeDistribution, CDFSketch
        ] = {}
        self._batch_metrics_time_distribution_per_batch: Dict[
            BatchMetricsTimeDistribution, DataSeries
        ] = {}
        for metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._batch_metrics_time_distribution_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Initialise completion metrics
        self._request_completion_metrics_time_series: Dict[
            RequestCompletionMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in RequestCompletionMetricsTimeSeries:
            self._request_completion_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
        self._token_completion_metrics_time_series: Dict[
            TokenCompletionMetricsTimeSeries, DataSeries
        ] = {}
        for metric_name in TokenCompletionMetricsTimeSeries:
            self._token_completion_metrics_time_series[metric_name] = DataSeries(
                TIME_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # Initialise operation metrics
        self._operation_metrics: Dict[OperationMetrics, CDFSketch] = {}
        self._operation_metrics_per_batch: Dict[OperationMetrics, DataSeries] = {}
        for metric_name in OperationMetrics:
            self._operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._operation_metrics_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        self._cpu_operation_metrics: Dict[CpuOperationMetrics, CDFSketch] = {}
        self._cpu_operation_metrics_per_batch: Dict[CpuOperationMetrics, DataSeries] = (
            {}
        )
        for metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name] = CDFSketch(
                metric_name.value,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )
            self._cpu_operation_metrics_per_batch[metric_name] = DataSeries(
                BATCH_ID_STR,
                metric_name.value,
                self._config.subsamples,
                self._config.save_table_to_wandb,
                self._config.store_plots,
            )

        # per replica metrics
        self._replica_memory_usage = []
        # per replica stage metrics
        self._replica_busy_time = []
        self._replica_mfu = []
        self._mfu_calculator = MFUCalculator(
            self._simulation_config.cluster_config.replica_config
        )
        #XXX for CSP
        self._replica_csp_vars = [{'m': {}, 'B': {}, 'e': {}, 'delta': {}, 'gen': {}, 'z': {}, 'c': {}} for _ in range(self._num_replicas)]
        self._vllm_schedule = {'e': {}, 'p': {}, 'c': {}, 'first_batch_id': {}, 'last_batch_id': {}, 'generated_tokens': {}}

        for replica_idx in range(self._num_replicas):
            self._replica_memory_usage.append(
                SeriesAverageMeter(
                    TIME_STR,
                    MEMORY_USAGE_STR,
                    self._config.save_table_to_wandb,
                )
            )
            self._replica_memory_usage[replica_idx].put(0, 0)

            self._replica_busy_time.append([])
            self._replica_mfu.append([])

            for stage_idx in range(self._num_pipeline_stages):
                self._replica_busy_time[replica_idx].append(
                    SeriesAverageMeter(
                        TIME_STR,
                        BUSY_TIME_PERCENT,
                        save_table_to_wandb=self._config.save_table_to_wandb,
                    )
                )
                self._replica_busy_time[replica_idx][stage_idx].put(0, 0)

                self._replica_mfu[replica_idx].append(
                    SeriesAverageMeter(
                        TIME_STR,
                        UTILIZATION_STR,
                        save_table_to_wandb=self._config.save_table_to_wandb,
                    )
                )
                self._replica_mfu[replica_idx][stage_idx].put(0, 0)

        self._init_wandb()

    def _init_wandb(self):
        if (
            not self._config.write_metrics
            or not self._config.wandb_project
            or not self._config.wandb_group
        ):
            return

        wandb.init(
            project=self._config.wandb_project,
            group=self._config.wandb_group,
            name=self._config.wandb_run_name,
            config=self._simulation_config.to_dict(),
        )

    def _save_as_csv(
        self,
        dataseries_list: List[DataSeries],
        key_to_join: str,
        base_path: str,
        file_name: str,
    ):
        os.makedirs(base_path, exist_ok=True)

        for dataseries in dataseries_list:
            print('dataseries')
            print(dataseries._to_df())
            print('key_to_join')
            print(key_to_join)

        # id: 0, ..., N-1
        if max(dataseries_list[0]._to_df()[key_to_join]) + 1 == len(dataseries_list[0]):
            print('merged')
            merged_df = reduce(
                lambda left, right: pd.merge(left, right, on=[key_to_join], how="outer"),
                [dataseries._to_df() for dataseries in dataseries_list],
            )
            merged_df.to_csv(f"{base_path}/{file_name}.csv", index=False)
            if wandb.run and self._config.save_table_to_wandb:
                wand_table = wandb.Table(dataframe=merged_df)
                wandb.log({f"{file_name}_table": wand_table}, step=0)
        else:
            print('not merged')
            for dataseries in dataseries_list:
                df = dataseries._to_df()
                col_name = df.columns.values.tolist()[-1]
                print('col_name', col_name)
                df.to_csv(f'{base_path}/{file_name}_{col_name}.csv')

    def _store_bar_plot(
        self,
        base_path: str,
        plot_name: str,
        x_label: str,
        y_label: str,
        data: Dict[str, float],
    ):
        if wandb.run:
            wandb.log(
                {
                    plot_name: wandb.plot.bar(
                        wandb.Table(
                            dataframe=pd.DataFrame(
                                data=data.items(), columns=[x_label, y_label]
                            )
                        ),
                        x_label,
                        y_label,
                        title=plot_name,
                    )
                },
                step=0,
            )
        if self._config.store_plots:
            fig = px.bar(
                x=list(data.keys()),
                y=list(data.values()),
                labels={"x": x_label, "y": y_label},
            )
            fig.write_image(f"{base_path}/{plot_name}.png")

    def _store_operation_metrics(self, base_plot_path: str):
        if not self._config.store_operation_metrics:
            return

        total_operation_runtimes: Dict[str, float] = {}

        total_operation_runtimes["model_execution_e2e"] = 0
        for dataseries in self._operation_metrics.values():
            dataseries.plot_cdf(
                base_plot_path, f"{dataseries._metric_name}_execution_time", TIME_STR_MS
            )
            total_operation_runtimes[dataseries._metric_name] = dataseries.sum
            total_operation_runtimes["model_execution_e2e"] += dataseries.sum

        for dataseries in self._cpu_operation_metrics.values():
            dataseries.plot_cdf(
                base_plot_path, f"{dataseries._metric_name}_execution_time", TIME_STR_MS
            )
            total_operation_runtimes[dataseries._metric_name] = dataseries.sum

        self._store_bar_plot(
            base_plot_path,
            "total_operation_runtimes",
            OPERATION_STR,
            TIME_STR_MS,
            total_operation_runtimes,
        )

        if not self._config.keep_individual_batch_metrics:
            return

        for dataseries in self._operation_metrics_per_batch.values():
            dataseries.consolidate()
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=TIME_STR_MS,
                y_cumsum=False,
            )
        operations_dataseries_list = list(self._operation_metrics_per_batch.values())
        self._save_as_csv(
            dataseries_list=operations_dataseries_list,
            key_to_join=BATCH_ID_STR,
            base_path=self._config.output_dir,
            file_name="operation_metrics",
        )

        for dataseries in self._cpu_operation_metrics_per_batch.values():
            dataseries.consolidate()
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=TIME_STR_MS,
                y_cumsum=False,
            )
        cpu_operations_dataseries_list = list(
            self._cpu_operation_metrics_per_batch.values()
        )
        self._save_as_csv(
            dataseries_list=cpu_operations_dataseries_list,
            key_to_join=BATCH_ID_STR,
            base_path=self._config.output_dir,
            file_name="cpu_operation_metrics",
        )

    def _store_request_metrics(self, base_plot_path: str):
        if not self._config.store_request_metrics:
            return

        all_request_metrics = list(
            self._request_metrics_time_distributions.values()
        ) + list(self._request_metrics_histogram.values())

        self._save_as_csv(
            dataseries_list=all_request_metrics,
            key_to_join=REQUEST_ID_STR,
            base_path=self._config.output_dir,
            file_name="request_metrics",
        )

        if self._config.store_csp:
            #self._store_csp(
            #    base_path=self._config.output_dir,
            #    file_name="csp.pkl",
            #)
            import pickle
            with open(self._config.output_dir + '/csp.pkl', 'wb') as f:
                pickle.dump(self._replica_csp_vars, f)

        if self._config.store_schedule:
            import pickle
            assert 0 in self._vllm_schedule['c'], f'self.schedule {self._vllm_schedule["c"]}'
            for request_id in self._vllm_schedule['c'][0].keys():
                bids = [batch_id for batch_id in self._vllm_schedule['c'].keys() if request_id in self._vllm_schedule['c'][batch_id] and self._vllm_schedule['c'][batch_id][request_id] > 0]
                self._vllm_schedule['first_batch_id'][request_id] = min(bids)
                self._vllm_schedule['last_batch_id'][request_id]  = max(bids)
                #self._vllm_schedule['generated_tokens'][request_id]  = sum([self._replica_csp_vars[0]['gen'][batch_id][request_id] for batch_id in self._replica_csp_vars[0]['gen'].keys() if request_id in self._replica_csp_vars[0]['gen'][batch_id]])
            with open(self._config.output_dir + '/schedule.pkl', 'wb') as f:
                pickle.dump(self._vllm_schedule, f)

        #XXX
        if self._config.store_plots:
            for dataseries in self._request_metrics_histogram.values():
                dataseries.plot_histogram(base_plot_path, dataseries._y_name)

            for dataseries in self._request_metrics_time_distributions.values():
                dataseries.plot_cdf(base_plot_path, dataseries._y_name, TIME_STR)

    def _store_batch_metrics(self, base_plot_path: str):
        if not self._config.store_batch_metrics:
            return

        for dataseries in self._batch_metrics_time_distribution.values():
            y_axis_label = (
                TIME_STR_MS
                if "model_execution" in dataseries._metric_name
                else TIME_STR
            )
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, y_axis_label)

        for dataseries in self._batch_metrics_count_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, COUNT_STR)

        if not self._config.keep_individual_batch_metrics:
            return

        for dataseries in self._batch_metrics_time_distribution_per_batch.values():
            y_axis_label = (
                TIME_STR_MS
                if "model_execution" in dataseries._metric_name
                else TIME_STR
            )
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=y_axis_label,
                y_cumsum=False,
            ),

        for dataseries in self._batch_metrics_count_distribution_per_batch.values():
            dataseries.plot_step(
                base_plot_path,
                f"{dataseries._metric_name}_per_batch",
                y_axis_label=COUNT_STR,
                y_cumsum=False,
            ),

        all_batch_metrics = list(
            self._batch_metrics_count_distribution_per_batch.values()
        ) + list(self._batch_metrics_time_distribution_per_batch.values())

        self._save_as_csv(
            dataseries_list=all_batch_metrics,
            key_to_join=BATCH_ID_STR,
            base_path=self._config.output_dir,
            file_name="batch_metrics",
        )

    def _store_completion_metrics(self, base_plot_path: str):
        if self._config.store_request_metrics:
            for dataseries in self._request_completion_metrics_time_series.values():
                dataseries.plot_step(
                    base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
                )

        if not self._config.store_token_completion_metrics:
            return

        for dataseries in self._token_metrics_time_distribution.values():
            dataseries.plot_cdf(base_plot_path, dataseries._metric_name, TIME_STR)

        for dataseries in self._token_completion_metrics_time_series.values():
            dataseries.plot_step(
                base_plot_path, f"{dataseries._y_name}_time_series", COUNT_STR
            )

    def _store_utilization_metrics(self, base_plot_path: str):
        if not self._config.store_utilization_metrics:
            return

        for replica_idx in range(self._num_replicas):
            self._replica_memory_usage[replica_idx].print_stats(
                f"replica_{replica_idx + 1}_memory_usage", base_plot_path
            )
            for stage_idx in range(self._num_pipeline_stages):
                self._replica_busy_time[replica_idx][stage_idx].print_stats(
                    f"replica_{replica_idx + 1}_stage_{stage_idx + 1}_busy_time_percent",
                    base_plot_path,
                )
                self._replica_mfu[replica_idx][stage_idx].print_stats(
                    f"replica_{replica_idx + 1}_stage_{stage_idx + 1}_mfu",
                    base_plot_path,
                )

    @if_write_metrics
    def plot(self) -> None:
        dir_plot_path = f"{self._config.output_dir}/plots"
        os.makedirs(dir_plot_path, exist_ok=True)

        self._store_request_metrics(dir_plot_path)
        self._store_batch_metrics(dir_plot_path)
        self._store_completion_metrics(dir_plot_path)
        self._store_operation_metrics(dir_plot_path)
        self._store_utilization_metrics(dir_plot_path)

    @if_write_metrics
    def on_request_arrival(self, time: float, request: Request) -> None:
        if not self._config.store_request_metrics:
            return

        self._request_completion_metrics_time_series[
            RequestCompletionMetricsTimeSeries.REQUEST_ARRIVAL
        ].put(time, 1)

        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_NUM_TOKENS].put(
            request.id, request.total_tokens
        )
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_PREFILL_TOKENS
        ].put(request.id, request.num_prefill_tokens)
        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_DECODE_TOKENS
        ].put(request.id, request.num_decode_tokens)
        self._request_metrics_histogram[RequestMetricsHistogram.REQUEST_PD_RATIO].put(
            request.id, request.pd_ratio
        )
        if self._last_request_arrived_at is not None:
            self._request_metrics_histogram[
                RequestMetricsHistogram.REQUEST_INTER_ARRIVAL_DELAY
            ].put(request.id, request.arrived_at - self._last_request_arrived_at)
        self._last_request_arrived_at = request.arrived_at

    @if_write_metrics
    def _on_request_end(self, time: float, request: Request) -> None:
        if not self._config.store_request_metrics:
            return

        self._request_completion_metrics_time_series[
            RequestCompletionMetricsTimeSeries.REQUEST_COMPLETION
        ].put(request.completed_at, 1)

        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME
        ].put(request.id, request.e2e_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_E2E_TIME_NORMALIZED
        ].put(request.id, request.e2e_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME
        ].put(request.id, request.execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME
        ].put(request.id, request.model_execution_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_MODEL_EXECUTION_TIME_NORMALIZED
        ].put(request.id, request.model_execution_time_normalized)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_PREEMPTION_TIME
        ].put(request.id, request.preempted_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
        ].put(request.id, request.scheduling_delay)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME
        ].put(request.id, request.execution_time + request.preempted_time)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED
        ].put(
            request.id,
            (request.execution_time + request.preempted_time)
            / request.num_decode_tokens,
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_E2E
        ].put(request.id, request.prefill_completed_at - request.arrived_at)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION
        ].put(request.id, request.prefill_completed_at - request.scheduled_at)
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.prefill_completed_at - request.scheduled_at)
            / request.num_prefill_tokens,
        )
        self._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED
        ].put(
            request.id,
            (request.completed_at - request.prefill_completed_at)
            / request.num_decode_tokens,
        )
        #XXX for TPOT
        if True:
            import numpy as np
            #print('put TPOT', len(request.TPOT), request.TPOT)
            assert len(request.TPOT) == 0 or len(request.TPOT) == request._initial_num_decode_tokens - 1, f'{len(request.TPOT)} != {request._initial_num_decode_tokens} - 1'
            assert request.completed_at == request.latest_iteration_completed_at, f'{request.completed_at} != {request.latest_iteration_completed_at}'
            #assert sum(request.TPOT) == (request.completed_at - request.prefill_completed_at), f'{sum(request.TPOT)} != {request.completed_at - request.prefill_completed_at}'
            self._request_metrics_time_distributions[
                RequestMetricsTimeDistributions.TPOT
            ].put(
                request.id,
                request.TPOT
            )

        self._request_metrics_time_distributions[RequestMetricsTimeDistributions.ARRIVED_AT].put(request.id, request.arrived_at)

        self._request_metrics_histogram[
            RequestMetricsHistogram.REQUEST_NUM_RESTARTS
        ].put(request.id, request.num_restarts)

    def _update_per_token_execution_times(
        self, time: float, request: Request, batch: Batch
    ) -> None:
        # if prefill has just finished in this iteration, update the prefill completion time series
        if (
            time == request.prefill_completed_at
            and self._config.store_token_completion_metrics
        ):
            self._token_completion_metrics_time_series[
                TokenCompletionMetricsTimeSeries.PREFILL_COMPLETIONS
            ].put(
                time,
                request.num_prefill_tokens,
            )

        # determine if this was prefill or decode token
        if not request.has_started_decode:
            return

        if not self._config.store_token_completion_metrics:
            return

        self._token_metrics_time_distribution[
            TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME
        ].put(
            time - batch.scheduled_at + request.latest_iteration_scheduling_delay,
        )

        self._token_completion_metrics_time_series[
            TokenCompletionMetricsTimeSeries.DECODE_COMPLETIONS
        ].put(time, 1)

    def _push_metric(
        self, metric_name: OperationMetrics, batch_id: int, value: float
    ) -> None:
        if metric_name in OperationMetrics:
            self._operation_metrics[metric_name].put(value)
            self._operation_metrics_per_batch[metric_name].put(batch_id, value)
        elif metric_name in CpuOperationMetrics:
            self._cpu_operation_metrics[metric_name].put(value)
            self._cpu_operation_metrics_per_batch[metric_name].put(batch_id, value)
        elif metric_name in BatchMetricsTimeDistribution:
            self._batch_metrics_time_distribution[metric_name].put(value)
            self._batch_metrics_time_distribution_per_batch[metric_name].put(
                batch_id, value
            )
        elif metric_name in BatchMetricsCountDistribution:
            self._batch_metrics_count_distribution[metric_name].put(value)
            self._batch_metrics_count_distribution_per_batch[metric_name].put(
                batch_id, value
            )
        else:
            raise ValueError(f"Invalid metric name {metric_name}")

    @if_write_metrics
    def on_batch_end(
        self, time: float, batch: Batch, replica_id: int, memory_usage_percent: int, replica_scheduler
    ) -> None:
        if (
            self._config.min_batch_index and batch.id < self._config.min_batch_index
        ) or (self._config.max_batch_index and batch.id > self._config.max_batch_index):
            return

        for request in batch.completed_requests:
            self._on_request_end(time, request)

        if self._config.store_utilization_metrics:
            self._replica_memory_usage[replica_id - 1].put(time, memory_usage_percent)

        for request in batch.requests:
            self._update_per_token_execution_times(time, request, batch)

        if not self._config.store_batch_metrics:
            return

        self._push_metric(
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME,
            batch.id,
            time - batch.scheduled_at,
        )
        #print('BATCH', batch.id, 'EXECUTION: ', time - batch.scheduled_at)
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_TOKENS,
            batch.id,
            batch.total_num_tokens,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_TOKENS,
            batch.id,
            batch.num_prefill_tokens,
        )
        #XXX
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_TOKENS_SQUARE,
            batch.id,
            batch.num_prefill_tokens_square,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_TOKENS_DOT_KVS,
            batch.id,
            batch.num_prefill_tokens_dot_kvs,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_DECODE_TOKENS,
            batch.id,
            batch.num_decode_tokens,
        )
        #XXX
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_KVS,
            batch.id,
            batch.num_prefill_kvs,
        )
        #XXX
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_DECODE_KVS,
            batch.id,
            batch.num_decode_kvs,
        )
        #XXX
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_PREFILL_REQUESTS,
            batch.id,
            batch.num_prefill_requests,
        )
        #XXX
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_NUM_DECODE_REQUESTS,
            batch.id,
            batch.num_decode_requests,
        )
        self._push_metric(
            BatchMetricsCountDistribution.BATCH_SIZE, batch.id, batch.size
        )
        #XXX
        self._push_metric(
            BatchMetricsCountDistribution.RUNNING_SIZE, batch.id, replica_scheduler.num_running_requests + len(batch.completed_requests)
        )
        #XXX
        self._push_metric(
            BatchMetricsCountDistribution.MEMORY_USAGE, batch.id, memory_usage_percent
        )

        assert replica_scheduler.num_running_requests + len(batch.completed_requests) >= batch.size, f'{replica_scheduler.num_running_requests + len(batch.completed_requests)} >= {batch.size}'

        # running requests
        all_requests = replica_scheduler._preempted_requests + replica_scheduler._request_queue

        if self._config.store_schedule:
            self._vllm_schedule['c'][batch.id] = {}
            self._vllm_schedule['p'][batch.id] = {}
            self._vllm_schedule['e'][batch.id] = {}

            #print('storing', len(all_requests), 'vars for batch', batch.id)
            for request in all_requests:
                delta, gen, z, c = request.csp_delta_gen_z_c()

                self._vllm_schedule['c'][batch.id][request.id] = c
                self._vllm_schedule['e'][batch.id][request.id] = request.csp_e()
                self._vllm_schedule['p'][batch.id][request.id] = (1 - delta) * z

            for request in batch.requests:
                if request not in all_requests: #, f'{request} not in all requests'
                    assert request.completed, f'{request} not in all requests and not completed'

                    delta, gen, z, c = request.csp_delta_gen_z_c()

                    self._vllm_schedule['c'][batch.id][request.id] = c
                    self._vllm_schedule['e'][batch.id][request.id] = request.csp_e()
                    self._vllm_schedule['p'][batch.id][request.id] = (1 - delta) * z

        elif self._config.store_csp:
            self._replica_csp_vars[replica_id - 1]['m'][batch.id] = {}
            self._replica_csp_vars[replica_id - 1]['B'][batch.id] = {}
            self._replica_csp_vars[replica_id - 1]['e'][batch.id] = {}
            self._replica_csp_vars[replica_id - 1]['delta'][batch.id] = {}
            self._replica_csp_vars[replica_id - 1]['gen'][batch.id] = {}
            self._replica_csp_vars[replica_id - 1]['z'][batch.id] = {}
            self._replica_csp_vars[replica_id - 1]['c'][batch.id] = {}

            #print('storing', len(all_requests), 'vars for batch', batch.id)
            for request in all_requests:
                self._replica_csp_vars[replica_id - 1]['m'][batch.id][request.id] = request.csp_m
                self._replica_csp_vars[replica_id - 1]['B'][batch.id][request.id] = request.csp_B
                self._replica_csp_vars[replica_id - 1]['e'][batch.id][request.id] = request.csp_e()
                self._replica_csp_vars[replica_id - 1]['delta'][batch.id][request.id], self._replica_csp_vars[replica_id - 1]['gen'][batch.id][request.id], self._replica_csp_vars[replica_id - 1]['z'][batch.id][request.id], self._replica_csp_vars[replica_id - 1]['c'][batch.id][request.id] = request.csp_delta_gen_z_c()

                #XXX due to reserving ONE BLOCK for every decode, no_evict can evict
                #if self._simulation_config.cluster_config.replica_scheduler_config.no_evict:
                #    assert self._vllm_schedule['e'][batch.id][request.id] == 0

            for request in batch.requests:
                if request not in all_requests: #, f'{request} not in all requests'
                    assert request.completed, f'{request} not in all requests and not completed'
                    #print('completed', request.id, 'm', request.csp_m, 'B', request.csp_B, 'e', request.csp_e, 'delta', request.csp_delta)
                    #assert request.csp_m == 0, f'{request} csp_m {request.csp_m} != 0'
                    #assert request.csp_B == 0, f'{request} csp_B {request.csp_B} != 0'
                    #assert request.csp_e == 0, f'{request} csp_e {request.csp_e} != 0'
                    #assert request.csp_delta == 0, f'{request} csp_e {request.csp_e} != 0'
                    self._replica_csp_vars[replica_id - 1]['m'][batch.id][request.id] = request.csp_m
                    self._replica_csp_vars[replica_id - 1]['B'][batch.id][request.id] = request.csp_B
                    self._replica_csp_vars[replica_id - 1]['e'][batch.id][request.id] = request.csp_e()
                    self._replica_csp_vars[replica_id - 1]['delta'][batch.id][request.id], self._replica_csp_vars[replica_id - 1]['gen'][batch.id][request.id], self._replica_csp_vars[replica_id - 1]['z'][batch.id][request.id], self._replica_csp_vars[replica_id - 1]['c'][batch.id][request.id] = request.csp_delta_gen_z_c()
                    #sum_delta = sum([self._replica_csp_vars[replica_id - 1]['delta'][b][request.id] for b in range(batch.id+1)])
                    #print('batch.id', batch.id, batch)
                    #print('request.id', request.id, request)
                    #print('csp_vars', self._replica_csp_vars[replica_id - 1]['gen'])
                    #XXX: fixed during rebuttal, that for some b, request.id is not presented
                    sum_delta = sum([self._replica_csp_vars[replica_id - 1]['gen'][b][request.id] for b in range(batch.id+1) if request.id in self._replica_csp_vars[replica_id - 1]['gen'][b]])
                    #XXX -1 since the finished batch is not included
                    assert sum_delta == request._num_generated_tokens, f'{sum_delta} != {request._num_generated_tokens} for request {request} and delta {[self._replica_csp_vars[replica_id - 1]["delta"][b][request.id] for b in range(batch.id+1)]} gen {[self._replica_csp_vars[replica_id - 1]["gen"][b][request.id] for b in range(batch.id+1)]} z {[self._replica_csp_vars[replica_id - 1]["z"][b][request.id] for b in range(batch.id+1)]} c {[self._replica_csp_vars[replica_id - 1]["c"][b][request.id] for b in range(batch.id+1)]} m {[self._replica_csp_vars[replica_id - 1]["m"][b][request.id] for b in range(batch.id+1)]} B {[self._replica_csp_vars[replica_id - 1]["B"][b][request.id] for b in range(batch.id+1)]} e {[self._replica_csp_vars[replica_id - 1]["e"][b][request.id] for b in range(batch.id+1)]}'

                    #if self._simulation_config.cluster_config.replica_scheduler_config.no_evict:
                    #    assert self._vllm_schedule['e'][batch.id][request.id] == 0

    @if_write_metrics
    def on_replica_schedule(
        self, time: float, replica_id: int, memory_usage_percent: int
    ) -> None:
        if not self._config.store_utilization_metrics:
            return

        self._replica_memory_usage[replica_id - 1].put(time, memory_usage_percent)

    @if_write_metrics
    def on_replica_stage_schedule(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch_stage: BatchStage,
        execution_time: ExecutionTime,
    ) -> None:
        if not self._config.store_utilization_metrics:
            assert False
            return

        self._replica_busy_time[replica_id - 1][stage_id - 1].put(time, 100)
        mfu = self._mfu_calculator.get_mfu(batch_stage)
        self._replica_mfu[replica_id - 1][stage_id - 1].put(time, mfu)

        if not self._config.store_operation_metrics:
            return

        batch_id = batch_stage._batch_id
        for _ in range(execution_time.num_layers):
            self._push_metric(
                OperationMetrics.MLP_UP_PROJ,
                batch_id,
                execution_time.mlp_layer_up_proj_execution_time,
            )
            self._push_metric(
                OperationMetrics.MLP_ACTIVATION,
                batch_id,
                execution_time.mlp_layer_act_execution_time,
            )
            self._push_metric(
                OperationMetrics.MLP_DOWN_PROJ,
                batch_id,
                execution_time.mlp_layer_down_proj_execution_time,
            )
            self._push_metric(
                OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
                batch_id,
                execution_time.mlp_all_reduce_time,
            )
            self._push_metric(
                OperationMetrics.ATTN_PRE_PROJ,
                batch_id,
                execution_time.attention_pre_proj_time,
            )
            self._push_metric(
                OperationMetrics.ATTN_POST_PROJ,
                batch_id,
                execution_time.attention_post_proj_time,
            )
            self._push_metric(
                OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
                batch_id,
                execution_time.attention_all_reduce_time,
            )

            if execution_time.attention_prefill_execution_time != 0:
                self._push_metric(
                    OperationMetrics.ATTN_PREFILL,
                    batch_id,
                    execution_time.attention_prefill_execution_time,
                )

            if execution_time.attention_decode_execution_time != 0:
                self._push_metric(
                    OperationMetrics.ATTN_DECODE,
                    batch_id,
                    execution_time.attention_decode_execution_time,
                )
            self._push_metric(
                OperationMetrics.ATTN_KV_CACHE_SAVE,
                batch_id,
                execution_time.attention_kv_cache_save_execution_time,
            )
            self._push_metric(
                OperationMetrics.ATTN_ROPE,
                batch_id,
                execution_time.attention_rope_execution_time,
            )
            self._push_metric(
                OperationMetrics.ADD, batch_id, execution_time.add_time * 2
            )
            self._push_metric(
                OperationMetrics.INPUT_LAYERNORM,
                batch_id,
                execution_time.attn_norm_time,
            )
            self._push_metric(
                OperationMetrics.POST_ATTENTION_LAYERNORM,
                batch_id,
                execution_time.mlp_norm_time,
            )

        self._push_metric(
            OperationMetrics.PIPELINE_SEND_RECV,
            batch_id,
            execution_time.pipeline_parallel_communication_time,
        )
        #print('BATCH', batch_id, 'PIPELINE: ', execution_time.pipeline_parallel_communication_time)
        self._push_metric(
            CpuOperationMetrics.SCHEDULE, batch_id, execution_time.schedule_time
        )
        self._push_metric(
            CpuOperationMetrics.SAMPLER_E2E, batch_id, execution_time.sampler_e2e_time
        )
        self._push_metric(
            CpuOperationMetrics.PREPARE_INPUTS_E2E,
            batch_id,
            execution_time.prepare_inputs_e2e_time,
        )
        self._push_metric(
            CpuOperationMetrics.MODEL_EXECUTION_E2E,
            batch_id,
            execution_time.model_time_ms,
        )
        self._push_metric(
            CpuOperationMetrics.PROCESS_MODEL_OUTPUTS,
            batch_id,
            execution_time.process_model_outputs_time,
        )
        self._push_metric(
            CpuOperationMetrics.RAY_COMM_TIME, batch_id, execution_time.ray_comm_time
        )

    @if_write_metrics
    def on_batch_stage_end(
        self, batch_stage: BatchStage, time: float, replica_id: int, stage_id: int
    ) -> None:
        if not self._config.store_utilization_metrics:
            return
        self._replica_busy_time[replica_id - 1][stage_id - 1].put(time, 0)
        self._replica_mfu[replica_id - 1][stage_id - 1].put(time, 0)
