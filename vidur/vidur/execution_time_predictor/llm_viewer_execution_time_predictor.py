from typing import List, Tuple

from vidur.config import (
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    BaseExecutionTimePredictorConfig,
    ReplicaConfig,
)
from vidur.entities import Batch
from vidur.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)

import os
import sys

# Add LLM-Viewer to path for importing
LLM_VIEWER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "LLM-Viewer")
sys.path.append(os.path.abspath(LLM_VIEWER_DIR))
from model_analyzer import ModelAnalyzer


class LLMViewerExecutionTimePredictor(BaseExecutionTimePredictor):
    """Execution time predictor using LLM-Viewer."""

    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )
        model_id = self._model_config.get_name()
        hardware = self._replica_config.device.upper()
        self._analyzer = ModelAnalyzer(model_id, hardware)
        self._cache = {}

    def _predict_per_layer_time_ms(self, batch: Batch) -> float:
        if batch.id not in self._cache:
            token_counts: List[Tuple[int, int]] = [
                (t, r.num_processed_tokens)
                for r, t in zip(batch.requests, batch.num_tokens)
            ]
            result = self._analyzer.analyze(mode="hybrid", token_counts=token_counts)
            total_time_s = result["total_results"]["hybrid"]["inference_time"]
            total_time_ms = total_time_s * 1000
            per_stage_ms = total_time_ms / self._replica_config.num_pipeline_stages
            per_layer_ms = per_stage_ms / self._num_layers_per_pipeline_stage
            self._cache[batch.id] = per_layer_ms
        return self._cache[batch.id]

    # Map all model time to attention decode execution time
    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        return self._predict_per_layer_time_ms(batch)

    def _get_other_time(self, batch: Batch) -> float:
        return 0.0

    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        return 0.0

    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        return 0.0

    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        return 0.0

    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        return 0.0

    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        return 0.0

    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        return 0.0

    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        return 0.0

    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        return 0.0

    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        return 0.0

    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        return 0.0

    def _get_schedule_time(self, batch: Batch) -> float:
        return 0.0

    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        return 0.0

    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        return 0.0

    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        return 0.0

    def _get_ray_comm_time(self, batch: Batch) -> float:
        return 0.0

    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        return 0.0

    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        return 0.0

    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        return 0.0
