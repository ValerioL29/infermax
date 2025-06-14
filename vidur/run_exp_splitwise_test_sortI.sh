max_B=1000000
bsize=1

C=16384

# XXX missing ones

M=100000

mode=${1:-single}
if [ "$mode" = "dual" ]; then
    ds_flag="--test_scheduler_config_dual_schedulers"
else
    ds_flag=""
fi

python -m vidur.main --replica_config_model_name meta-llama/Meta-Llama-3-70B \
    --replica_config_num_pipeline_stages 1 \
    --replica_config_tensor_parallel_size 4 \
    --replica_scheduler_config_type test \
    --test_scheduler_config_block_size $bsize \
    --test_scheduler_config_batch_size_cap $max_B \
    --test_scheduler_config_num_blocks $M \
    --test_scheduler_config_chunk_size $C \
    --test_scheduler_config_sortI \
    --request_generator_config_type synthetic \
    --synthetic_request_generator_config_num_requests 0 \
    --interval_generator_config_type trace \
    --length_generator_config_type trace \
    --trace_request_length_generator_config_trace_file data/processed_traces/splitwise_conv.csv \
    --interval_generator_config_type trace \
    --trace_request_interval_generator_config_trace_file data/processed_traces/splitwise_conv.csv \
    --trace_request_interval_generator_config_start_time 0.0 \
    --trace_request_interval_generator_config_end_time 3600.0 \
    --execution_time_predictor_config_type linear_regression \
    --linear_regression_execution_time_predictor_config_kv_cache_prediction_granularity 1 \
    --linear_regression_execution_time_predictor_config_prediction_max_prefill_chunk_size $C \
    --linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request $C \
    --test_scheduler_config_watermark_blocks_fraction 0 \
    --no-metrics_config_write_json_trace \
    --metrics_config_store_operation_metrics \
    --no-metrics_config_store_token_completion_metrics \
    --metrics_config_keep_individual_batch_metrics \
    --no-metrics_config_store_plots \
    --no-metrics_config_enable_chrome_trace \
    --metrics_config_store_schedule \
    --metrics_config_save_table_to_wandb \
    $ds_flag
