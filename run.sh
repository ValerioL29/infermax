PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
python vllm-scheduler/benchmarks/scheduler/simple.py \
    --model meta-llama/Meta-Llama-3-70B \
    --enforce-eager \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 1.0 \
    --num-gpu-blocks-override 6250 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 8192 \
    --trace-file vidur/data/processed_traces/splitwise_conv.csv \
    --use-srf-preemption \
    --output-file outputs/llama_3_70b_8192_16384_srf_vllm_re.pkl
