VLLM_USE_TRITON_FLASH_ATTN="False" \
VLLM_ATTENTION_BACKEND="XFORMERS" \
python3 run_vllm.py --vidur \
    --dtype half \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --model meta-llama/Meta-Llama-3-8B \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 1024 \
    --disable-async-output-proc \
    --schedule /home/jli/Workspaces/infermax/vidur/simulator_output/2025-06-06_16-47-58-823872/schedule.pkl