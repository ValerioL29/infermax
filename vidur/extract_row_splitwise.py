import json
import pandas as pd
import numpy as np
import os
from hashlib import md5

def extract_config_values(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)

    replica_config   = config['cluster_config']['replica_config']
    scheduler_config = config['cluster_config']['replica_scheduler_config']
    generator_config = config['request_generator_config']
    length_generator_config = generator_config['length_generator_config']
    execution_config = config['execution_time_predictor_config']

    name = scheduler_config['name']
    if name == 'sarathi' and scheduler_config['chunk_size'] == execution_config['prediction_max_tokens_per_request']:
        if scheduler_config['no_hybrid_batch']:
            name = 'sarathi_nohy'
        elif scheduler_config['no_chunked_prefill']:
            name = 'sarathi_nocp'
        else:
            name = 'sarathi_unlimitedP'
    elif name == 'vllm':
        if scheduler_config['hybrid_batch']:
            name = 'vllm_hy'

    if scheduler_config['no_evict']:
        name += '_noe'

    if 'sortI' in scheduler_config and scheduler_config['sortI']:
        name += '_sortI'
    
    if 'histogram' in scheduler_config and scheduler_config['histogram']:
        name += '_hist'

    return {
        #'I': length_generator_config['prefill_tokens'],
        #'O': length_generator_config['decode_tokens'],
        #'B': generator_config['num_requests'],
        'C': execution_config['prediction_max_tokens_per_request'], #scheduler_config['max_tokens_in_batch'],
        'M': scheduler_config['num_blocks'],
        'max_B': scheduler_config['batch_size_cap'],
        'scheduler': name,
        'page_size': scheduler_config['block_size'],
        'model': replica_config['model_name'],
        'TP': replica_config['tensor_parallel_size'],
        'PP': replica_config['num_pipeline_stages'],
    }

def compute_metrics(base_path):
    # Read request_metrics.csv
    request_df = pd.read_csv(f"{base_path}/request_metrics.csv")

    avg_TTFT = request_df['prefill_e2e_time'].mean()

    # Compute avg_TPOT with zero handling
    request_df['TPOT'] = np.where(
        request_df['request_num_decode_tokens'] > 1,
        #request_df['decode_time_execution_plus_preemption_normalized'] * request_df['request_num_decode_tokens'] / (request_df['request_num_decode_tokens'] - 1),
        request_df['decode_time_execution_plus_preemption_normalized'] * request_df['request_num_decode_tokens'],
        0
    )
    #avg_TPOT = request_df['TPOT'].mean()
    avg_TPOT = 0 if sum(request_df['TPOT']) == 0 else sum(request_df['TPOT']) / (sum(request_df['request_num_decode_tokens']) - len(request_df))

    num_evicted = request_df['request_num_restarts'].sum()

    # Read batch_metrics.csv
    batch_df = pd.read_csv(f"{base_path}/batch_metrics.csv")

    #op_df = pd.read_csv(f"{base_path}/operation_metrics.csv")

    latency = batch_df['batch_execution_time'].sum()
    total_decode_tokens = request_df['request_num_decode_tokens'].sum()
    TPS = total_decode_tokens / latency
    avg_batch_size = batch_df['batch_size'].mean()
    avg_running_size = batch_df['running_size'].mean()
    avg_memory_usage = batch_df['memory_usage'].mean()

    assert avg_running_size >= avg_batch_size, f'avg_running_size: {avg_running_size}, avg_batch_size: {avg_batch_size}'

    num_tokens = batch_df['batch_num_tokens'].sum()

    num_prefill_tokens =  batch_df['batch_num_prefill_tokens'].sum()
    num_prefill_tokens_square = batch_df['batch_num_prefill_tokens_square'].sum()
    num_prefill_kvs = batch_df['batch_num_prefill_kvs'].sum()
    num_prefill_requests = batch_df['batch_num_prefill_requests'].sum()
    num_prefill_batches = len(batch_df[batch_df['batch_num_prefill_requests'] > 0])

    num_decode_tokens =  batch_df['batch_num_decode_tokens'].sum()
    num_decode_kvs = batch_df['batch_num_decode_kvs'].sum()
    num_decode_requests = batch_df['batch_num_decode_requests'].sum()
    num_decode_batches = len(batch_df[batch_df['batch_num_decode_requests'] > 0])

    num_batches = len(batch_df)

    return {
        'latency': latency,
        'TPS': TPS,
        'avg_TTFT': avg_TTFT,
        'avg_TPOT': avg_TPOT,
        'evict_m': num_evicted,
        'batch': len(batch_df),
        'avg_batch_size': avg_batch_size,
        'avg_running_size': avg_running_size,
        'avg_memory_usage': avg_memory_usage,

        'num_tokens': num_tokens,
        'num_prefill_tokens': num_prefill_tokens,
        'num_prefill_tokens_square': num_prefill_tokens_square,
        'num_prefill_kvs': num_prefill_kvs,
        'num_prefill_requests': num_prefill_requests,
        'num_prefill_batches': num_prefill_batches,

        'num_decode_tokens': num_decode_tokens,
        'num_decode_kvs': num_decode_kvs,
        'num_decode_requests': num_decode_requests,
        'num_decode_batches': num_decode_batches,

        'num_batches': num_batches,
    }

def process_directory(directory):
    config_values = extract_config_values(os.path.join(directory, "config.json"))
    metrics = compute_metrics(directory)
    return {**config_values, **metrics, 'directory': os.path.basename(directory)}

def dict_hash(dictionary):
    """MD5 hash of a dictionary."""
    dhash = md5()
    # Create a copy of the dictionary without the 'directory' key
    hash_dict = {k: v for k, v in dictionary.items() if k in ['I', 'O', 'B', 'C', 'M', 'max_B', 'scheduler', 'page_size', 'model', 'TP', 'PP']}
    encoded = json.dumps(hash_dict, sort_keys=True, default=convert_to_serializable).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def convert_to_serializable(obj):
    """Convert numpy and pandas types to standard Python types."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Main execution
simulator_output_dir = 'simulator_output'
subdirs = []

# Get all subdirectories with their modification times
for subdir in os.listdir(simulator_output_dir):
    full_path = os.path.join(simulator_output_dir, subdir)
    if os.path.isdir(full_path):
        mod_time = os.path.getmtime(full_path)
        subdirs.append((subdir, mod_time))

# Sort subdirectories by modification time, most recent first
subdirs.sort(key=lambda x: x[1], reverse=True)

results = {}

for subdir, _ in subdirs:
    full_path = os.path.join(simulator_output_dir, subdir)
    try:
        result = process_directory(full_path)
        # Convert result to serializable types before hashing
        serializable_result = json.loads(json.dumps(result, default=convert_to_serializable))
        result_hash = dict_hash(serializable_result)
        if result_hash not in results:
            results[result_hash] = result
            print(f"Processed directory: {subdir}")
        else:
            print(f"Duplicate result found for directory {subdir}, skipping.")
    except Exception as e:
        print(f"Error processing directory {subdir}: {str(e)}")

# Create DataFrame from the dictionary values
df = pd.DataFrame(results.values())

# Sort the DataFrame by directory name
df = df.sort_values('directory')

# Display the DataFrame
print(df)

# Optionally, save to CSV
df.to_csv('simulation_results.csv', index=False)
