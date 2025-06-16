import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from hardwares.hardware_params import hardware_params
from model_analyzer import ModelAnalyzer

# Initialize model analyzer
model_id = "meta-llama/Llama-2-7b-hf"
hardware = "nvidia_A100"
analyzer = ModelAnalyzer(model_id, hardware)

def plot_operator_breakdown(seqlen, batchsize):
    """Plot operator breakdown for a given sequence length and batch size."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    def get_operator_breakdown(configs, use_ratio=True):
        all_operators = set()
        config_data = []
        
        for config in configs:
            result = analyzer.analyze(
                config["seqlen"], 
                config["batchsize"], 
                w_bit=16, 
                a_bit=16, 
                kv_bit=16,
                use_flashattention=True
            )
            phase_data = result[config["phase"]]
            total_time = sum(data["inference_time"] for data in phase_data.values())
            
            operator_times = {}
            kv_time = 0
            ffn_time = 0
            others_time = 0
            
            for op, data in phase_data.items():
                if op == "lm_head":
                    continue
                elif op.endswith(("_norm", "_act", "_add")):
                    others_time += data["inference_time"]
                elif op in ["k_proj", "v_proj"]:
                    kv_time += data["inference_time"]
                elif op in ["out_proj", "gate_proj", "up_proj", "down_proj"]:
                    ffn_time += data["inference_time"]
                elif op == "q_proj":
                    operator_times["q_proj"] = (data["inference_time"] / total_time) * 100 if use_ratio else data["inference_time"]
                elif op == "fused_attention":
                    operator_times["fused_attention"] = (data["inference_time"] / total_time) * 100 if use_ratio else data["inference_time"]
            
            if kv_time > 0:
                operator_times["kv_proj"] = (kv_time / total_time) * 100 if use_ratio else kv_time
            if ffn_time > 0:
                operator_times["FFN"] = (ffn_time / total_time) * 100 if use_ratio else ffn_time
            if others_time > 0:
                operator_times["others"] = (others_time / total_time) * 100 if use_ratio else others_time
            
            all_operators.update(operator_times.keys())
            config_data.append(operator_times)
        
        return all_operators, config_data

    def plot_stacked_bar(ax, configs, title, x_axis_label):
        all_operators, config_data = get_operator_breakdown(configs)
        operators = sorted(list(all_operators))
        x_labels = [str(c[x_axis_label]) for c in configs]
        
        # Prepare data for each operator
        data_matrix = []
        for operator in operators:
            data_matrix.append([d.get(operator, 0) for d in config_data])
        data_matrix = np.array(data_matrix)
        
        # Plot
        bottom = np.zeros(len(configs))
        for i, operator in enumerate(operators):
            ax.bar(x_labels, data_matrix[i], bottom=bottom, label=operator)
            bottom += data_matrix[i]
        
        ax.set_ylabel('Percentage of Total Inference Time (%)')
        ax.set_xlabel(x_axis_label.capitalize())
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Calculate sequence lengths and batch sizes that multiply to 4096
    total_size = 4096
    seqlens = [32, 64, 128, 256, 512, 1024, 2048]
    batchsizes = [total_size // s for s in seqlens]

    # Plot decode phase
    decode_configs = [{"seqlen": s, "batchsize": b, "phase": "decode"} 
                     for s, b in zip(seqlens, batchsizes)]
    plot_stacked_bar(ax1, decode_configs, 
                    "Decode Phase - Fixed Total Size (seqlen × batchsize = 4096)", 
                    "seqlen")

    # Plot prefill phase
    prefill_configs = [{"seqlen": s, "batchsize": b, "phase": "prefill"} 
                      for s, b in zip(seqlens, batchsizes)]
    plot_stacked_bar(ax2, prefill_configs, 
                    "Prefill Phase - Fixed Total Size (seqlen × batchsize = 4096)", 
                    "seqlen")

    plt.tight_layout()
    return fig

# Create and show the plot
fig = plot_operator_breakdown(seqlen=1024, batchsize=4)  # Example values
plt.savefig("../output/operator_breakdown_4096.pdf", bbox_inches='tight')
plt.show() 