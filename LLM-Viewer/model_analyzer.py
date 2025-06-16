import os
import importlib
from hardwares.hardware_params import hardware_params
from roofline_model import roofline_analyze
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from utils import str_number, str_number_time
import math
import numpy as np
import matplotlib.pyplot as plt

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
]


class ModelAnalyzer:
    def __init__(self, model_id, hardware, config_file=None, source="huggingface"):
        """
        source: 'huggingface' or 'DiT'
        """
        self.model_id = model_id
        self.hardware = hardware
        if config_file is None:
            # get the current file directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # auto search the config
            for file in os.listdir(current_dir + "/configs"):
                if file.endswith(".py") and file.replace(".py", "") in model_id:
                    config_file = "configs/" + file
                # print(f"auto search config file {config_file} {file} {model_id}")
        assert config_file is not None, "config file is not found, please specify it manually."
        print(f"use config file {config_file} for {model_id}")
        if source == "huggingface":
            self.model_params = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        else:
            if not os.path.exists(f"model_params/{source}.py"):
                raise Exception(f"model_params/{source}.py is not found")
            # from model_params.DiT import model_params
            module = importlib.import_module(f"model_params.{source}")
            self.model_params = module.model_params[model_id]
        self.config = importlib.import_module(config_file.replace("/", ".").replace(".py", ""))

        # temporary variables
        self.results = None
        self.w_bit = None
        self.a_bit = None
        self.kv_bit = None
        self.batchsize = None
        self.seqlen = None

    def _analyze_to_results(
        self,
        stage,
        name,
        OPs=0,
        load_weight=0,
        load_act=0,
        store_act=0,
        load_kv_cache=0,
        store_kv_cache=0,
    ):

        bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
        memory_access = load_weight + load_act + store_act + load_kv_cache + store_kv_cache
        arithmetic_intensity, performance, bound = roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
        inference_time = OPs / performance
        self.results[stage][name] = {
            "OPs": OPs,
            "memory_access": memory_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": inference_time,
        }

    def save_csv(self, save_path=None):
        if save_path is None:
            save_path = f"output/{self.model_id[:self.model_id.rfind('/')]}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path += f"{self.model_id[self.model_id.rfind('/'):]}"

        decode_file_name = f"{save_path}_decode.csv"
        prefill_file_name = f"{save_path}_prefill.csv"
        print(f"save to {decode_file_name} and {prefill_file_name}")

        for file_name, stage in [
            (decode_file_name, "decode"),
            (prefill_file_name, "prefill"),
        ]:
            with open(file_name, "a+") as f:

                f.write(
                    f"\n\n=== {self.model_id} {self.hardware} w_bit={self.w_bit} a_bit={self.a_bit} kv_bit={self.kv_bit} batchsize={self.batchsize} seqlen={self.seqlen} tp_size={self.tp_size} ===\n"
                )
                # legend
                f.write(
                    f"layer_name,OPs,Access,arithmetic_intensity,performance,bound,load_weight,load_act,store_act,load_kv_cache,store_kv_cache,inference_time\n"
                )
            with open(file_name, "a+") as f:
                for layer_name, result in self.results[stage].items():
                    f.write(
                        f"{layer_name},{str_number(result['OPs'])},{str_number(result['memory_access'])}B,{str_number(result['arithmetic_intensity'])},{str_number(result['performance'])},"
                        f"{result['bound']},{str_number(result['load_weight'])}B,{str_number(result['load_act'])}B,{str_number(result['store_act'])}B,{str_number(result['load_kv_cache'])}B,"
                        f"{str_number(result['store_kv_cache'])}B,{str_number_time(result['inference_time'])}s\n"
                    )

    def analyze(
        self,
        seqlen=None,
        batchsize=None,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        kv_token_ratio=1,
        tp_size: int = 1,
        mode="standard",
        token_counts=None
    ):
        """
        mode: "standard" or "hybrid". 
            - If "standard", use seqlen and batchsize parameters
            - If "hybrid", use token_counts instead of seqlen and batchsize
        
        seqlen: sequence length (required for standard mode)
        batchsize: batch size (required for standard mode)
        token_counts: list of (c, m) pairs where c is the number of tokens to process and 
                     m is number of tokens processed (required for hybrid mode)
        
        w_bit: weight bit
        a_bit: activation bit
        kv_bit: key and value bit. if it is None, it will be the same as a_bit
        use_flashattention: use flash attention/flash decoding
        kv_token_ratio: use this for KV compression
        tp_size: the number of devices for tensor parallelism to use

        return is a dict with the following format:
        {
            "decode": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "prefill": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "hybrid": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "total_results": {
                "decode": {},
                "prefill": {},
                "hybrid": {}
            }
        }
        """
        if mode == "standard":
            assert seqlen is not None and seqlen > 0, "seqlen must be provided for standard mode"
            assert batchsize is not None and batchsize > 0, "batchsize must be provided for standard mode"
            self.results = {"decode": {}, "prefill": {}}
        else:  # hybrid mode
            assert token_counts is not None, "token_counts must be provided for hybrid mode"
            self.results = {"hybrid": {}}
            
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.tp_size = tp_size
        self.mode = mode

        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        config = self.config
        model_params = self.model_params
        num_attention_heads = config.get_num_attention_heads(model_params)
        hidden_size = config.get_hidden_size(model_params)
        num_key_value_heads = config.get_num_key_value_heads(model_params)
        num_hidden_layers = config.get_num_hidden_layers(model_params)

        # name = q_proj, ic = hidden_size, oc = hidden_size // tp_size
        for name, (ic, oc) in config.get_linear_layers(model_params, tp_size).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            
            if mode == "standard":
                self._analyze_to_results(
                    "decode",
                    name,
                    # for q_proj, it's 4096 * 4096 * 2 = 33.6M = OPs, this is just for one layer
                    OPs=ic * oc * batchsize * 2,
                    load_weight=ic * oc * w_byte,
                    load_act=ic * batchsize * a_byte,
                    store_act=0 if is_kv_proj else oc * batchsize * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
                )
                # for prefill
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=ic * oc * batchsize * seqlen * 2,
                    load_weight=ic * oc * w_byte,
                    load_act=ic * batchsize * seqlen * a_byte,
                    store_act=(0 if is_kv_proj else oc * batchsize * seqlen * a_byte),
                    load_kv_cache=0,
                    store_kv_cache=(0 if is_normal_proj else oc * batchsize * seqlen * kv_byte),
                )
            else:  # hybrid mode
                # Calculate totals for hybrid mode using token_counts
                total_tokens_to_process = sum(c for c, m in token_counts)
                
                self._analyze_to_results(
                    "hybrid",
                    name,
                    OPs=ic * oc * total_tokens_to_process * 2,
                    load_weight=ic * oc * w_byte,
                    load_act=ic * total_tokens_to_process * a_byte,
                    store_act=0 if is_kv_proj else oc * total_tokens_to_process * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=(0 if is_normal_proj else oc * total_tokens_to_process * kv_byte),
                )

        # for attention
        head_size = hidden_size // num_attention_heads
    
        if mode == "standard":

            # for decode
            qk_matmul_OPs = seqlen * head_size * num_attention_heads * batchsize * 2
            sv_matmul_OPs = 1 * head_size * seqlen * num_attention_heads * batchsize * 2
            # the softmax operation takes five steps:
            # max_x=max(x)
            # x=x-max_x
            # x_exp=exp(x)
            # sum_x_exp=sum(x_exp)
            # y=x_exp/sum(x_exp)
            softmax_OPs = batchsize * num_attention_heads * seqlen * 1 * 5

            if use_flashattention:
                name = f"fused_attention"
                bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
                # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
                block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
                n_blocks_r = math.ceil(1 / block_size_r)
                q_numel = (1) * head_size * batchsize * num_attention_heads * a_byte
                o_numel = 1 * seqlen * batchsize * num_attention_heads * a_byte
                self._analyze_to_results(
                    "decode",
                    name,
                    OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                    load_weight=0,
                    load_act=q_numel,
                    store_act=o_numel * 2,  # initialize O and save O
                    load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                    store_kv_cache=0,
                )

            else:
                name = f"qk_matmul"
                self._analyze_to_results(
                    "decode",
                    name,
                    OPs=qk_matmul_OPs,
                    load_weight=0,
                    load_act=(1) * head_size * batchsize * num_attention_heads * a_byte,
                    store_act=1 * seqlen * batchsize * num_attention_heads * a_byte,
                    load_kv_cache=(seqlen) * head_size * batchsize * num_key_value_heads * kv_byte,
                    store_kv_cache=0,
                )
                name = f"sv_matmul"
                self._analyze_to_results(
                    "decode",
                    name,
                    OPs=sv_matmul_OPs,
                    load_weight=0,
                    load_act=(1 * seqlen * batchsize * num_attention_heads) * a_byte,
                    store_act=1 * head_size * batchsize * num_attention_heads * a_byte,
                    load_kv_cache=(seqlen * head_size * batchsize * num_key_value_heads) * kv_byte,
                    store_kv_cache=0,
                )

                name = f"softmax"
                # max sub exp sum div
                self._analyze_to_results(
                    "decode",
                    name,
                    OPs=softmax_OPs,
                    load_weight=0,
                    load_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                    store_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )

            for name in config.get_norm_layers(model_params):
                # sum sub pow sum div mul add
                self._analyze_to_results(
                    "decode",
                    name,
                    OPs=batchsize * hidden_size * 1 * 7,
                    load_weight=0,
                    load_act=batchsize * hidden_size * 1 * a_byte,
                    store_act=batchsize * hidden_size * 1 * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )

            for name in ["attn_add", "mlp_add"]:
                self._analyze_to_results(
                    "decode",
                    name,
                    OPs=batchsize * hidden_size * 1,
                    load_weight=0,
                    load_act=batchsize * hidden_size * 1 * a_byte,
                    store_act=batchsize * hidden_size * 1 * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
            for name in ["mlp_act"]:
                self._analyze_to_results(
                    "decode",
                    name,
                    OPs=batchsize * hidden_size * 1 * 2,
                    load_weight=0,
                    load_act=batchsize * hidden_size * 1 * a_byte * 2,
                    store_act=batchsize * hidden_size * 1 * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )

            # for prefill
            qk_matmul_OPs = seqlen * seqlen * head_size * num_attention_heads * batchsize * 2
            sv_matmul_OPs = seqlen * head_size * seqlen * num_attention_heads * batchsize * 2
            softmax_OPs = batchsize * num_attention_heads * seqlen * seqlen * 5
            if use_flashattention:
                name = f"fused_attention"
                bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
                # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
                block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
                n_blocks_r = math.ceil(seqlen / block_size_r)
                q_numel = seqlen * head_size * batchsize * num_attention_heads * a_byte
                o_numel = seqlen * seqlen * batchsize * num_attention_heads * a_byte
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                    load_weight=0,
                    load_act=q_numel,
                    store_act=o_numel * 2,  # initialize O and save O
                    load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                    store_kv_cache=0,
                )
            else:
                name = f"qk_matmul"
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=qk_matmul_OPs,
                    load_weight=0,
                    load_act=seqlen * head_size * batchsize * num_key_value_heads * a_byte,
                    store_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                    load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                    store_kv_cache=0,
                )
                name = f"sv_matmul"
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=sv_matmul_OPs,
                    load_weight=0,
                    load_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                    store_act=seqlen * head_size * batchsize * num_attention_heads * a_byte,
                    load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                    store_kv_cache=0,
                )
                name = f"softmax"
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=softmax_OPs,
                    load_weight=0,
                    load_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                    store_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
            for name in config.get_norm_layers(model_params):
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=batchsize * hidden_size * seqlen * 7,
                    load_weight=0,
                    load_act=batchsize * hidden_size * seqlen * a_byte,
                    store_act=batchsize * hidden_size * seqlen * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
            for name in ["attn_add", "mlp_add"]:
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=batchsize * hidden_size * seqlen * 1,
                    load_weight=0,
                    load_act=batchsize * hidden_size * seqlen * a_byte,
                    store_act=batchsize * hidden_size * seqlen * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
            for name in ["mlp_act"]:
                self._analyze_to_results(
                    "prefill",
                    name,
                    OPs=batchsize * hidden_size * seqlen * 1 * 2,
                    load_weight=0,
                    load_act=batchsize * hidden_size * seqlen * a_byte * 2,
                    store_act=batchsize * hidden_size * seqlen * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
        else:  # hybrid mode
            qk_matmul_OPs = sum(c * (c + m) for c, m in token_counts) * head_size * num_attention_heads * 2
            sv_matmul_OPs = sum(c * (c + m) for c, m in token_counts) * head_size * num_attention_heads * 2
            softmax_OPs = total_tokens_to_process * num_attention_heads * 1 * 5
            if use_flashattention:
                name = f"fused_attention"
                bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
                # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
                block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
                #n_blocks_r = math.ceil(seqlen / block_size_r)
                q_numel = total_tokens_to_process * head_size * num_attention_heads * a_byte
                o_numel = sum(c * (c + m) for c, m in token_counts) * num_attention_heads * a_byte
                self._analyze_to_results(
                    "hybrid",
                    name,
                    OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                    load_weight=0,
                    load_act=q_numel,
                    store_act=o_numel * 2,  # initialize O and save O
                    load_kv_cache=sum(math.ceil(c / block_size_r) * (c + m) for c, m in token_counts) * head_size * num_key_value_heads * kv_byte * 2,
                    store_kv_cache=0,
                )
            else:
                assert False, "flash attention is not supported for hybrid mode"

            for name in config.get_norm_layers(model_params):
                self._analyze_to_results(
                    "hybrid",
                    name,
                    OPs=total_tokens_to_process * hidden_size * 7,
                    load_weight=0,
                    load_act=total_tokens_to_process * hidden_size * a_byte,
                    store_act=total_tokens_to_process * hidden_size * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
            for name in ["attn_add", "mlp_add"]:
                self._analyze_to_results(
                    "hybrid",
                    name,
                    OPs=total_tokens_to_process * hidden_size * 1,
                    load_weight=0,
                    load_act=total_tokens_to_process * hidden_size * a_byte,
                    store_act=total_tokens_to_process * hidden_size * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )
            for name in ["mlp_act"]:
                self._analyze_to_results(
                    "hybrid",
                    name,
                    OPs=total_tokens_to_process * hidden_size * 1 * 2,
                    load_weight=0,
                    load_act=total_tokens_to_process * hidden_size * a_byte * 2,
                    store_act=total_tokens_to_process * hidden_size * a_byte,
                    load_kv_cache=0,
                    store_kv_cache=0,
                )

        # compute total
        if mode == "standard":
            total_results = {"decode": {}, "prefill": {}}
            for data_name in ALL_DATA_NAMES:
                total_results["decode"][data_name] = 0
                total_results["prefill"][data_name] = 0
        else:
            total_results = {"hybrid": {}}
            for data_name in ALL_DATA_NAMES:
                total_results["hybrid"][data_name] = 0
            
        for stage in total_results.keys():
            for layer_name, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += result[data_name] * num_hidden_layers

        # memory footprint
        weight_kv_footprint = total_results["prefill"]["load_weight"] + total_results["prefill"]["store_kv_cache"]
        decode_tmp_act = 0
        for layer_name, result in self.results["decode"].items():
            decode_tmp_act += result["store_act"]
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        prefill_tmp_act = 0
        for layer_name, result in self.results["prefill"].items():
            prefill_tmp_act += result["store_act"]
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]

        # lm_head
        name = "lm_head"
        args = {"batchsize": batchsize, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.config.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[layer_info["stage"]][layer_info["name"]][
                    data_name
                ]
        # for stage in ["prefill", "decode"]:
        #     self._analyze_to_results(
        #         stage,
        #         name,
        #         OPs=batchsize * hidden_size * vocab_size * 1,
        #         load_weight=hidden_size * vocab_size,
        #         load_act=hidden_size * a_byte,
        #         store_act=vocab_size * a_byte,
        #         load_kv_cache=0,
        #         store_kv_cache=0,
        #     )
        #     for data_name in ALL_DATA_NAMES:
        #         total_results[stage][data_name] += self.results[stage][name][data_name]

        self.results["total_results"] = total_results
        return self.results

    def analyze_generate_task(
        self,
        prompt_len,
        gen_len,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention = False,
        tp_size: int = 1
    ):
        prefill_result = self.analyze(
            prompt_len,
            batchsize,
            w_bit,
            a_bit,
            kv_bit,
            use_flashattention=use_flashattention,
            tp_size=tp_size
        )
        prefill_time = inference_time = prefill_result["total_results"]["prefill"]["inference_time"]

        for i in range(prompt_len, prompt_len + gen_len):
            result = self.analyze(i, batchsize, w_bit, a_bit, kv_bit, use_flashattention=use_flashattention, tp_size=tp_size)
            inference_time += result["total_results"]["decode"]["inference_time"]
        return {"inference_time": inference_time, "prefill_time": prefill_time}

    def get_hardware_info(self):
        bandwidth = hardware_params[self.hardware]["bandwidth"]
        if self.w_bit <= 8 and self.a_bit <= 8 and self.kv_bit <= 8:
            max_OPS = hardware_params[self.hardware]["INT8"]
        else:
            max_OPS = hardware_params[self.hardware]["FP16"]
        onchip_buffer = hardware_params[self.hardware]["onchip_buffer"]
        return bandwidth, max_OPS, onchip_buffer

    def get_model_info(self):
        if self.config.get_num_attention_heads(self.model_params) != self.config.get_num_key_value_heads(
            self.model_params
        ):
            GQA = True
        else:
            GQA = False

        info = {"GQA": GQA}  # group query attention
        return info

    def find_turning_point_token_counts(self, w_bit=16, a_bit=16, kv_bit=None, use_flashattention=True, tp_size=1, num_requests=10):
        """
        Find token counts for multiple requests that together achieve arithmetic intensity at the turning point.
        Returns a list of (c, m) pairs where c is number of tokens to process and m is number of tokens processed.
        
        Args:
            num_requests: Number of requests to generate token counts for
        """
        if kv_bit is None:
            kv_bit = a_bit
            
        bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
        turning_point = max_OPS / bandwidth
        
        config = self.config
        model_params = self.model_params
        num_attention_heads = config.get_num_attention_heads(model_params)
        hidden_size = config.get_hidden_size(model_params)
        head_size = hidden_size // num_attention_heads
        
        w_byte = w_bit / 8
        a_byte = a_bit / 8
        kv_byte = kv_bit / 8
        
        # For multiple requests, we want:
        # sum(c_i * (c_i + m_i)) * head_size * num_attention_heads * 2 / 
        # (sum(c_i) * (hidden_size*a_byte + head_size*num_attention_heads*kv_byte)) = turning_point
        
        # Let's generate token counts that follow a pattern:
        # For each request i:
        # c_i = base_c * (1 + i * step)
        # m_i = base_m * i
        
        # First, calculate base values that would achieve turning point for a single request
        base_c = turning_point * (hidden_size*a_byte + head_size*num_attention_heads*kv_byte) / (head_size * num_attention_heads)
        base_c = int(base_c)  # Round to integer
        
        # Now generate token counts for multiple requests
        token_counts = []
        step = 0.1  # Step size for increasing c_i
        base_m = base_c // 2  # Base value for m_i
        
        for i in range(num_requests):
            c = int(base_c * (1 + i * step))
            m = int(base_m * i)
            token_counts.append((c, m))
            
        return token_counts

    def calculate_arithmetic_intensity(self, token_counts, w_bit=16, a_bit=16, kv_bit=None, tp_size=1):
        """
        Calculate arithmetic intensity for given token counts.
        
        Args:
            token_counts: List of (c, m) pairs where c is number of tokens to process and m is number of tokens processed
            w_bit: weight bitwidth
            a_bit: activation bitwidth
            kv_bit: kv cache bitwidth
            
        Returns:
            arithmetic_intensity: OPs per byte
        """
        if kv_bit is None:
            kv_bit = a_bit
            
        w_byte = w_bit / 8
        a_byte = a_bit / 8
        kv_byte = kv_bit / 8
        
        config = self.config
        model_params = self.model_params
        num_attention_heads = config.get_num_attention_heads(model_params)
        hidden_size = config.get_hidden_size(model_params)
        head_size = hidden_size // num_attention_heads
        num_key_value_heads = config.get_num_key_value_heads(model_params)
        
        # Calculate block size for flash attention
        bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
        block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
        print(f"block_size_r: {block_size_r}")
        
        # Sum of tokens to process
        sum_c = sum(c for c, m in token_counts)
        # Sum of c * (c + m) for attention computations
        sum_c_cm = sum(c * (c + m) for c, m in token_counts)
        # Sum of ceil(c/block_size_r) * c for kv cache
        sum_block_c = sum(math.ceil(c / block_size_r) * (c + m) for c, m in token_counts)
        
        # Calculate total OPs
        total_OPs = 0
        total_memory_access = 0

        OP_sum_c_constant = 0
        OP_sum_c_cm_constant = 0
        OP_sum_block_c_constant = 0
        OP_1_constant = 0

        memory_sum_c_constant = 0
        memory_sum_c_cm_constant = 0
        memory_sum_block_c_constant = 0
        memory_1_constant = 0

        for name, (ic, oc) in config.get_linear_layers(model_params, tp_size).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj

            total_OPs += ic * oc * sum_c * 2  
            OP_sum_c_constant += ic * oc * 2

            total_memory_access += (
                ic * oc * w_byte + 
                ic * sum_c * a_byte + 
                (0 if is_kv_proj else oc * sum_c * a_byte) + 
                (0 if is_normal_proj else oc * sum_c * kv_byte)
            )
            memory_sum_c_constant += ic * a_byte + (0 if is_kv_proj else oc * a_byte) + (0 if is_normal_proj else oc * a_byte)
            memory_1_constant += ic * oc * w_byte
        
        # attention
        total_OPs += (
            sum_c_cm * head_size * num_attention_heads * 4 +  # qk and sv matmuls
            sum_c * num_attention_heads * 5 # softmax
        )
        OP_sum_c_constant += num_attention_heads * 5
        OP_sum_c_cm_constant += head_size * num_attention_heads * 4

        total_memory_access += (
            sum_c * head_size * num_attention_heads * a_byte +  # q loading
            sum_c_cm * num_attention_heads * a_byte * 2 +  # o loading and storing
            sum_block_c * head_size * num_key_value_heads * kv_byte * 2 # kv cache
        )
        memory_sum_c_constant += head_size * num_attention_heads * a_byte
        memory_sum_c_cm_constant += num_attention_heads * a_byte * 2
        #memory_sum_block_c_constant += head_size * num_key_value_heads * kv_byte * 2
        memory_sum_c_cm_constant += head_size * num_key_value_heads * kv_byte * 2 / block_size_r

        for name in config.get_norm_layers(model_params):
            total_OPs += sum_c * hidden_size * 7
            OP_sum_c_constant += hidden_size * 7
            
            total_memory_access += sum_c * hidden_size * a_byte * 2
            memory_sum_c_constant += hidden_size * a_byte * 2

        for name in ["attn_add", "mlp_add"]:
            total_OPs += sum_c * hidden_size * 1
            OP_sum_c_constant += hidden_size * 1

            total_memory_access += sum_c * hidden_size * a_byte * 2
            memory_sum_c_constant += hidden_size * a_byte * 2
        
        for name in ["mlp_act"]:
            total_OPs += sum_c * hidden_size * 1 * 2
            OP_sum_c_constant += hidden_size * 1 * 2

            total_memory_access += sum_c * hidden_size * a_byte * 3   
            memory_sum_c_constant += hidden_size * a_byte * 3

        OP_sum_c_constant /= max_OPS
        OP_sum_c_cm_constant /= max_OPS
        OP_sum_block_c_constant /= max_OPS
        OP_1_constant /= max_OPS
        memory_sum_c_constant /= bandwidth
        memory_sum_c_cm_constant /= bandwidth
        memory_sum_block_c_constant /= bandwidth
        memory_1_constant /= bandwidth

        print('sum_c_constant', OP_sum_c_constant, memory_sum_c_constant, OP_sum_c_constant - memory_sum_c_constant)
        print('sum_cm_constant', OP_sum_c_cm_constant, memory_sum_c_cm_constant, OP_sum_c_cm_constant - memory_sum_c_cm_constant)
        print('sum_block_c_constant', OP_sum_block_c_constant, memory_sum_block_c_constant, OP_sum_block_c_constant - memory_sum_block_c_constant)
        print('1_constant', OP_1_constant, memory_1_constant, OP_1_constant - memory_1_constant)
        
        # total_OPS
        # sum_c: sum([(ic * oc * 2) for name, (ic, oc) in config.get_linear_layers(model_params, tp_size).items()])
        # + (num_attention_heads * 5) 
        # + (hidden_size * 7 * len(config.get_norm_layers(model_params))) 
        # + (hidden_size * 1 * 4)
        # sum_c_cm: head_size * num_attention_heads * 4
        # 1: 
        
        # total_memory_access
        # sum_c: sum([ic * a_byte for name, (ic, oc) in config.get_linear_layers(model_params, tp_size).items()])
        # + (head_size * num_attention_heads * a_byte)
        # + (hidden_size * a_byte * 2 * len(config.get_norm_layers(model_params))
        # + (hidden_size * a_byte * 7)
        # 
        
        return total_OPs / total_memory_access

    def generate_random_token_counts(self, num_requests=10, max_tokens=1000):
        """
        Generate random token counts for multiple requests.
        
        Args:
            num_requests: Number of requests to generate
            max_tokens: Maximum number of tokens per request
            
        Returns:
            List of (c, m) pairs where:
            - c is number of tokens to process
            - m is number of tokens already processed
        """
        token_counts = []
        for _ in range(num_requests):
            c = np.random.randint(1, max_tokens)  # Random number of tokens to process
            m = np.random.randint(0, c)           # Random number of tokens already processed
            token_counts.append((c, m))
        return token_counts

    def plot_turning_point_curve(self):
        """
        Plot the curve of (c, m) points that satisfy the turning point equation.
        """
        # Generate c values
        c = np.linspace(1, 1000, 1000)

        plt.figure(figsize=(10, 6))

        for B in [1, 32, 1024]:
            # Calculate m values using the equation
            m = (1.1338710947316351e-06 * c - 1.1211740456756533e-10 * c**2 - 0.00026028960514469453 / B) / (1.1211740456756533e-10 * c)
            
            # Filter out invalid points (m < 0 or m > c)
            valid_mask = m >= 0 & (m <= 100000) #(m >= 0) & (m <= c)
            c_valid = c[valid_mask]
            m_valid = m[valid_mask]
            
            plt.plot(c_valid, m_valid, label=f'Turning Point Curve for B={B}')
            #plt.fill_between(c_valid, 0, m_valid, alpha=0.2, label='Valid Region')
        
        # Add labels and title
        plt.xlabel('Number of tokens to process (c)')
        plt.ylabel('Number of tokens processed (m)')
        plt.title('Token Counts that Achieve Turning Point Arithmetic Intensity')
        plt.grid(True)
        plt.legend()
        
        # Show the plot
        plt.show()
