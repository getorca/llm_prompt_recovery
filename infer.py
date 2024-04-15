import random
from jinja2 import Environment, Template, meta
import sys
from itertools import product
from datasets import load_dataset
import polars as pl
from vllm import LLM, SamplingParams
from typing import Dict
import numpy as np
import ray
from transformers import AutoTokenizer
from typing import Any, Dict
from vllm.lora.request import LoRARequest


# model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
model_name = '/gpfs/home/yiyayu/scratch/cache/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873'
# model_name = '/home/lawrence/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873'
# # model_name = 'google/gemma-7b-it'

tokenizer = AutoTokenizer.from_pretrained(
    model_name
) 


def get_chat_input(row: Dict[str, Any]) -> Dict[str, Any]:
    
    #original text prefix
    orig_prefix = "Original Text:"
    #modified text prefix
    rewrite_prefix = "Rewritten Text:"
    # response start
    response_start = "The prompt was:"

    sys_prompt = """You are an expert in "Reverse Prompt Engineering". You are able to reverse-engineer prompts used to rewrite text.\n\nI will be providing you with an "original text" and "rewritten text". Please try to be as specific as possible and come up with a prompt that is based on the tone, style, and any other properties you consider relevant."""

    messages = [
        #actual prompt
        {"role": "user", "content": f"{sys_prompt}\n{orig_prefix} {row['original_text']}\n{rewrite_prefix} {row['rewritten_text']}"},
        {"role": "assistant", "content": response_start},
    ]
        
    #give it to Mistral
    row['input'] = tokenizer.apply_chat_template(messages, tokenize=False)
    
    return row


import pandas as pd

df = pd.read_csv('data/predictions/llm_dataset_1.csv') #2_above_65
df.rename(columns={'rewrite_prompt': 'old_rewrite_prompt'}, inplace=True)
ds = (
    ray.data.from_pandas(df)
    .map(get_chat_input)
).repartition(250) # need to force into multiple blocks <https://discuss.ray.io/t/single-node-4x-gpu-map-batches-only-using-1/12313/2>



# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=30, n=2)


# Create a class to do batch inference.
class LLMPredictor:

    def __init__(self):
        self.llm = LLM(
            model=model_name,
            # cache_dir="/gpfs/home/yiyayu/scratch/cache",
            gpu_memory_utilization=0.85,
            max_model_len=1200,
            trust_remote_code=True,
            enable_lora=True 
        )
        # self.lora_path = 'mistral_pr_lora_over65'

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        outputs = self.llm.generate(
            batch["input"], 
            sampling_params,
            # lora_request=LoRARequest("pr_adapter", 1, self.lora_path)
        )
        
        generated_text = []
        for output in outputs:
            generated_text.append([o.text for o in output.outputs])
        batch["rewrite_prompts"] = generated_text

        return batch

ds = ds.map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    concurrency=2,
    # Specify the number of GPUs required per LLM instance.
    # NOTE: Do NOT set `num_gpus` when using vLLM with tensor-parallelism
    # (i.e., `tensor_parallel_size`).
    num_gpus=1,
    # Specify the batch size for inference.
    batch_size=4,
)



ds.write_parquet("./data/exp_test/multi_n_with_lora3/")

