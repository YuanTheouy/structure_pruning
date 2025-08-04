import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig, OPTForCausalLM

def get_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        cache_dir="llm_weights",
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.seqlen = 2048

    layers = model.model.layers
    print(layers)
    raise RuntimeError


model_path = "/home/lisiqi/amc-LLM/model/llama-2-7b-hf"
get_model(model_path)