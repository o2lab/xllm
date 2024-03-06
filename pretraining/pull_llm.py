from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

import os
os.environ['HF_DATASETS_CACHE'] = '/scratch/user/siweicui/LLM/huggingface/'

# model = "meta-llama/Llama-2-7b-chat-hf"
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

checkpoint = "meta-llama/Llama-2-70b-chat-hf"
device = "cpu" # for GPU usage or "cpu" for CPU usage


tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="/scratch/user/siweicui/LLM/huggingface/")
# to save memory consider using fp16 or bf16 by specifying torch_dtype=torch.float16 for example
model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir="/scratch/user/siweicui/LLM/huggingface/").to(device)

