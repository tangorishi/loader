from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# download meta-llama/Llama-2-7b-chat-hf
start = time.time()
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
model = AutoModelForCausalLM.from_pretrained(model_name, force_download=True, torch_dtype=torch.bfloat16, device_map="auto")

downloaded = time.time()

model_size_bytes = 0
for p in model.parameters():
    model_size_bytes += p.nelement() * p.element_size()

model.to("cuda")
moved = time.time()

print(f"Downloaded in {downloaded - start} seconds, moved to GPU in {moved - downloaded} seconds. Total model size: {model_size_bytes / 1024 / 1024} MB")


