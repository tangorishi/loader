from huggingface_hub.file_download import hf_hub_download
import time
import safetensors

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
filename = "model.safetensors"

start = time.time()
path = hf_hub_download(model_name, filename, force_download=True)
downloaded = time.time()


data = {}
with safetensors.safe_open(path,framework="pt",device="cuda") as f:
    for name in f.keys():
        data[name] = f.get_tensor(name)
total_size = 0
for tensor in data.values():
    total_size += tensor.nelement() * tensor.element_size()
moved = time.time()

print(f"Downloaded in {downloaded - start} seconds, moved to GPU in {moved - downloaded} seconds. Total model size: {total_size / 1024 / 1024} MB")