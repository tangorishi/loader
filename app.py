import time
import io
import pickle
import struct
import tempfile
import os
from contextlib import contextmanager

import requests
import torch

from model_loader import download_to_device, zip_extract, zip_list_range

def get_safe_tensor_metadata(url):
    # resolve any redirect
    resp = requests.head(url, allow_redirects=True)

    resolved_url = resp.url
    content_size = resp.headers.get("Content-Length")

    # Fetch the first 8 bytes of the file
    headers = {'Range': 'bytes=0-7'}
    response = requests.get(resolved_url, headers=headers)
    # Interpret the bytes as a little-endian unsigned 64-bit integer
    length_of_header = struct.unpack('<Q', response.content)[0]
    # Fetch length_of_header bytes starting from the 9th byte
    headers = {'Range': f'bytes=8-{7 + length_of_header}'}
    response = requests.get(url, headers=headers)
    # Interpret the response as a JSON object
    tensor_header = response.json()

    return {
        "resolved_url": resolved_url,
        "content_size": content_size,
        "header_size": length_of_header + 8,
        "data_size": int(content_size) -length_of_header - 8,
        "tensor_header": tensor_header,
    }

class LoaderByteTensor:
    def __init__(self, ptr, nbytes):
        self.ptr = ptr
        self.nbytes = nbytes

    @property
    def __cuda_array_interface__(self):
        return {
            "data": (self.ptr, False), # second item is read-only flag
            "shape": (self.nbytes,),
            "typestr": "|u1",
        }

class SafeTensorLoader:
    def __init__(self, urls, num_workers=None):
        assert len(urls) == 1, "Only one URL supported for now"
        for url in urls:
            self.metadata = get_safe_tensor_metadata(url)
        _, self.raw_pointer = download_to_device([self.metadata["resolved_url"]], [self.metadata["header_size"]], [self.metadata["data_size"]], num_workers)[0]
        self.data = self._create_torch_tensors(self.metadata, self.raw_pointer)

    def _create_torch_tensors(self, metadata, raw_pointer):
        data = {}
        for key, value in metadata["tensor_header"].items():
            if key == "__metadata__":
                continue

            dtype = value["dtype"]
            shape = value["shape"]
            start_offset, end_offset = value["data_offsets"]

            torch_dtype = {
                "BF16": torch.bfloat16,
                "F16": torch.float16,
                "F32": torch.float32,
            }[dtype]

            # create torch tensor from pointer
            ptr = raw_pointer + start_offset
            our_tensor = LoaderByteTensor(ptr, nbytes=end_offset-start_offset)
            untyped_storage = torch.as_tensor(our_tensor, device=torch.device("cuda")).untyped_storage()
            torch_tensor = torch.tensor(untyped_storage, device=torch.device("cuda"), dtype=torch_dtype).view(shape)

            assert torch_tensor.numel() == (end_offset-start_offset) // torch_tensor.element_size()
            assert torch_tensor.data_ptr() == ptr

            data[key] = torch_tensor

        return data

    def keys(self):
        return self.data.keys()

    def get_tensor(self, key):
        return self.data[key]

class _TorchUnpickler(pickle.Unpickler):
    def __init__(self, file):
        super().__init__(file)
        self.data_lookup = {}

    def persistent_load(self, inp):
        # print(inp)
        typename, storage_cls, fn, device, nelements = inp
        assert typename == "storage"
        size_per_element = {
            torch.BFloat16Storage: 2,
            torch.FloatStorage: 4,
            torch.DoubleStorage: 8,
        }[storage_cls]
        dtype = {
            torch.BFloat16Storage: torch.bfloat16,
            torch.FloatStorage: torch.float,
            torch.DoubleStorage: torch.double,
        }[storage_cls]

        # make an empty, shallow file to mmap
        fd, fp = tempfile.mkstemp()
        os.truncate(fd, nelements * size_per_element)

        untyped_storage = torch.UntypedStorage.from_file(fp, shared=False, nbytes=nelements*size_per_element)
        self.data_lookup[untyped_storage._cdata] = fn
        return torch.storage.TypedStorage(
            wrap_storage=untyped_storage, dtype=dtype
        )

class TorchCkptLoader:
    def __init__(self, urls, num_workers=None):
        assert len(urls) == 1, "Only one URL supported for now"
        for url in urls:
            resp = requests.head(url, allow_redirects=True)
            data_size = int(resp.headers.get("Content-Length"))
            self.metadata = {
                "resolved_url": resp.url,
                "header_size": 0,
                "data_size": data_size,
            }
        host_ptr, device_ptr = download_to_device([self.metadata["resolved_url"]], [self.metadata["header_size"]], [self.metadata["data_size"]], num_workers)[0]

        # TODO: some legacy models are not zipfile :(, maybe just do a fallback there.
        data_pkl_raw = zip_extract(host_ptr, data_size, "pytorch_model/data.pkl")

        fn_to_range = {fn: (offset, size) for (fn, offset, size) in zip_list_range(host_ptr, data_size)}

        unpickler = _TorchUnpickler(io.BytesIO(data_pkl_raw))
        loaded = unpickler.load()
        self.data = {}
        for k, shallow_tensor in loaded.items():
            fn = unpickler.data_lookup[shallow_tensor.untyped_storage()._cdata]
            offset, nbytes = fn_to_range[f"pytorch_model/data/{fn}"]

            ptr = device_ptr+offset
            our_tensor = LoaderByteTensor(ptr, nbytes=nbytes)
            untyped_storage = torch.as_tensor(our_tensor, device=torch.device("cuda")).untyped_storage()
            torch_tensor = torch.tensor(untyped_storage, device=torch.device("cuda"), dtype=shallow_tensor.dtype).view(shallow_tensor.shape)

            self.data[k] = torch_tensor

    def keys(self):
        return self.data.keys()

    def get_tensor(self, key):
        return self.data[key]

    def __getitem__(self, key):
        return self.get_tensor(key)

@contextmanager
def timeit(name):
    start = time.perf_counter_ns()
    yield
    end = time.perf_counter_ns()
    print(f"{name}: {(end-start)/1e9:.2f} s")

if __name__ == "__main__":
    # url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.6/resolve/main/model.safetensors"
    # url = "https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha/resolve/main/model-00001-of-00003.safetensors"
    # url = "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin"
    url = "https://huggingface.co/Locutusque/TinyMistral-248M/resolve/main/pytorch_model.bin"

    num_workers = 8

    with timeit("our download"):
        # loader = SafeTensorLoader([url], num_workers)
        loader = TorchCkptLoader([url], num_workers)


    # import safetensors
    # f = safetensors.safe_open("model.safetensors", "pt", device="cuda")
    # assert set(f.keys()) == set(loader.keys()), f"Keys don't match: {f.keys()} vs {loader.keys()}"
    # for key in f.keys():
    #     assert torch.allclose(f.get_tensor(key), loader.get_tensor(key)), f"Tensor {key} doesn't match"
    # print("All tensors match!")


    f = torch.load("pytorch_model.bin.1", map_location=torch.device("cuda"))
    assert set(f.keys()) == set(loader.keys()), f"Keys don't match: {f.keys()} vs {loader.keys()}"
    for key in f.keys():
        assert torch.allclose(f[key], loader[key]), f"Tensor {key} doesn't match"
    print("All tensors match!")
