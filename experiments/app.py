import requests
import time
import httpx
import asyncio


async def download_url(url, num_workers: int):
    # resolve any redirect
    resp = requests.head(url, allow_redirects=True)
    final_url = resp.url
    content_size = resp.headers.get("Content-Length")

    buffers = []

    async with httpx.AsyncClient() as client:
        tasks = []
        start = 0
        chunksize = int(content_size) // num_workers

        for i in range(num_workers):
            if i == num_workers - 1:
                chunksize = int(content_size) - start
            tasks.append(
                asyncio.create_task(
                    client.get(
                        final_url,
                        headers={"Range": f"bytes={start}-{start + chunksize}"},
                    )
                )
            )
            start += chunksize

        resps = await asyncio.gather(*tasks)
        for resp in resps:
            buffers.append(resp.content)

    total_bytes = sum(map(len, buffers))
    return total_bytes


class Loader:
    def __init__(self):
        ...

    def mmap(self, url):
        total_bytes = asyncio.run(download_url(url, 4))
        print(f"Downloaded {total_bytes} bytes")


if __name__ == "__main__":
    url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.6/resolve/main/model.safetensors"

    loader = Loader()

    start = time.perf_counter_ns()
    loader.mmap(url)
    duration = time.perf_counter_ns() - start
    print(f"Downloaded in {duration / 1e9} seconds")
