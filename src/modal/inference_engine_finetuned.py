import json
from typing import Any

import aiohttp

import modal

MODEL_NAME = "kozonhf/room-analysis-qwen3-vl-8b"

GPU = "A10G"

# To be reduced
MAX_SEQ_LEN = "12288"

FAST_BOOT = True

PUBLIC_ENDPOINT = "https://kaustubhkumar05--inference-engine-finetuned-serve.modal.run"

inference_engine = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.13.0", "huggingface-hub==0.36.0", "bitsandbytes>=0.46.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
inference_engine_cache_vol = modal.Volume.from_name(
    "inference-engine-cache", create_if_missing=True
)


app = modal.App("inference-engine-finetuned")
N_GPU = 1
STARTUP_TIMEOUT = 5 * 60  # 5 min to start vLLM
SCALEDOWN_WINDOW = 1 * 60  # 1 min idle before shutdown
PORT = 8000


@app.function(
    image=inference_engine,
    gpu=f"{GPU}:{N_GPU}",
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=STARTUP_TIMEOUT,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": inference_engine_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],  # For private model access
)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT, requires_proxy_auth=True)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--max-num-seqs",
        "2",
        "--block-size",
        "32",
        "--swap-space",
        "2",
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", str(N_GPU)]
    cmd += ["--max-model-len", str(MAX_SEQ_LEN)]
    cmd += ["--enable-prefix-caching"]

    print(*cmd)
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
async def test(test_timeout=STARTUP_TIMEOUT, content=None):
    url = serve.get_web_url()

    system_prompt = {
        "role": "system",
        "content": "Respond tersely",
    }
    if content is None:
        content = "Wake up"

    messages = [
        system_prompt,
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * 60) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, "llm", messages)


async def _send_request(session: aiohttp.ClientSession, model: str, messages: list) -> None:
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post("/v1/chat/completions", json=payload, headers=headers) as resp:
        async for raw in resp.content:
            resp.raise_for_status()

            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert chunk["object"] == "chat.completion.chunk"
            print(chunk["choices"][0]["delta"]["content"], end="")
    print()
