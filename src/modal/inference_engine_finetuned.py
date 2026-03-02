import json
import socket
import subprocess
from typing import Any

import aiohttp

import modal

MODEL_NAME = "kozonhf/room-analysis-qwen3-vl-8b-10e"

GPU = "H100"
N_GPU = 1

MAX_SEQ_LEN = "4096"

MINUTES = 60
PORT = 8000
STARTUP_TIMEOUT = 10 * MINUTES
SCALEDOWN_WINDOW = 1 * MINUTES

PUBLIC_ENDPOINT = "https://kaustubhkumar05--inference-engine-finetuned-vllmserver-serve.modal.run"

inference_engine = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.13.0", "huggingface-hub==0.36.0", "bitsandbytes>=0.46.1")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
inference_engine_cache_vol = modal.Volume.from_name(
    "inference-engine-cache", create_if_missing=True
)

app = modal.App("inference-engine-finetuned")

with inference_engine.imports():
    import requests


def sleep(level=1):
    requests.post(f"http://localhost:{PORT}/sleep?level={level}").raise_for_status()


def wake_up():
    requests.post(f"http://localhost:{PORT}/wake_up").raise_for_status()


def wait_ready(proc: subprocess.Popen):
    while True:
        try:
            socket.create_connection(("localhost", PORT), timeout=1).close()
            return
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {proc.returncode}")


def warmup():
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "Who are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://localhost:{PORT}/v1/chat/completions",
            json=payload,
            timeout=300,
        ).raise_for_status()


@app.cls(
    image=inference_engine,
    gpu=f"{GPU}:{N_GPU}",
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=STARTUP_TIMEOUT,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": inference_engine_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class VllmServer:
    @modal.enter(snap=True)
    def start(self):
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
            "--tensor-parallel-size",
            str(N_GPU),
            "--max-model-len",
            str(MAX_SEQ_LEN),
            "--quantization",
            "fp8",
            "--enable-prefix-caching",
            "--enable-sleep-mode",
            "--max-num-seqs",
            "2",
            "--block-size",
            "32",
            "--swap-space",
            "2",
            "--gpu-memory-utilization",
            "0.7",
            "--no-enforce-eager"
        ]
        print(*cmd)
        self.vllm_proc = subprocess.Popen(cmd)
        wait_ready(self.vllm_proc)
        warmup()
        sleep()

    @modal.enter(snap=False)
    def wake(self):
        wake_up()
        wait_ready(self.vllm_proc)

    @modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT, requires_proxy_auth=True)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.vllm_proc.terminate()


@app.local_entrypoint()
async def test(test_timeout=STARTUP_TIMEOUT, content=None):
    url = VllmServer.serve.get_web_url()

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
                line = line[len("data: "):]

            chunk = json.loads(line)
            assert chunk["object"] == "chat.completion.chunk"
            print(chunk["choices"][0]["delta"]["content"], end="")
    print()
