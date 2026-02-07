"""Modal-hosted vLLM client for vision LLM."""

import os
import time
from pathlib import Path

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI

from src.utils import encode_image_b64

from src.modal.inference_engine_container import (
    MODEL_NAME as BASE_MODEL_NAME,
)
from src.modal.inference_engine_container import (
    PUBLIC_ENDPOINT as BASE_ENDPOINT,
)
from src.modal.inference_engine_finetuned import (
    MODEL_NAME as FINETUNED_MODEL_NAME,
)
from src.modal.inference_engine_finetuned import (
    PUBLIC_ENDPOINT as FINETUNED_ENDPOINT,
)

MODAL_MODELS = {
    "modal-hosted/qwen3-vl-8b-fp8": (BASE_MODEL_NAME, BASE_ENDPOINT),
    "modal-hosted/room-analysis-qwen3-vl-8b": (FINETUNED_MODEL_NAME, FINETUNED_ENDPOINT),
}


class ModalHostedClient:
    def __init__(self, model_key: str, timeout: int = 120):
        if model_key not in MODAL_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODAL_MODELS.keys())}")

        self.model_name, self.endpoint = MODAL_MODELS[model_key]
        self.model_key = model_key
        self.timeout = timeout
        self._warmed_up = False

        self.headers = {
            "Modal-Key": os.environ.get("MODAL_TOKEN_ID", ""),
            "Modal-Secret": os.environ.get("MODAL_TOKEN_SECRET", ""),
        }

        self.client = OpenAI(
            api_key="not-needed",
            base_url=f"{self.endpoint}/v1",
            timeout=timeout,
            default_headers=self.headers,
        )

    def warmup(self, max_retries: int = 10, wait: float = 10.0) -> bool:
        """Wait for endpoint to be ready. Returns True if ready."""
        if self._warmed_up:
            return True

        print(f"Warming up {self.model_key}...")
        for attempt in range(1, max_retries + 1):
            try:
                resp = httpx.get(f"{self.endpoint}/health", headers=self.headers, timeout=30.0)
                if resp.status_code == 200:
                    print(f"Endpoint ready after {attempt} attempt(s)")
                    self._warmed_up = True
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            if attempt < max_retries:
                print(f"  Attempt {attempt}/{max_retries} - waiting {wait}s...")
                time.sleep(wait)

        print("Failed to warm up endpoint")
        return False

    def analyze_image(self, image_path: Path, system_prompt: str, user_prompt: str) -> dict:
        b64, mime = encode_image_b64(image_path)

        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                                },
                            ],
                        },
                    ],
                    max_tokens=4000,
                    temperature=0.0,
                )
                return {
                    "response": resp.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                        "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                    },
                    "cost": None,
                    "model": self.model_key,
                }
            except (APIConnectionError, APITimeoutError):
                if attempt < 4:
                    time.sleep(10)
                continue

        raise Exception("Modal endpoint failed after 5 retries")
