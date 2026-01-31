"""Self-hosted vLLM client for vision LLM interactions."""

import base64
import time
from pathlib import Path
from typing import Any, Dict

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI

from src.modal.inference_engine_container import MODEL_NAME, PUBLIC_ENDPOINT


class SelfHostedClient:
    """Client for interacting with self-hosted vLLM endpoint on Modal."""

    def __init__(
        self,
        base_url: str = PUBLIC_ENDPOINT,
        timeout: int = 120,
        max_retries: int = 5,
        warmup_wait: float = 10.0,
    ):
        """
        Initialize the self-hosted vLLM client.

        Args:
            base_url: Base URL for the Modal endpoint
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for cold start
            warmup_wait: Seconds to wait between retries during warmup

        Auth: Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars
              (create at https://modal.com/settings/proxy-auth-tokens)
        """
        import os

        self.base_url = base_url.rstrip("/")
        self.model_name = MODEL_NAME
        self.timeout = timeout
        self.max_retries = max_retries
        self.warmup_wait = warmup_wait

        # Modal proxy auth tokens
        self.modal_key = os.environ.get("MODAL_TOKEN_ID", "")
        self.modal_secret = os.environ.get("MODAL_TOKEN_SECRET", "")

        self.client = OpenAI(
            api_key="not-needed",
            base_url=f"{self.base_url}/v1",
            timeout=timeout,
            default_headers={
                "Modal-Key": self.modal_key,
                "Modal-Secret": self.modal_secret,
            },
        )
        self._warmed_up = False

    def warmup(self) -> bool:
        """
        Wait for the endpoint to be ready (handles cold start).

        Returns:
            True if endpoint is ready, False if warmup failed
        """
        if self._warmed_up:
            return True

        print(f"Warming up self-hosted endpoint at {self.base_url}...")

        headers = {
            "Modal-Key": self.modal_key,
            "Modal-Secret": self.modal_secret,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.get(
                    f"{self.base_url}/health",
                    headers=headers,
                    timeout=30.0,
                )
                if response.status_code == 200:
                    print(f"Self-hosted endpoint ready after {attempt} attempt(s)")
                    self._warmed_up = True
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            if attempt < self.max_retries:
                print(f"  Attempt {attempt}/{self.max_retries} - waiting {self.warmup_wait}s...")
                time.sleep(self.warmup_wait)

        print("Failed to warm up self-hosted endpoint")
        return False

    def encode_image(self, image_path: Path) -> str:
        """
        Encode an image file to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_image(
        self, image_path: Path, system_prompt: str, user_prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze a kitchen image using the self-hosted vision model.

        Args:
            image_path: Path to the kitchen image
            system_prompt: System prompt for the model
            user_prompt: User prompt for the model

        Returns:
            Dictionary containing:
                - response: The model's JSON response
                - usage: Token usage statistics
                - cost: None (self-hosted, no per-request cost)
        """
        base64_image = self.encode_image(image_path)
        image_extension = image_path.suffix.lower().lstrip(".")

        mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_types.get(image_extension, "image/jpeg")

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                                },
                            ],
                        },
                    ],
                    max_tokens=4000,
                    temperature=0.0,
                )

                content = response.choices[0].message.content

                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": (
                        response.usage.completion_tokens if response.usage else 0
                    ),
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }

                return {
                    "response": content,
                    "usage": usage,
                    "cost": None,  # Self-hosted, no per-request cost
                    "model": f"self-hosted/{self.model_name}",
                }

            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                if attempt < self.max_retries:
                    print(f"  Request failed, retrying ({attempt}/{self.max_retries})...")
                    time.sleep(self.warmup_wait)
                continue
            except Exception as e:
                raise Exception(f"Error calling self-hosted vLLM endpoint: {str(e)}")

        raise Exception(
            f"Self-hosted endpoint failed after {self.max_retries} retries: {last_error}"
        )
