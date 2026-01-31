"""OpenRouter API client for vision LLM interactions."""

import base64
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""

    def __init__(
        self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", timeout: int = 60
    ):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key
            base_url: OpenRouter API base URL
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

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
        self, model: str, image_path: Path, system_prompt: str, user_prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze a kitchen image using a vision model.

        Args:
            model: Model identifier (e.g., "openai/gpt-4o")
            image_path: Path to the kitchen image
            system_prompt: System prompt for the model
            user_prompt: User prompt for the model

        Returns:
            Dictionary containing:
                - response: The model's JSON response
                - usage: Token usage statistics
                - cost: Estimated cost (if available)
        """
        # Encode image to base64
        base64_image = self.encode_image(image_path)
        image_extension = image_path.suffix.lower().lstrip(".")

        # Determine MIME type
        mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_types.get(image_extension, "image/jpeg")

        # Create the API request
        try:
            response = self.client.chat.completions.create(
                model=model,
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
                temperature=0.0,  # Deterministic output for benchmarking
            )
            # Extract response content
            content = response.choices[0].message.content

            # Extract usage statistics
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            # Try to extract cost information from OpenRouter
            cost = None
            if hasattr(response, "usage") and response.usage:
                # Check for native_tokens fields (OpenRouter specific)
                if hasattr(response.usage, "native_tokens_prompt") and hasattr(
                    response.usage, "native_tokens_completion"
                ):
                    # OpenRouter uses native tokens for cost calculation
                    # This data is available in the raw response
                    pass

                # Try different cost field locations
                if hasattr(response.usage, "total_cost"):
                    cost = response.usage.total_cost
                elif hasattr(response.usage, "cost"):
                    cost = response.usage.cost

            # Also check response headers or metadata if available
            if cost is None and hasattr(response, "_raw_response"):
                raw = response._raw_response
                if hasattr(raw, "headers"):
                    # OpenRouter may include cost in headers
                    cost_header = raw.headers.get("x-ratelimit-cost") or raw.headers.get(
                        "X-RateLimit-Cost"
                    )
                    if cost_header:
                        try:
                            cost = float(cost_header)
                        except ValueError:
                            pass

            return {
                "response": content,
                "usage": usage,
                "cost": cost,
                "model": model,
            }

        except Exception as e:
            raise Exception(f"Error calling OpenRouter API for model {model}: {str(e)}")
