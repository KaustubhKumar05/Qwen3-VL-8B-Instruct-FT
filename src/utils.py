"""Shared utilities."""

import base64
import io
from pathlib import Path

from PIL import Image

MAX_IMAGE_DIM = 512


def resize_image(image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    """Resize image so its longest side is max_dim, preserving aspect ratio."""
    if max(image.size) <= max_dim:
        return image
    ratio = max_dim / max(image.size)
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
    return image.resize(new_size, Image.LANCZOS)


def encode_image_b64(image_path: Path, max_dim: int = MAX_IMAGE_DIM) -> tuple[str, str]:
    """Load, resize, and base64-encode an image. Returns (b64_string, mime_type)."""
    image = Image.open(image_path).convert("RGB")
    image = resize_image(image, max_dim)

    ext = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
    fmt = "PNG" if ext == "png" else "JPEG"

    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return b64, mime
