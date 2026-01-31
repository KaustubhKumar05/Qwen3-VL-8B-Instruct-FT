# Modal Inference Cost Optimization

Ideas for reducing costs when running Qwen3-VL-8B-Instruct on Modal.

## Current Setup

| Setting | Value | Cost |
|---------|-------|------|
| Model | Qwen/Qwen3-VL-8B-Instruct (BF16) | ~16GB weights |
| GPU | L40S (48GB VRAM) | ~$1.40/hr |
| max_model_len | 153,520 | High KV cache usage |
| scaledown_window | 2 min | Idle cost |

## Optimization Options

### 1. Use FP8 Quantization (Recommended)

Qwen provides an official FP8 quantized version with "nearly identical" quality.

```python
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct-FP8"
GPU = "A10G"  # 24GB VRAM, ~$0.60/hr
MAX_SEQ_LEN = "32768"  # Reduce if needed to fit in memory
```

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| VRAM usage | ~16GB weights | ~9GB weights | ~44% |
| GPU cost | $1.40/hr (L40S) | $0.60/hr (A10G) | **57%** |

**Links:**
- [Qwen3-VL-8B-Instruct-FP8 on HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8)
- [vLLM Qwen3-VL guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)

### 2. Use AWQ 4-bit Quantization

Community-provided AWQ quantization for even smaller memory footprint.

```python
MODEL_NAME = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
GPU = "T4"  # 16GB VRAM, ~$0.27/hr
MAX_SEQ_LEN = "16384"
```

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| VRAM usage | ~16GB weights | ~4GB weights | ~75% |
| GPU cost | $1.40/hr (L40S) | $0.27/hr (T4) | **81%** |

**Tradeoff:** May have slight quality degradation compared to BF16/FP8.

### 3. Reduce Context Length

If you don't need 153K context, reducing `max_model_len` saves KV cache memory.

```python
# For typical image analysis (image + prompt + response)
MAX_SEQ_LEN = "32768"  # 32K is usually sufficient

# For very short responses
MAX_SEQ_LEN = "16384"  # 16K minimum
```

### 4. Reduce Idle Time

Already implemented - container shuts down after 2 min idle.

```python
SCALEDOWN_WINDOW = 2 * 60  # 2 minutes
```

### 5. GPU Options by VRAM

| GPU | VRAM | Cost/hr | Best For |
|-----|------|---------|----------|
| T4 | 16GB | ~$0.27 | AWQ 4-bit, short context |
| A10G | 24GB | ~$0.60 | FP8, medium context |
| L4 | 24GB | ~$0.50 | FP8, medium context |
| L40S | 48GB | ~$1.40 | BF16, long context |
| A100 | 80GB | ~$2.50 | Full BF16, max context |

## Implementation Checklist

- [ ] Test FP8 model quality on sample images
- [ ] Measure actual token usage to determine min context length needed
- [ ] Benchmark latency on A10G vs L40S
- [ ] Consider AWQ if quality is acceptable

## Estimated Savings

| Configuration | Cost/hr | Per 1-min inference | Monthly (100 runs) |
|---------------|---------|---------------------|-------------------|
| Current (L40S BF16) | $1.40 | ~$0.05 | ~$5.00 |
| FP8 on A10G | $0.60 | ~$0.02 | ~$2.00 |
| AWQ on T4 | $0.27 | ~$0.01 | ~$1.00 |
