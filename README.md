# VLM Benchmark for room interior inspection

A comprehensive benchmarking system for evaluating vision language models on room interior analysis tasks using OpenRouter.

## Overview

This project benchmarks multiple vision LLMs by comparing their JSON outputs against ground truth samples. It uses structural JSON comparison with perceptual color similarity scoring to evaluate model performance, tracks token usage and costs, and generates detailed markdown reports.

## Features

- 🎯 **Automated Benchmarking**: Test multiple models against sample images
- 📊 **Structural Scoring**: Field-by-field JSON comparison with LAB color space similarity
- 💰 **Cost Tracking**: Monitor token usage and API costs per model
- 📝 **Detailed Reports**: Comprehensive markdown reports with per-model and per-sample breakdowns
- 🔄 **Response Storage**: Save all model responses for manual review
- ⚙️ **Easy Configuration**: YAML-based config for models and scoring weights

## Project Structure

```
.
├── config.yaml              # Benchmark configuration
├── pyproject.toml           # UV project dependencies
├── .env                     # API key
├── src/
│   ├── benchmark.py         # Main benchmark orchestrator
│   ├── openrouter_client.py # OpenRouter API client
│   ├── judge.py             # JSON comparison/scoring logic
│   └── reporter.py          # Report generation
├── Samples/                 # Sample images and ground truth JSON
│   ├── Kitchen_01.jpg
│   ├── Kitchen_01.json
│   └── ...
└── results/                 # Benchmark results (timestamped)
    └── YYYYMMDD_HHMMSS/
        ├── model_name/      # Per-model responses
        └── report.md        # Final benchmark report
```

## Setup

### Prerequisites

- Python 3.11 or higher
- [UV](https://github.com/astral-sh/uv) package manager

### Installation

1. **Install UV** (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone/navigate to the project directory**:

   ```bash
   cd /path/to/snipe007-ai-test
   ```

3. **Install dependencies**:

   ```bash
   uv sync
   ```

4. **Get your OpenRouter API key**:

   - Visit [https://openrouter.ai/keys](https://openrouter.ai/keys)
   - Sign up or log in (GitHub OAuth supported)
   - Click "Create Key" and copy your API key
   - Add credits to your account (pay-per-use pricing)

5. **Configure environment variables**:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API key:

   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

### Configuration

Edit `config.yaml` to customize the benchmark:

```yaml
# Models to test (from OpenRouter)
models_to_test:
  - openai/gpt-4o
  - anthropic/claude-3.5-sonnet
  - google/gemini-pro-vision

# OpenRouter settings
openrouter:
  base_url: https://openrouter.ai/api/v1
  timeout: 60
  max_concurrent_requests: 10 # Parallel requests per model (increase for speed)

# Scoring weights for components
scoring_weights:
  base: 0.15
  wall: 0.15
  tall: 0.15
  loft: 0.15
  dado: 0.05
  floor: 0.05
  colors: 0.20
  handles: 0.10
```

**Performance Tips:**

- Increase `max_concurrent_requests` (5-10) for faster benchmarking
- Be mindful of API rate limits per model
- More workers = faster but higher chance of rate limiting

Browse available models at [https://openrouter.ai/models](https://openrouter.ai/models)

## Usage

### Run the Benchmark

```bash
uv run python -m src.benchmark
```

The benchmark will:

1. Discover all sample image/JSON pairs in `Samples/`
2. Test each configured model on each sample
3. Score responses against ground truth using structural comparison
4. Save all responses to `results/{timestamp}/{model_name}/`
5. Generate a comprehensive markdown report at `results/{timestamp}/report.md`

### Understanding the Output

During execution, you'll see:

```
============================================================
Vision LLM Benchmark
============================================================

Found 10 samples
Testing 3 models

Testing model: openai/gpt-4o
============================================================
  Score for Kitchen_01: 87.5%
  Score for Kitchen_02: 92.1%
  ...
```

After completion:

```
Benchmark complete!
Report saved to: results/20260111_143022/report.md
Results saved to: results/20260111_143022
```

## Scoring Methodology

The benchmark uses **structural JSON comparison** with scores displayed as **percentages (0-100%)**.

### Component Scoring

Each kitchen component (base, wall, tall, loft, dado, floor) is scored based on:

- **Detection accuracy** (30%): Boolean match
- **Finish family** (20%): Exact string match
- **Color similarity** (30%): LAB color space Delta E calculation
- **Handle detection** (20%): Type, finish, and color matching

### Color Similarity

Colors are compared using the LAB color space for perceptual accuracy:

- Delta E < 2: Barely perceptible difference (score = 1.0)
- Delta E > 50: Very different colors (score = 0.0)
- Linear interpolation between these values

### Overall Score

The final score is a weighted average of all components using the weights defined in `config.yaml`.

## Report Contents

Generated reports include:

1. **Summary Table**: Average score (%), total cost, average cost per request, and duration per model
2. **Detailed Results**: Per-sample scores (%) and costs with links to response files
3. **Component Analysis**: Average scores (%) for each kitchen component
4. **Best/Worst Samples**: Identify strengths and weaknesses

All scores are displayed as percentages from 0-100% for easy interpretation.

### Cost Metrics

The benchmark tracks two key cost metrics:

- **Total Cost**: Total spent on all samples for that model
- **Avg Cost/Request**: Average cost per API request (total cost ÷ number of samples)

## Adding New Samples

To add new test samples:

1. Place the image in `Samples/` (e.g., `Kitchen_11.jpg`)
2. Create the ground truth JSON (e.g., `Kitchen_11.json`) following the schema
3. Ensure the filename prefix matches (e.g., `Kitchen_11`)
4. Run the benchmark - new samples are automatically discovered

## Troubleshooting

### "OPENROUTER_API_KEY not found"

- Ensure `.env` file exists in the project root
- Verify the API key is properly set in `.env`

### Model errors

- Check that model names in `config.yaml` match OpenRouter's format
- Verify you have sufficient credits in your OpenRouter account
- Some models may have rate limits or availability restrictions

### Color scoring issues

- Ensure hex colors in ground truth are valid `#RRGGBB` format
- Check that color coverage values sum to approximately 100

## OpenRouter Pricing

Pricing varies by model. Check current rates at [https://openrouter.ai/models](https://openrouter.ai/models)

## Development

### Project Dependencies

- `openai`: OpenRouter-compatible API client
- `pyyaml`: Configuration file parsing
- `pillow`: Image processing
- `python-dotenv`: Environment variable management

### Running in Development

```bash
# Activate virtual environment
source .venv/bin/activate

# Run benchmark
python -m src.benchmark
```

Check the OpenRouter documentation: [https://openrouter.ai/docs](https://openrouter.ai/docs)

## Sample benchmark result:

| Model                          | Avg Score (%) | Samples | Total Cost | Avg Cost/Request | Time  |
| ------------------------------ | ------------- | ------- | ---------- | ---------------- | ----- |
| openai/gpt-5.2                 | 50.2%         | 10      | $0.1801    | $0.0180          | 1.4m  |
| qwen/qwen3-vl-30b-a3b-instruct | 46.1%         | 10      | $0.0101    | $0.0010          | 38.2s |
| mistralai/ministral-14b-2512   | 41.7%         | 10      | $0.0093    | $0.0009          | 14.4s |
| allenai/molmo-2-8b:free        | 37.1%         | 10      | $0.0000    | $0.0000          | 20.5s |
| qwen/qwen3-vl-8b-instruct      | 47.7%         | 10      | $0.0124    | $0.0012          | 25.9s |
| mistralai/ministral-3b-2512    | 42.4%         | 10      | $0.0047    | $0.0005          | 13.2s |

### What Gets Scored

Each kitchen image is analyzed for 7 components:

1. **Base Cabinets** (15%) - Lower cabinets below the counter
2. **Wall Cabinets** (15%) - Upper cabinets on the wall
3. **Tall Cabinets** (15%) - Full-height units (pantries, etc.)
4. **Loft Cabinets** (15%) - Highest cabinets near the ceiling
5. **Dado/Backsplash** (5%) - Wall covering behind counters
6. **Floor** (5%) - Flooring material
7. **Colors** (20%) - How accurately colors are identified
8. **Handles** (10%) - Hardware type and finish

### For Each Component, We Check:

- **Did the AI see it?** (30%) - Is the component detected?
- **Right material?** (20%) - Correct finish (laminate, wood, etc.)
- **Right colors?** (30%) - How close are the colors?
- **Right hardware?** (20%) - Correct handle type and finish

### Color Matching

Colors are scored using human perception:
- **Identical colors** = 100%
- **Slightly different** = High score (hard to notice)
- **Completely different** = 0%

The system uses LAB color space
