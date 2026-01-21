#!/usr/bin/env python3
"""Verification script to test the benchmark setup without calling APIs."""

import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv


def main():
    print("=" * 60)
    print("Vision LLM Benchmark - Setup Verification")
    print("=" * 60)
    print()

    errors = []
    warnings = []

    # Check Python version
    print("✓ Python version:", sys.version.split()[0])

    # Check config file
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        models = config.get("models_to_test", [])
        print(f"✓ Config file loaded: {len(models)} models configured")
        for model in models:
            print(f"  - {model}")
    except Exception as e:
        errors.append(f"Config file error: {e}")
        print(f"✗ Config file error: {e}")

    # Check API key
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key and api_key != "your_api_key_here":
        print("✓ OpenRouter API key configured")
    elif not api_key:
        errors.append("OPENROUTER_API_KEY not found in .env file")
        print("✗ OPENROUTER_API_KEY not found in .env file")
    else:
        warnings.append("API key is still set to placeholder value")
        print("⚠ API key is still set to placeholder value")

    # Check samples
    samples_dir = Path("Samples")
    if samples_dir.exists():
        json_files = list(samples_dir.glob("*.json"))
        img_files = (
            list(samples_dir.glob("*.jpg"))
            + list(samples_dir.glob("*.jpeg"))
            + list(samples_dir.glob("*.png"))
        )
        print(
            f"✓ Samples directory: {len(json_files)} JSON files, {len(img_files)} images"
        )

        # Validate sample pairs
        for json_file in json_files:
            name = json_file.stem
            has_image = False
            for ext in [".jpg", ".jpeg", ".png"]:
                if (samples_dir / f"{name}{ext}").exists():
                    has_image = True
                    break
            if not has_image:
                warnings.append(f"No image found for {json_file.name}")
    else:
        errors.append("Samples directory not found")
        print("✗ Samples directory not found")

    # Check dependencies
    try:
        from src.main import VisionLLMOrchestrator
        from src.judge import KitchenAnalysisJudge
        from src.openrouter_client import OpenRouterClient
        from src.reporter import BenchmarkReporter

        print("✓ All Python modules can be imported")
    except Exception as e:
        errors.append(f"Module import error: {e}")
        print(f"✗ Module import error: {e}")

    # Test judge functionality
    try:
        judge = KitchenAnalysisJudge()
        score = judge.score_color_similarity("#FFFFFF", "#FFFFFF")
        if score == 1.0:
            print("✓ Color scoring working correctly")
        else:
            warnings.append("Color scoring may have issues")
    except Exception as e:
        errors.append(f"Judge test failed: {e}")
        print(f"✗ Judge test failed: {e}")

    # Test sample loading
    try:
        if json_files:
            with open(json_files[0], "r") as f:
                sample_data = json.load(f)
            required_keys = [
                "base",
                "wall",
                "tall",
                "loft",
                "dado",
                "floor",
                "ceilingLighting",
            ]
            if all(key in sample_data for key in required_keys):
                print("✓ Sample JSON format validated")
            else:
                warnings.append("Sample JSON may be missing required keys")
    except Exception as e:
        warnings.append(f"Sample validation error: {e}")

    print()
    print("=" * 60)

    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  ✗ {error}")
        print()

    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
        print()

    if not errors:
        print("✅ Setup verification passed!")
        print()
        print("Ready to run benchmark:")
        print("  uv run python -m src.main")
        print()
        print("Or generate ground truth:")
        print("  uv run python -m src.main --generate-ground-truth")
        print()
        return 0
    else:
        print("❌ Setup verification failed. Please fix the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
