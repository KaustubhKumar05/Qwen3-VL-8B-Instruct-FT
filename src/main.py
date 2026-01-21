"""Main orchestrator for vision LLM evaluation and ground truth generation."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from dotenv import load_dotenv

from src.judge import KitchenAnalysisJudge
from src.openrouter_client import OpenRouterClient
from src.reporter import BenchmarkReporter


# ============================================================================
# CORE UTILITY FUNCTIONS (Reusable)
# ============================================================================


def load_prompts(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Load system and user prompts from the prompts dir.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    prompt_file = Path(f"prompts/{config.get('prompt_file_name', 'base')}.txt")
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"{config.get('prompt_file_name', 'base')}.txt not found in prompts directory"
        )

    with open(prompt_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse the simplified prompt file
    if "SYSTEM_PROMPT:" not in content or "USER_PROMPT:" not in content:
        raise ValueError(
            "Prompt file must contain SYSTEM_PROMPT: and USER_PROMPT: sections"
        )

    # Extract system prompt
    system_start = content.find("SYSTEM_PROMPT:") + len("SYSTEM_PROMPT:")
    user_start = content.find("USER_PROMPT:")
    system_prompt = content[system_start:user_start].strip()

    # Extract user prompt
    user_prompt_start = user_start + len("USER_PROMPT:")
    user_prompt = content[user_prompt_start:].strip()

    return system_prompt, user_prompt


def discover_samples(samples_dir: Path) -> List[Dict[str, Path]]:
    """
    Discover all sample image/JSON pairs in the Samples directory.

    Args:
        samples_dir: Path to the samples directory

    Returns:
        List of dictionaries with 'name', 'image', and 'ground_truth' paths
    """
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    samples = []

    # Find all JSON files
    for json_file in sorted(samples_dir.glob("*.json")):
        # Find corresponding image file
        image_name = json_file.stem  # e.g., "Kitchen_01"
        image_file = None

        for ext in [".jpg", ".jpeg", ".png"]:
            potential_image = samples_dir / f"{image_name}{ext}"
            if potential_image.exists():
                image_file = potential_image
                break

        if image_file:
            samples.append(
                {"name": image_name, "image": image_file, "ground_truth": json_file}
            )
        else:
            print(f"Warning: No image found for {json_file.name}")

    return samples


def call_model_for_image(
    client: OpenRouterClient,
    model: str,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """
    Call a vision model to analyze an image.

    Args:
        client: OpenRouter client instance
        model: Model identifier
        image_path: Path to the image file
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model

    Returns:
        Dictionary containing response, usage, cost, and model info
    """
    return client.analyze_image(
        model=model,
        image_path=image_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON response from model output.

    Handles markdown code blocks and extracts JSON content.

    Args:
        response_text: Raw response text from the model

    Returns:
        Parsed JSON dictionary

    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    # Try direct parse first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        return json.loads(response_text)


# ============================================================================
# BENCHMARK-SPECIFIC FUNCTIONS
# ============================================================================


def score_and_save_result(
    sample_name: str,
    model: str,
    parsed_response: Dict[str, Any],
    ground_truth: Dict[str, Any],
    judge: KitchenAnalysisJudge,
    output_path: Path,
    usage: Dict[str, int],
    cost: float,
) -> Dict[str, Any]:
    """
    Score a model response and save the results to a file.

    Args:
        sample_name: Name of the sample
        model: Model identifier
        parsed_response: Parsed model response
        ground_truth: Ground truth data
        judge: Judge instance for scoring
        output_path: Path to save the result file
        usage: Token usage statistics
        cost: API call cost

    Returns:
        Dictionary with scoring results
    """
    # Score the response
    detailed_scores = judge.score_analysis(ground_truth, parsed_response)
    score = detailed_scores["overall_score"]

    # Save response to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "sample": sample_name,
                "model": model,
                "response": parsed_response,
                "ground_truth": ground_truth,
                "score": score,
                "detailed_scores": detailed_scores,
                "usage": usage,
                "cost": cost,
            },
            f,
            indent=2,
        )

    # Convert score to percentage
    score_pct = score * 100

    print(f"\nScore for {sample_name}: {score_pct:.1f}%")

    return {
        "sample_name": sample_name,
        "score": score_pct,
        "detailed_scores": detailed_scores,
        "usage": usage,
        "cost": cost,
        "response_file": str(output_path),
        "success": True,
    }


# ============================================================================
# GROUND TRUTH GENERATION FUNCTIONS
# ============================================================================


def generate_ground_truth_for_sample(
    client: OpenRouterClient,
    model: str,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """
    Generate ground truth data for a sample using a benchmark model.

    Args:
        client: OpenRouter client instance
        model: Benchmark model identifier
        image_path: Path to the image file
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model

    Returns:
        Parsed JSON response from the model
    """
    response = call_model_for_image(client, model, image_path, system_prompt, user_prompt)
    parsed_response = parse_json_response(response["response"])
    return parsed_response


def save_ground_truth_file(
    parsed_response: Dict[str, Any], output_path: Path
) -> None:
    """
    Save parsed response as a ground truth JSON file.

    Args:
        parsed_response: Parsed model response
        output_path: Path to save the ground truth file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_response, f, indent=4)


# ============================================================================
# MAIN ORCHESTRATOR CLASS
# ============================================================================


class VisionLLMOrchestrator:
    """Orchestrator for running vision LLM benchmarks and generating ground truth."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the orchestrator.

        Args:
            config_path: Path to the configuration file
        """
        # Load environment variables
        load_dotenv()

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Get API key
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment. "
                "Please create a .env file with your API key."
            )

        # Initialize client
        openrouter_config = self.config.get("openrouter", {})
        self.client = OpenRouterClient(
            api_key=self.api_key,
            base_url=openrouter_config.get("base_url", "https://openrouter.ai/api/v1"),
            timeout=openrouter_config.get("timeout", 60),
        )

        # Initialize judge
        scoring_weights = self.config.get("scoring_weights", {})
        self.judge = KitchenAnalysisJudge(weights=scoring_weights)

        # Load prompts
        self.system_prompt, self.user_prompt = load_prompts(self.config)

        # Get models to test
        self.models = self.config.get("models_to_test", [])

    def run_model_score_and_save_sample(
        self, model: str, sample: Dict[str, Path]
    ) -> Dict[str, Any]:
        """
        Run model on a sample, score it against ground truth, and save results.

        Args:
            model: Model identifier
            sample: Sample dictionary with image and ground_truth paths

        Returns:
            Dictionary with results including score, usage, and response
        """
        try:
            # Call the model
            response = call_model_for_image(
                self.client,
                model,
                sample["image"],
                self.system_prompt,
                self.user_prompt,
            )

            # Parse the JSON response
            parsed_response = parse_json_response(response["response"])

            # Load ground truth
            with open(sample["ground_truth"], "r", encoding="utf-8") as f:
                ground_truth = json.load(f)

            # Score and save the result
            model_dir = Path("results") / self.results_dir_name / model.replace("/", "_")
            response_file = model_dir / f"{sample['name']}_response.json"

            return score_and_save_result(
                sample_name=sample["name"],
                model=model,
                parsed_response=parsed_response,
                ground_truth=ground_truth,
                judge=self.judge,
                output_path=response_file,
                usage=response["usage"],
                cost=response["cost"],
            )

        except Exception as e:
            print(f"ERROR: {str(e)}")
            return {
                "sample_name": sample["name"],
                "score": 0.0,
                "detailed_scores": {},
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "cost": None,
                "response_file": None,
                "success": False,
                "error": str(e),
            }

    def benchmark_model_on_all_samples(self, model: str, samples: List[Dict[str, Path]]) -> Dict[str, Any]:
        """
        Benchmark a single model on all samples (parallelized), score and aggregate results.

        Args:
            model: Model identifier
            samples: List of sample dictionaries

        Returns:
            Dictionary with aggregated results for the model
        """
        print(f"\nTesting model: {model}")
        print("=" * 60)

        start_time = datetime.now()
        sample_results = []

        # Get max concurrent requests from config
        openrouter_config = self.config.get("openrouter", {})
        max_workers = openrouter_config.get("max_concurrent_requests", 5)

        print(f"Running with {max_workers} concurrent requests...")

        # Use ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(self.run_model_score_and_save_sample, model, sample): sample
                for sample in samples
            }

            # Collect results as they complete
            for future in as_completed(future_to_sample):
                result = future.result()
                sample_results.append(result)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Calculate aggregates
        successful_results = [r for r in sample_results if r["success"]]

        if successful_results:
            average_score = sum(r["score"] for r in successful_results) / len(
                successful_results
            )
            total_tokens = sum(r["usage"]["total_tokens"] for r in successful_results)
            total_prompt_tokens = sum(
                r["usage"]["prompt_tokens"] for r in successful_results
            )
            total_completion_tokens = sum(
                r["usage"]["completion_tokens"] for r in successful_results
            )

            costs = [r["cost"] for r in successful_results if r["cost"] is not None]
            total_cost = sum(costs) if costs else None
        else:
            average_score = 0.0
            total_tokens = 0
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = None

        print(f"\nCompleted {len(successful_results)}/{len(sample_results)} samples")
        print(f"Average score: {average_score:.1f}%")
        print(f"Duration: {duration:.1f}s")

        return {
            "model": model,
            "sample_results": sample_results,
            "average_score": average_score,
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_cost": total_cost,
            "duration": duration,
            "num_successful": len(successful_results),
            "num_total": len(sample_results),
        }

    def run_benchmark(self):
        """Run the complete benchmark."""
        print("=" * 60)
        print("Vision LLM Benchmark")
        print("=" * 60)

        if not self.models:
            raise ValueError("No models specified in config.yaml")

        # Setup results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir_name = timestamp
        results_dir = Path("results") / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Results will be saved to: {results_dir}")

        # Discover samples
        samples = discover_samples(Path("Samples"))
        print(f"\nFound {len(samples)} samples")
        print(f"Testing {len(self.models)} models")

        start_time = datetime.now()

        # Run each model
        all_results = []
        for model in self.models:
            model_result = self.benchmark_model_on_all_samples(model, samples)
            all_results.append(model_result)

        end_time = datetime.now()

        # Generate report
        print("\n" + "=" * 60)
        print("Generating report...")
        print("=" * 60)

        reporter = BenchmarkReporter(results_dir)
        report = reporter.generate_report(all_results, start_time, end_time)
        report_path = reporter.save_report(report)

        print("\nBenchmark complete!")
        print(f"Report saved to: {report_path}")
        print(f"Results saved to: {results_dir}")

    def generate_ground_truth(self):
        """Generate or regenerate ground truth files using the benchmark model."""
        print("=" * 60)
        print("Ground Truth Generation")
        print("=" * 60)

        # Get benchmark model from config
        benchmark_model = self.config.get("benchmark_model")
        if not benchmark_model:
            raise ValueError(
                "benchmark_model not specified in config.yaml. "
                "Please add a benchmark_model entry to generate ground truth."
            )

        # Get replace_all flag
        ground_truth_config = self.config.get("ground_truth", {})
        replace_all = ground_truth_config.get("replace_all", False)

        print(f"Using model: {benchmark_model}")
        print(f"Replace all: {replace_all}")
        print()

        # Discover samples
        samples_dir = Path("Samples")
        
        # Find all images in the samples directory
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(samples_dir.glob(ext))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print("No images found in Samples directory")
            return

        print(f"Found {len(image_files)} images")

        processed = 0
        skipped = 0
        errors = 0

        for image_file in image_files:
            image_name = image_file.stem
            ground_truth_path = samples_dir / f"{image_name}.json"

            # Check if we should skip this sample
            if ground_truth_path.exists() and not replace_all:
                print(f"⏭️  Skipping {image_name} (ground truth already exists)")
                skipped += 1
                continue

            try:
                print(f"🔄 Generating ground truth for {image_name}...")
                
                # Generate ground truth
                parsed_response = generate_ground_truth_for_sample(
                    self.client,
                    benchmark_model,
                    image_file,
                    self.system_prompt,
                    self.user_prompt,
                )

                # Save ground truth file
                save_ground_truth_file(parsed_response, ground_truth_path)
                
                print(f"✅ Generated ground truth for {image_name}")
                processed += 1

            except Exception as e:
                print(f"❌ Error generating ground truth for {image_name}: {str(e)}")
                errors += 1

        print("\n" + "=" * 60)
        print("Ground Truth Generation Complete")
        print("=" * 60)
        print(f"Processed: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Errors: {errors}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Vision LLM Benchmark and Ground Truth Generator"
    )
    parser.add_argument(
        "--generate-ground-truth",
        action="store_true",
        help="Generate ground truth files using benchmark model",
    )
    args = parser.parse_args()

    try:
        orchestrator = VisionLLMOrchestrator()

        if args.generate_ground_truth:
            orchestrator.generate_ground_truth()
        else:
            orchestrator.run_benchmark()

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
