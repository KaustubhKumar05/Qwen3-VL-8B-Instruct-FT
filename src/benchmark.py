"""Main benchmark orchestrator for vision LLM evaluation."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from src.judge import KitchenAnalysisJudge
from src.openrouter_client import OpenRouterClient
from src.reporter import BenchmarkReporter


class VisionLLMBenchmark:
    """Orchestrator for running vision LLM benchmarks."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the benchmark.

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
        self.system_prompt, self.user_prompt = self.load_prompts(self.config)

        # Get models to test
        self.models = self.config.get("models_to_test", [])
        if not self.models:
            raise ValueError("No models specified in config.yaml")

        # Setup results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("results") / timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Results will be saved to: {self.results_dir}")

    def load_prompts(self, config) -> tuple[str, str]:
        """
        Load system and user prompts from the prompts dir.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        prompt_file = Path(f"prompts/{self.config.get('prompt_file_name', 'base')}.txt")
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"{self.config.get('prompt_file_name', 'base')}.txt not found in prompts directory"
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

    def discover_samples(self) -> List[Dict[str, Path]]:
        """
        Discover all sample image/JSON pairs in the Samples directory.

        Returns:
            List of dictionaries with 'image' and 'ground_truth' paths
        """
        samples_dir = Path("Samples")
        if not samples_dir.exists():
            raise FileNotFoundError("Samples directory not found")

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

    def run_model_on_sample(
        self, model: str, sample: Dict[str, Path]
    ) -> Dict[str, Any]:
        """
        Run a single model on a single sample.

        Args:
            model: Model identifier
            sample: Sample dictionary with image and ground_truth paths

        Returns:
            Dictionary with results including score, usage, and response
        """

        try:
            # Call the model
            response = self.client.analyze_kitchen_image(
                model=model,
                image_path=sample["image"],
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
            )
            # Parse the JSON response
            response_text = response["response"]

            # Try to extract JSON from response (in case model adds markdown formatting)
            try:
                # Try direct parse first
                parsed_response = json.loads(response_text)
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

                parsed_response = json.loads(response_text)

            # Load ground truth
            with open(sample["ground_truth"], "r", encoding="utf-8") as f:
                ground_truth = json.load(f)

            # Score the response
            detailed_scores = self.judge.score_analysis(ground_truth, parsed_response)
            score = detailed_scores["overall_score"]

            # Save response to file
            model_dir = self.results_dir / model.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            response_file = model_dir / f"{sample['name']}_response.json"
            with open(response_file, "w") as f:
                json.dump(
                    {
                        "sample": sample["name"],
                        "model": model,
                        "response": parsed_response,
                        "ground_truth": ground_truth,
                        "score": score,
                        "detailed_scores": detailed_scores,
                        "usage": response["usage"],
                        "cost": response["cost"],
                    },
                    f,
                    indent=2,
                )

            # Convert score to percentage
            score_pct = score * 100

            print(f"\nScore for {sample['name']}: {score_pct:.1f}%")

            return {
                "sample_name": sample["name"],
                "score": score_pct,
                "detailed_scores": detailed_scores,
                "usage": response["usage"],
                "cost": response["cost"],
                "response_file": str(response_file),
                "success": True,
            }

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

    def run_model(self, model: str, samples: List[Dict[str, Path]]) -> Dict[str, Any]:
        """
        Run a single model on all samples (parallelized).

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
                executor.submit(self.run_model_on_sample, model, sample): sample
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

    def run(self):
        """Run the complete benchmark."""
        print("=" * 60)
        print("Vision LLM Benchmark")
        print("=" * 60)

        # Discover samples
        samples = self.discover_samples()
        print(f"\nFound {len(samples)} samples")
        print(f"Testing {len(self.models)} models")

        start_time = datetime.now()

        # Run each model
        all_results = []
        for model in self.models:
            model_result = self.run_model(model, samples)
            all_results.append(model_result)

        end_time = datetime.now()

        # Generate report
        print("\n" + "=" * 60)
        print("Generating report...")
        print("=" * 60)

        reporter = BenchmarkReporter(self.results_dir)
        report = reporter.generate_report(all_results, start_time, end_time)
        report_path = reporter.save_report(report)

        print("\nBenchmark complete!")
        print(f"Report saved to: {report_path}")
        print(f"Results saved to: {self.results_dir}")


def main():
    """Main entry point."""
    try:
        benchmark = VisionLLMBenchmark()
        benchmark.run()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
