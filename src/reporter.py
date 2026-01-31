"""Report generation for benchmark results."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class BenchmarkReporter:
    """Generate markdown reports for benchmark results."""

    def __init__(self, results_dir: Path):
        """
        Initialize the reporter.

        Args:
            results_dir: Directory where results are stored
        """
        self.results_dir = results_dir

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def generate_report(
        self,
        benchmark_results: List[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        """
        Generate a markdown report from benchmark results.

        Args:
            benchmark_results: List of results for each model
            start_time: Benchmark start time
            end_time: Benchmark end time

        Returns:
            Markdown report string
        """
        duration = (end_time - start_time).total_seconds()

        report = []
        report.append("# Vision LLM Benchmark Report")
        report.append("")
        report.append(f"**Generated:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Duration:** {self.format_duration(duration)}")
        report.append("")

        # Summary Table
        report.append("## Summary")
        report.append("")
        report.append("| Model | Avg Score (%) | Samples | Total Cost | Avg Cost/Request | Time |")
        report.append("|-------|---------------|---------|------------|------------------|------|")

        for result in benchmark_results:
            model_name = result["model"]
            avg_score = result["average_score"]
            num_samples = len(result["sample_results"])
            total_cost = result["total_cost"]
            model_duration = result["duration"]

            if total_cost is not None:
                total_cost_str = f"${total_cost:.4f}"
                avg_cost_per_request = total_cost / num_samples if num_samples > 0 else 0
                avg_cost_str = f"${avg_cost_per_request:.4f}"
            else:
                total_cost_str = "N/A"
                avg_cost_str = "N/A"

            report.append(
                f"| {model_name} | {avg_score:.1f}% | {num_samples} | "
                f"{total_cost_str} | {avg_cost_str} | {self.format_duration(model_duration)} |"
            )

        report.append("")

        # Add note about costs if all are N/A
        all_costs_na = all(r["total_cost"] is None for r in benchmark_results)
        if all_costs_na:
            report.append(
                "*Note: Cost data not available. "
                "OpenRouter may not provide cost information for all models or API configurations.*"
            )
            report.append("")

        # Per-Model Detailed Results
        report.append("## Detailed Results by Model")
        report.append("")

        for result in benchmark_results:
            model_name = result["model"]
            report.append(f"### {model_name}")
            report.append("")

            # Sample scores table
            report.append("| Sample | Score (%) | Cost | Response File |")
            report.append("|--------|-----------|------|---------------|")

            for sample_result in result["sample_results"]:
                sample_name = sample_result["sample_name"]
                score = sample_result["score"]
                cost = sample_result["cost"]
                response_file = sample_result["response_file"]

                cost_str = f"${cost:.4f}" if cost is not None else "N/A"
                rel_path = Path(response_file).relative_to(self.results_dir.parent)

                report.append(f"| {sample_name} | {score:.1f}% | {cost_str} | [View]({rel_path}) |")

            report.append("")

            # Best and worst samples
            sorted_samples = sorted(
                result["sample_results"], key=lambda x: x["score"], reverse=True
            )

            if sorted_samples:
                best = sorted_samples[0]
                worst = sorted_samples[-1]

                report.append(
                    f"**Best Sample:** {best['sample_name']} (Score: {best['score']:.1f}%)"
                )
                report.append("")
                report.append(
                    f"**Worst Sample:** {worst['sample_name']} (Score: {worst['score']:.1f}%)"
                )
                report.append("")

            # Score distribution
            scores = [s["score"] for s in result["sample_results"]]
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)

                report.append(f"**Score Range:** {min_score:.1f}% - {max_score:.1f}%")
                report.append("")
                report.append(f"**Average Score:** {avg_score:.1f}%")
                report.append("")

        # Component Analysis (if available)
        report.append("## Component Performance Analysis")
        report.append("")
        report.append("*Models ordered by average performance across all components*")
        report.append("")

        # Aggregate component scores across all models
        component_names = ["base", "wall", "tall", "loft", "dado", "floor", "ceilingLighting"]

        # Calculate overall averages for sorting
        model_overall_avgs = []
        for result in benchmark_results:
            model_name = result["model"]
            component_avgs = {comp: [] for comp in component_names}

            for sample_result in result["sample_results"]:
                detailed_scores = sample_result.get("detailed_scores", {})
                component_scores = detailed_scores.get("component_scores", {})

                for comp in component_names:
                    if comp in component_scores:
                        component_avgs[comp].append(component_scores[comp]["total"])

            # Calculate average for each component
            comp_scores = []
            for comp in component_names:
                if component_avgs[comp]:
                    avg = sum(component_avgs[comp]) / len(component_avgs[comp])
                    comp_scores.append(avg)

            # Calculate overall average
            overall_avg = sum(comp_scores) / len(comp_scores) if comp_scores else 0
            model_overall_avgs.append(
                {"model": model_name, "overall_avg": overall_avg, "component_avgs": component_avgs}
            )

        # Sort by overall average (descending)
        model_overall_avgs.sort(key=lambda x: x["overall_avg"], reverse=True)

        report.append("| Model | Base | Wall | Tall | Loft | Dado | Floor | Ceiling |")
        report.append("|-------|------|------|------|------|------|-------|---------|")

        for model_data in model_overall_avgs:
            model_name = model_data["model"]
            component_avgs = model_data["component_avgs"]

            # Calculate averages and convert to percentages
            avg_values = []
            for comp in component_names:
                if component_avgs[comp]:
                    avg = sum(component_avgs[comp]) / len(component_avgs[comp])
                    avg_values.append(f"{avg * 100:.1f}%")
                else:
                    avg_values.append("N/A")

            report.append(f"| {model_name} | {' | '.join(avg_values)} |")

        report.append("")

        # Footer
        report.append("---")
        report.append("")
        report.append("*Generated by Vision LLM Benchmark*")
        report.append("")

        return "\n".join(report)

    def save_report(self, report: str, filename: str = "report.md") -> Path:
        """
        Save the report to a file.

        Args:
            report: Markdown report string
            filename: Output filename

        Returns:
            Path to the saved report
        """
        report_path = self.results_dir / filename
        with open(report_path, "w") as f:
            f.write(report)

        return report_path
