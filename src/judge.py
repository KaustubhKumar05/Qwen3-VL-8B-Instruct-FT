"""JSON comparison and scoring logic for kitchen analysis results."""

import math
from typing import Any, Dict, List, Tuple

FINISH_SIMILARITY = {
    frozenset({"Laminate", "PU"}): 0.6,
    frozenset({"PU", "Acrylic"}): 0.7,
    frozenset({"Laminate", "Acrylic"}): 0.4,
}


class KitchenAnalysisJudge:
    """Judge for comparing kitchen analysis JSON outputs."""

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the judge with scoring weights.

        Args:
            weights: Dictionary of component weights for scoring
        """
        self.weights = weights or {
            "base": 0.14,
            "wall": 0.14,
            "tall": 0.14,
            "loft": 0.14,
            "dado": 0.04,
            "floor": 0.04,
            "colors": 0.16,
            "handles": 0.10,
            "baseUnitCoverage": 0.05,
            "wallUnitCoverage": 0.05,
        }

    def score_finish_similarity(self, expected: str, actual: str) -> float:
        """
        Score finish family similarity with partial credit for commonly confused types.

        Args:
            expected: Expected finish family
            actual: Actual finish family

        Returns:
            Score from 0.0 to 1.0
        """
        if expected == actual:
            return 1.0
        return FINISH_SIMILARITY.get(frozenset({expected, actual}), 0.0)

    def score_coverage(self, expected: int, actual: int, tolerance: int = 10) -> float:
        """
        Score unit coverage percentage with tolerance.

        Args:
            expected: Expected coverage percentage (0-100)
            actual: Actual coverage percentage (0-100)
            tolerance: Allowed difference for full score

        Returns:
            Score from 0.0 to 1.0
        """
        if expected is None and actual is None:
            return 1.0
        if expected is None or actual is None:
            return 0.0
        diff = abs(expected - actual)
        if diff <= tolerance:
            return 1.0
        return max(0.0, 1.0 - (diff - tolerance) / (100 - tolerance))

    def hex_to_lab(self, hex_color: str) -> Tuple[float, float, float]:
        """
        Convert hex color to LAB color space for perceptual comparison.

        Args:
            hex_color: Hex color string (e.g., "#FFFFFF")

        Returns:
            Tuple of (L, a, b) values
        """
        # Remove # if present
        hex_color = hex_color.lstrip("#")

        # Convert to RGB
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0

        # Convert RGB to XYZ
        def rgb_to_xyz_component(c):
            if c > 0.04045:
                return ((c + 0.055) / 1.055) ** 2.4
            else:
                return c / 12.92

        r = rgb_to_xyz_component(r)
        g = rgb_to_xyz_component(g)
        b = rgb_to_xyz_component(b)

        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        # Convert XYZ to LAB (using D65 illuminant)
        x = x / 0.95047
        y = y / 1.00000
        z = z / 1.08883

        def xyz_to_lab_component(c):
            if c > 0.008856:
                return c ** (1 / 3)
            else:
                return (7.787 * c) + (16 / 116)

        x = xyz_to_lab_component(x)
        y = xyz_to_lab_component(y)
        z = xyz_to_lab_component(z)

        L = (116 * y) - 16
        a = 500 * (x - y)
        b = 200 * (y - z)

        return (L, a, b)

    def color_distance(self, hex1: str, hex2: str) -> float:
        """
        Calculate perceptual color distance (Delta E) between two hex colors.

        Args:
            hex1: First hex color
            hex2: Second hex color

        Returns:
            Delta E value (0 = identical, higher = more different)
        """
        try:
            L1, a1, b1 = self.hex_to_lab(hex1)
            L2, a2, b2 = self.hex_to_lab(hex2)

            # Calculate Delta E (CIE76 formula)
            delta_e = math.sqrt((L2 - L1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)
            return delta_e
        except (ValueError, KeyError):
            # If color parsing fails, return maximum distance
            return 100.0

    def score_color_similarity(self, expected: str, actual: str) -> float:
        """
        Score color similarity between expected and actual hex colors.

        Args:
            expected: Expected hex color
            actual: Actual hex color

        Returns:
            Score from 0.0 (completely different) to 1.0 (identical)
        """
        delta_e = self.color_distance(expected, actual)

        # Convert Delta E to score (Delta E < 2 is barely perceptible)
        # Delta E > 50 is very different
        if delta_e < 2:
            return 1.0
        elif delta_e > 50:
            return 0.0
        else:
            # Linear scaling between 2 and 50
            return 1.0 - ((delta_e - 2) / 48)

    def score_colors_array(self, expected: List[Dict], actual: List[Dict]) -> float:
        """
        Score color arrays considering both color similarity and coverage.

        Args:
            expected: Expected colors array
            actual: Actual colors array

        Returns:
            Score from 0.0 to 1.0
        """
        if not expected and not actual:
            return 1.0
        if not expected or not actual:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        # For each expected color, find the best matching actual color
        for exp_color in expected:
            exp_hex = exp_color.get("hex", "")
            exp_coverage = exp_color.get("coverage", 0)

            best_match_score = 0.0
            best_match_coverage = 0

            for act_color in actual:
                act_hex = act_color.get("hex", "")
                act_coverage = act_color.get("coverage", 0)

                color_sim = self.score_color_similarity(exp_hex, act_hex)

                if color_sim > best_match_score:
                    best_match_score = color_sim
                    best_match_coverage = act_coverage

            # Coverage accuracy (how close the coverages match)
            coverage_diff = abs(exp_coverage - best_match_coverage)
            coverage_score = max(0.0, 1.0 - (coverage_diff / 100.0))

            # Combined score: 70% color similarity, 30% coverage accuracy
            combined_score = (best_match_score * 0.7) + (coverage_score * 0.3)

            total_score += combined_score * exp_coverage
            total_weight += exp_coverage

        return total_score / total_weight if total_weight > 0 else 0.0

    def score_handle(self, expected: Dict, actual: Dict) -> float:
        """
        Score handle detection and attributes.

        Args:
            expected: Expected handle dict
            actual: Actual handle dict

        Returns:
            Score from 0.0 to 1.0
        """
        if not expected and not actual:
            return 1.0
        if not expected or not actual:
            return 0.0

        scores = []

        # Detection score
        exp_detected = expected.get("detected", False)
        act_detected = actual.get("detected", False)
        scores.append(1.0 if exp_detected == act_detected else 0.0)

        # If both detected as false, rest doesn't matter
        if not exp_detected and not act_detected:
            return 1.0

        # Type score
        exp_type = expected.get("type", "unknown")
        act_type = actual.get("type", "unknown")
        scores.append(1.0 if exp_type == act_type else 0.0)

        # Finish score
        exp_finish = expected.get("finish", "unknown")
        act_finish = actual.get("finish", "unknown")
        scores.append(1.0 if exp_finish == act_finish else 0.0)

        # Hex color score
        exp_hex = expected.get("hex")
        act_hex = actual.get("hex")
        if exp_hex and act_hex:
            scores.append(self.score_color_similarity(exp_hex, act_hex))
        elif exp_hex is None and act_hex is None:
            scores.append(1.0)
        else:
            scores.append(0.0)

        return sum(scores) / len(scores)

    def score_component(
        self, component_name: str, expected: Dict, actual: Dict
    ) -> Dict[str, float]:
        """
        Score a single component (base, wall, tall, loft, dado, floor).

        Args:
            component_name: Name of the component
            expected: Expected component dict
            actual: Actual component dict

        Returns:
            Dictionary with detailed scores
        """
        if not expected and not actual:
            return {
                "total": 1.0,
                "detected": 1.0,
                "finishFamily": 1.0,
                "colors": 1.0,
                "handle": 1.0,
            }
        if not expected or not actual:
            return {
                "total": 0.0,
                "detected": 0.0,
                "finishFamily": 0.0,
                "colors": 0.0,
                "handle": 0.0,
            }

        scores = {}

        # Detection score
        exp_detected = expected.get("detected", False)
        act_detected = actual.get("detected", False)
        scores["detected"] = 1.0 if exp_detected == act_detected else 0.0

        # FinishFamily score (with partial credit for similar finishes)
        exp_finish = expected.get("finishFamily", "Unknown")
        act_finish = actual.get("finishFamily", "Unknown")
        scores["finishFamily"] = self.score_finish_similarity(exp_finish, act_finish)

        # Colors score
        exp_colors = expected.get("colors", [])
        act_colors = actual.get("colors", [])
        scores["colors"] = self.score_colors_array(exp_colors, act_colors)

        # Handle score (only for cabinet components)
        if component_name in ["base", "wall", "tall", "loft"]:
            exp_handle = expected.get("handle", {})
            act_handle = actual.get("handle", {})
            scores["handle"] = self.score_handle(exp_handle, act_handle)
        else:
            scores["handle"] = 1.0  # N/A for non-cabinet components

        # Overall component score (weighted average)
        scores["total"] = (
            scores["detected"] * 0.3
            + scores["finishFamily"] * 0.2
            + scores["colors"] * 0.3
            + scores["handle"] * 0.2
        )

        return scores

    def score_ceiling_lighting(self, expected: Dict, actual: Dict) -> Dict[str, float]:
        """
        Score ceiling lighting detection and attributes.

        Args:
            expected: Expected ceiling lighting dict
            actual: Actual ceiling lighting dict

        Returns:
            Dictionary with detailed scores
        """
        if not expected and not actual:
            return {"total": 1.0}
        if not expected or not actual:
            return {"total": 0.0}

        scores = {}

        # Detection
        exp_detected = expected.get("detected", False)
        act_detected = actual.get("detected", False)
        scores["detected"] = 1.0 if exp_detected == act_detected else 0.0

        # Type
        exp_type = expected.get("type", "unknown")
        act_type = actual.get("type", "unknown")
        scores["type"] = 1.0 if exp_type == act_type else 0.0

        # Count (with tolerance)
        exp_count = expected.get("count")
        act_count = actual.get("count")
        if exp_count is not None and act_count is not None:
            count_diff = abs(exp_count - act_count)
            scores["count"] = max(0.0, 1.0 - (count_diff / max(exp_count, 1)))
        elif exp_count is None and act_count is None:
            scores["count"] = 1.0
        else:
            scores["count"] = 0.0

        # Layout
        exp_layout = expected.get("layout", "unknown")
        act_layout = actual.get("layout", "unknown")
        scores["layout"] = 1.0 if exp_layout == act_layout else 0.0

        # Color temperature
        exp_temp = expected.get("colorTemperature", "unknown")
        act_temp = actual.get("colorTemperature", "unknown")
        scores["colorTemperature"] = 1.0 if exp_temp == act_temp else 0.0

        # Overall score
        scores["total"] = sum(scores.values()) / len(scores)

        return scores

    def score_analysis(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score the complete kitchen analysis.

        Args:
            expected: Expected (ground truth) analysis
            actual: Actual (model output) analysis

        Returns:
            Dictionary containing detailed scores and overall score
        """
        component_scores = {}

        # Score each cabinet component
        for component in ["base", "wall", "tall", "loft", "dado", "floor"]:
            exp_comp = expected.get(component, {})
            act_comp = actual.get(component, {})
            component_scores[component] = self.score_component(component, exp_comp, act_comp)

        # Score ceiling lighting
        exp_lighting = expected.get("ceilingLighting", {})
        act_lighting = actual.get("ceilingLighting", {})
        component_scores["ceilingLighting"] = self.score_ceiling_lighting(
            exp_lighting, act_lighting
        )

        # Score unit coverage percentages
        exp_base_coverage = expected.get("baseUnitCoverage")
        act_base_coverage = actual.get("baseUnitCoverage")
        base_coverage_score = self.score_coverage(exp_base_coverage, act_base_coverage)
        component_scores["baseUnitCoverage"] = {"total": base_coverage_score}

        exp_wall_coverage = expected.get("wallUnitCoverage")
        act_wall_coverage = actual.get("wallUnitCoverage")
        wall_coverage_score = self.score_coverage(exp_wall_coverage, act_wall_coverage)
        component_scores["wallUnitCoverage"] = {"total": wall_coverage_score}

        # Calculate overall score using weights
        overall_score = 0.0
        for component, weight in self.weights.items():
            if component in component_scores:
                overall_score += component_scores[component]["total"] * weight

        return {
            "overall_score": overall_score,
            "component_scores": component_scores,
        }
