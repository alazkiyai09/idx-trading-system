"""
Response Validator Module

Validates and parses structured LLM outputs.
Ensures JSON responses conform to expected schemas.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ResponseValidator:
    """Validates LLM responses against expected schemas and formats.

    Example:
        validator = ResponseValidator()
        data = validator.parse_json(response.content)
        validator.validate_agent_report(data)
    """

    @staticmethod
    def extract_json(content: str) -> str:
        """Extract JSON from a response that may contain non-JSON text.

        Handles common cases:
        - Pure JSON
        - JSON wrapped in markdown code fences
        - JSON embedded in explanatory text

        Args:
            content: Raw response content.

        Returns:
            Extracted JSON string.
        """
        content = content.strip()

        # Try parsing as-is first
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in content
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            candidate = json_match.group()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Try to find JSON array in content
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            candidate = json_match.group()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        return content

    @staticmethod
    def parse_json(content: str) -> Any:
        """Parse JSON from LLM response content.

        Args:
            content: Raw response content.

        Returns:
            Parsed JSON data.

        Raises:
            ValueError: If content cannot be parsed as JSON.
        """
        extracted = ResponseValidator.extract_json(content)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {e}\n"
                f"Content (first 500 chars): {content[:500]}"
            )

    @staticmethod
    def validate_required_fields(
        data: Dict[str, Any],
        required_fields: List[str],
    ) -> List[str]:
        """Validate that required fields are present.

        Args:
            data: Parsed JSON data.
            required_fields: List of required field names.

        Returns:
            List of missing field names (empty if all present).
        """
        return [f for f in required_fields if f not in data]

    @staticmethod
    def validate_score_range(
        score: Any,
        min_val: float = 0.0,
        max_val: float = 100.0,
        field_name: str = "score",
    ) -> float:
        """Validate and clamp a score to a valid range.

        Args:
            score: Score value to validate.
            min_val: Minimum valid value.
            max_val: Maximum valid value.
            field_name: Name of the field for error messages.

        Returns:
            Validated score, clamped to range.
        """
        try:
            score = float(score)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid {field_name} value: {score}. Defaulting to {(min_val + max_val) / 2}"
            )
            return (min_val + max_val) / 2

        if score < min_val or score > max_val:
            logger.warning(
                f"{field_name} {score} outside range [{min_val}, {max_val}]. Clamping."
            )
            return max(min_val, min(max_val, score))

        return score

    @staticmethod
    def validate_agent_report(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize an agent report from LLM.

        Expected structure:
        {
            "overall_score": 0-100,
            "findings": [...],
            "recommendation": "string",
            "confidence": 0-1
        }

        Args:
            data: Parsed agent report data.

        Returns:
            Validated and normalized report data.
        """
        validated = {}

        # Validate overall_score
        validated["overall_score"] = ResponseValidator.validate_score_range(
            data.get("overall_score", 50.0),
            field_name="overall_score",
        )

        # Validate findings
        findings = data.get("findings", [])
        if not isinstance(findings, list):
            logger.warning("findings is not a list, wrapping in list")
            findings = [findings] if findings else []

        validated_findings = []
        for f in findings:
            if isinstance(f, dict):
                validated_findings.append({
                    "category": str(f.get("category", "general")),
                    "description": str(f.get("description", "")),
                    "severity": str(f.get("severity", "info")),
                    "score": ResponseValidator.validate_score_range(
                        f.get("score", 50.0), field_name="finding.score"
                    ),
                })
        validated["findings"] = validated_findings

        # Validate recommendation
        validated["recommendation"] = str(data.get("recommendation", ""))

        # Validate confidence
        validated["confidence"] = ResponseValidator.validate_score_range(
            data.get("confidence", 0.5),
            min_val=0.0,
            max_val=1.0,
            field_name="confidence",
        )

        return validated

    @staticmethod
    def validate_stop_loss(
        proposed_stop: float,
        current_stop: Optional[float],
        entry_price: float,
        max_distance_pct: float = 0.15,
    ) -> float:
        """Validate a proposed stop loss from LLM.

        Rules:
        - Stop loss must be below entry price (for BUY signals)
        - Stop can only move UP (tighter), never down
        - Stop must be within max_distance_pct of entry price

        Args:
            proposed_stop: Proposed stop loss price.
            current_stop: Current stop loss price (if any).
            entry_price: Entry price.
            max_distance_pct: Maximum distance from entry as decimal.

        Returns:
            Validated stop loss price.
        """
        # Must be below entry
        if proposed_stop >= entry_price:
            logger.warning(
                f"Stop loss {proposed_stop} >= entry {entry_price}. Using 5% below entry."
            )
            proposed_stop = entry_price * 0.95

        # Cannot move stop DOWN
        if current_stop is not None and proposed_stop < current_stop:
            logger.info(
                f"Stop loss can only move up. Keeping current stop {current_stop} "
                f"instead of proposed {proposed_stop}."
            )
            return current_stop

        # Must be within max distance
        min_stop = entry_price * (1 - max_distance_pct)
        if proposed_stop < min_stop:
            logger.warning(
                f"Stop loss {proposed_stop} too far from entry {entry_price} "
                f"(max {max_distance_pct*100}%). Clamping to {min_stop:.2f}."
            )
            proposed_stop = min_stop

        return proposed_stop
