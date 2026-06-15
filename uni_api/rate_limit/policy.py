from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


TPR_PERIOD = -1
DEFAULT_RATE_LIMIT = "999999/min"

_TIME_UNITS = {
    "s": 1,
    "sec": 1,
    "second": 1,
    "m": 60,
    "min": 60,
    "minute": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
    "d": 86400,
    "day": 86400,
    "mo": 2592000,
    "month": 2592000,
    "y": 31536000,
    "year": 31536000,
    "tpr": TPR_PERIOD,
}


@dataclass(frozen=True, slots=True)
class RateLimitRule:
    count: int
    period_seconds: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.count, self.period_seconds)


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    rules_by_model: Mapping[str, tuple[RateLimitRule, ...]]

    @classmethod
    def from_config(cls, value: Any) -> "RateLimitPolicy":
        if isinstance(value, str):
            return cls({"default": _parse_rules(value)})
        if isinstance(value, Mapping):
            return cls({
                str(model): _parse_rules(raw_value)
                for model, raw_value in value.items()
            })
        return cls({"default": _parse_rules(DEFAULT_RATE_LIMIT)})

    @classmethod
    def from_legacy_rules(cls, rules: Any) -> "RateLimitPolicy":
        return cls(
            {
                "default": tuple(
                    RateLimitRule(int(count), int(period))
                    for count, period in list(rules or [])
                )
            }
        )

    def rules_for(self, model: str | None) -> tuple[RateLimitRule, ...]:
        if model and model in self.rules_by_model:
            return self.rules_by_model[model]

        if model:
            for limit_model, rules in self.rules_by_model.items():
                if limit_model != "default" and limit_model in model:
                    return rules

        return self.rules_by_model.get("default", _parse_rules(DEFAULT_RATE_LIMIT))

    def as_legacy_dict(self) -> dict[str, list[tuple[int, int]]]:
        return {
            model: [rule.as_tuple() for rule in rules]
            for model, rules in self.rules_by_model.items()
        }


def parse_rate_limit(limit_string: str) -> list[tuple[int, int]]:
    return [rule.as_tuple() for rule in _parse_rules(limit_string)]


def _parse_rules(value: Any) -> tuple[RateLimitRule, ...]:
    if not isinstance(value, str):
        raise ValueError(f"Invalid rate limit format: {value}")

    rules: list[RateLimitRule] = []
    for raw_limit in value.split(","):
        limit = raw_limit.strip()
        match = re.match(r"^(\d+)/(\w+)$", limit)
        if not match:
            raise ValueError(f"Invalid rate limit format: {limit}")

        count_text, unit = match.groups()
        if unit not in _TIME_UNITS:
            raise ValueError(f"Unknown time unit: {unit}")

        rules.append(RateLimitRule(int(count_text), _TIME_UNITS[unit]))
    return tuple(rules)
