"""Information propagation model — category-dependent event delays by tier."""

from __future__ import annotations

import random
from typing import Any


# Delay rules: {category: {access_level: (min_delay, max_delay)}}
# Delay is in simulation steps. Step T+0 = immediate.
PROPAGATION_DELAYS: dict[str, dict[str, tuple[int, int]]] = {
    "REGULATORY": {"HIGH": (0, 0), "MEDIUM": (1, 1)},
    "EARNINGS":   {"HIGH": (0, 0), "MEDIUM": (1, 1)},
    "NEWS":       {"HIGH": (0, 0), "MEDIUM": (0, 1)},
    "MACRO":      {"HIGH": (0, 0), "MEDIUM": (0, 1)},
    "POLITICAL":  {"HIGH": (0, 0), "MEDIUM": (1, 1)},
    "RUMOR":      {"HIGH": (1, 1), "MEDIUM": (0, 2)},
}

# Tier -> access level mapping
TIER_ACCESS: dict[int, str] = {
    1: "HIGH",
    2: "MEDIUM",
    3: "LOW",  # Tier 3 never receives events
}


def distribute_events(
    events: list[dict[str, Any]],
    current_step: int,
    tier: int,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Filter events visible to this tier at this step.

    Events are tagged with their injection step. This function returns
    events whose delay has elapsed for the given tier.
    """
    if tier == 3:
        return []  # Tier 3 reacts to prices only

    if rng is None:
        rng = random.Random()

    access = TIER_ACCESS[tier]
    visible: list[dict[str, Any]] = []

    for event in events:
        category = event.get("category", "NEWS")
        delays = PROPAGATION_DELAYS.get(category, {"HIGH": (0, 0), "MEDIUM": (0, 1)})
        delay_range = delays.get(access, (0, 0))
        delay = rng.randint(delay_range[0], delay_range[1])

        injection_step = event.get("_injection_step", current_step)
        if current_step >= injection_step + delay:
            visible.append(event)

    return visible
