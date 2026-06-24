"""Leakage-safe smoothed Markov transition probabilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MarkovObservation:
    source: str
    target: str
    day_type: str | None = None


class SmoothedMarkovModel:
    """Estimate ``P(j|i)=(c_ij + alpha*P0(j))/(c_i + alpha)``."""

    def __init__(
        self,
        *,
        alpha: float = 5.0,
        state_space_size: int = 10_000,
        prior_pseudocount: float = 1.0,
        min_day_type_support: int = 25,
    ) -> None:
        if alpha <= 0 or state_space_size <= 0 or prior_pseudocount <= 0:
            raise ValueError("alpha, state_space_size, and prior_pseudocount must be positive")
        self.alpha = float(alpha)
        self.state_space_size = int(state_space_size)
        self.prior_pseudocount = float(prior_pseudocount)
        self.min_day_type_support = int(min_day_type_support)
        self.target_counts: Counter[str] = Counter()
        self.global_transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self.day_type_transitions: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.observation_count = 0

    def update(self, observation: MarkovObservation) -> None:
        self.target_counts[observation.target] += 1
        self.global_transitions[observation.source][observation.target] += 1
        if observation.day_type is not None:
            self.day_type_transitions[
                (observation.day_type, observation.source)
            ][observation.target] += 1
        self.observation_count += 1

    def fit(self, observations: Iterable[MarkovObservation]) -> "SmoothedMarkovModel":
        for observation in observations:
            self.update(observation)
        return self

    def prior_probability(self, target: str) -> float:
        denominator = (
            self.observation_count
            + self.prior_pseudocount * self.state_space_size
        )
        return (
            self.target_counts[target] + self.prior_pseudocount
        ) / denominator

    def _counts(self, source: str, day_type: str | None) -> Counter[str]:
        if day_type is not None:
            local = self.day_type_transitions.get((day_type, source), Counter())
            if sum(local.values()) >= self.min_day_type_support:
                return local
        return self.global_transitions.get(source, Counter())

    def probability(
        self,
        source: str,
        target: str,
        *,
        day_type: str | None = None,
    ) -> float:
        counts = self._counts(source, day_type)
        support = sum(counts.values())
        return (
            counts[target] + self.alpha * self.prior_probability(target)
        ) / (support + self.alpha)

    def mixture_probability(
        self,
        sources: Sequence[str],
        target: str,
        *,
        day_type: str | None = None,
    ) -> float:
        if not sources:
            return self.prior_probability(target)
        return sum(
            self.probability(source, target, day_type=day_type)
            for source in sources
        ) / len(sources)


def select_alpha_walk_forward(
    observations: Sequence[MarkovObservation],
    *,
    alphas: Sequence[float] = (1.0, 5.0, 10.0, 25.0),
    warmup: int = 100,
    state_space_size: int = 10_000,
) -> tuple[float, dict[float, float]]:
    """Select smoothing strength by past-only prequential log-loss."""
    if len(observations) <= warmup:
        raise ValueError("not enough observations for alpha selection")
    losses: dict[float, float] = {}
    for alpha in alphas:
        model = SmoothedMarkovModel(
            alpha=float(alpha), state_space_size=state_space_size
        )
        cumulative = 0.0
        count = 0
        for index, observation in enumerate(observations):
            if index >= warmup:
                probability = model.probability(
                    observation.source,
                    observation.target,
                    day_type=observation.day_type,
                )
                cumulative -= math.log(max(probability, 1e-15))
                count += 1
            model.update(observation)
        losses[float(alpha)] = cumulative / max(1, count)
    best = min(losses, key=lambda value: (losses[value], value))
    return best, losses
