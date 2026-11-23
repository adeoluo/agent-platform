"""Evaluation: validity rate with and without the constraint, and selectivity."""

from __future__ import annotations

from dataclasses import dataclass

from constrained_generation_library.constrainer import (
    constrained_generate,
    unconstrained_generate,
)
from constrained_generation_library.data import default_vocab, random_model
from constrained_generation_library.regex import Regex


@dataclass(frozen=True)
class ValidityReport:
    """Validity rates over a batch of generations and the constraint's selectivity."""

    trials: int
    constrained_valid: int
    unconstrained_valid: int
    mean_allowed_fraction: float

    @property
    def constrained_rate(self) -> float:
        return self.constrained_valid / self.trials if self.trials else 0.0

    @property
    def unconstrained_rate(self) -> float:
        return self.unconstrained_valid / self.trials if self.trials else 0.0

    def format(self) -> str:
        return (
            f"constrained valid {self.constrained_rate:.3f} "
            f"({self.constrained_valid}/{self.trials}), "
            f"unconstrained valid {self.unconstrained_rate:.3f} "
            f"({self.unconstrained_valid}/{self.trials})"
        )


def validity_report(
    regex: Regex, *, trials: int = 100, max_tokens: int = 120, vocab: list[str] | None = None
) -> ValidityReport:
    """Run ``trials`` random models with and without the constraint, and tally validity."""

    vocab = vocab or default_vocab()
    constrained_valid = 0
    unconstrained_valid = 0
    fraction_total = 0.0
    for seed in range(trials):
        model = random_model(seed)
        result = constrained_generate(regex, vocab, model, max_tokens=max_tokens)
        if regex.fullmatch(result.text):
            constrained_valid += 1
        fraction_total += result.mean_allowed_fraction
        if regex.fullmatch(unconstrained_generate(vocab, model, max_tokens=max_tokens)):
            unconstrained_valid += 1
    return ValidityReport(
        trials=trials,
        constrained_valid=constrained_valid,
        unconstrained_valid=unconstrained_valid,
        mean_allowed_fraction=fraction_total / trials if trials else 0.0,
    )
