"""A token vocabulary and deterministic fake models for demonstration.

There is no language model here. The vocabulary is a set of short strings a model
might emit when writing JSON, and the fake models produce scores over them. A
random model stands in for an unconstrained model that frequently emits invalid
JSON; a biased model prefers plausible characters but still needs the constraint
to stay valid. The point is to show that masking to the automaton's allowed set
makes the output valid regardless of what the model prefers.
"""

from __future__ import annotations

import random
import string

from constrained_generation_library.constrainer import EOS


def default_vocab() -> list[str]:
    """Return a vocabulary of JSON-ish tokens plus an end-of-sequence marker."""

    tokens: list[str] = []
    tokens.extend(string.ascii_lowercase)
    tokens.extend(string.ascii_uppercase)
    tokens.extend(string.digits)
    tokens.extend(['"', ":", ",", "{", "}", "-", ".", " ", "_"])
    tokens.extend(["true", "false"])
    tokens.append(EOS)
    return tokens


def random_model(seed: int):  # type: ignore[no-untyped-def]
    """Return a score function assigning random, prefix-dependent preferences."""

    rng = random.Random(seed)
    cache: dict[tuple[int, str], float] = {}

    def score(token_id: int, prefix: str) -> float:
        key = (token_id, prefix)
        if key not in cache:
            cache[key] = rng.random()
        return cache[key]

    return score


def biased_model(target: str):  # type: ignore[no-untyped-def]
    """Return a score function that prefers emitting the characters of ``target``.

    This simulates a model that "wants" to produce a particular document; the
    constraint still guarantees validity even if the target drifts.
    """

    def score(token_id: int, prefix: str) -> float:
        from constrained_generation_library.data import default_vocab

        vocab = default_vocab()
        token = vocab[token_id]
        if token == EOS:
            return 1.0 if prefix == target else -1.0
        if target.startswith(prefix + token):
            return 2.0
        return -len(token)

    return score
