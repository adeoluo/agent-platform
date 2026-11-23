"""Token-level constraint masking and constrained decoding.

A constraint tracks the current automaton state. Given a vocabulary of token
strings, it reports which tokens keep the output on a path the automaton can still
accept: a token is allowed only if feeding its characters never drives the
automaton into a dead state. An end-of-sequence token is allowed only when the
current state already accepts. Constrained decoding then masks a model's scores to
the allowed set, so the generated string is valid by construction, not by luck.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from constrained_generation_library.regex import Regex

EOS = "<eos>"
ScoreFn = Callable[[int, str], float]


def advance_token(regex: Regex, state: frozenset[int], token: str) -> frozenset[int] | None:
    """Return the state after consuming ``token``, or ``None`` if it dies."""

    current = state
    for ch in token:
        current = regex.step(current, ch)
        if regex.is_dead(current):
            return None
    return current


@dataclass
class Constraint:
    """Tracks automaton state and reports allowed tokens from a vocabulary."""

    regex: Regex
    vocab: Sequence[str]
    state: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = self.regex.start_state

    @property
    def accepting(self) -> bool:
        assert self.state is not None
        return self.regex.is_accepting(self.state)

    def allowed_ids(self) -> list[int]:
        """Return the token ids (indices into ``vocab``) allowed from the state."""

        assert self.state is not None
        allowed: list[int] = []
        for token_id, token in enumerate(self.vocab):
            if token == EOS:
                if self.accepting:
                    allowed.append(token_id)
                continue
            if advance_token(self.regex, self.state, token) is not None:
                allowed.append(token_id)
        return allowed

    def advance(self, token_id: int) -> None:
        """Advance the automaton by the chosen token (a no-op for EOS)."""

        token = self.vocab[token_id]
        if token == EOS:
            return
        assert self.state is not None
        nxt = advance_token(self.regex, self.state, token)
        if nxt is None:
            raise ValueError(f"token {token!r} is not allowed in the current state")
        self.state = nxt


@dataclass(frozen=True)
class GenerationResult:
    """The output of a constrained decode."""

    text: str
    token_ids: list[int]
    steps: int
    mean_allowed_fraction: float


def constrained_generate(
    regex: Regex,
    vocab: Sequence[str],
    score: ScoreFn,
    *,
    max_tokens: int = 64,
) -> GenerationResult:
    """Greedily decode a valid string, masking the model's scores to allowed tokens.

    ``score(token_id, prefix)`` returns a preference for each candidate; the
    highest-scoring allowed token is chosen at each step. Decoding stops when the
    end-of-sequence token is selected.
    """

    constraint = Constraint(regex=regex, vocab=vocab)
    chosen: list[int] = []
    prefix = ""
    allowed_fractions: list[float] = []
    eos_id = _eos_id(vocab)
    for _ in range(max_tokens):
        allowed = constraint.allowed_ids()
        if not allowed:
            break
        allowed_fractions.append(len(allowed) / len(vocab))
        best = max(allowed, key=lambda token_id: score(token_id, prefix))
        if best == eos_id:
            break
        chosen.append(best)
        prefix += vocab[best]
        constraint.advance(best)
    mean_fraction = sum(allowed_fractions) / len(allowed_fractions) if allowed_fractions else 0.0
    return GenerationResult(
        text=prefix, token_ids=chosen, steps=len(chosen), mean_allowed_fraction=mean_fraction
    )


def unconstrained_generate(vocab: Sequence[str], score: ScoreFn, *, max_tokens: int = 64) -> str:
    """Greedily decode without any constraint (used to show how often it is invalid)."""

    eos_id = _eos_id(vocab)
    prefix = ""
    for _ in range(max_tokens):
        best = max(range(len(vocab)), key=lambda token_id: score(token_id, prefix))
        if best == eos_id:
            break
        prefix += vocab[best]
    return prefix


def _eos_id(vocab: Sequence[str]) -> int:
    return vocab.index(EOS) if EOS in vocab else -1
