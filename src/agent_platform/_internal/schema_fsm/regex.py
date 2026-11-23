"""A small regular-expression engine: parse to an NFA, then run a lazy DFA.

The engine supports the subset of regular expressions needed to describe
structured output: literals, the wildcard ``.``, character classes ``[...]`` with
ranges and negation, the escapes ``\\d \\w \\s \\. \\" \\\\ \\n \\t``, the
quantifiers ``* + ?``, alternation ``|``, grouping ``(...)``, and concatenation.

A regular expression is parsed into a nondeterministic finite automaton (Thompson
construction) with epsilon transitions. Matching is then performed as a lazy
deterministic automaton: a DFA state is the epsilon-closed set of NFA states, and
transitions are computed on demand for the characters that actually appear. This
avoids enumerating the alphabet and is exactly what token-level masking needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class CharClass:
    """A predicate over single characters used on an NFA transition."""

    __slots__ = ("negated", "ranges", "singles")

    def __init__(self, singles: set[str], ranges: list[tuple[str, str]], negated: bool) -> None:
        self.singles = singles
        self.ranges = ranges
        self.negated = negated

    def matches(self, ch: str) -> bool:
        inside = ch in self.singles or any(lo <= ch <= hi for lo, hi in self.ranges)
        return inside != self.negated


def _literal(ch: str) -> CharClass:
    return CharClass({ch}, [], negated=False)


_ANY = CharClass(set(), [], negated=True)  # matches any single character


@dataclass
class _State:
    identifier: int
    epsilon: list[int] = field(default_factory=list)
    transitions: list[tuple[CharClass, int]] = field(default_factory=list)


class _Fragment:
    def __init__(self, start: int, out: int) -> None:
        self.start = start
        self.out = out


class RegexParseError(ValueError):
    """Raised when a pattern cannot be parsed."""


_ESCAPES = {
    "d": CharClass(set(), [("0", "9")], negated=False),
    "w": CharClass({"_"}, [("0", "9"), ("a", "z"), ("A", "Z")], negated=False),
    "s": CharClass({" ", "\t", "\n", "\r"}, [], negated=False),
    "n": _literal("\n"),
    "t": _literal("\t"),
}


class _Builder:
    """Parses a pattern into an NFA, returning the state table and start/accept."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.pos = 0
        self.states: list[_State] = []

    def _new_state(self) -> int:
        state = _State(len(self.states))
        self.states.append(state)
        return state.identifier

    def _peek(self) -> str | None:
        return self.pattern[self.pos] if self.pos < len(self.pattern) else None

    def _next(self) -> str:
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    def build(self) -> tuple[list[_State], int, int]:
        frag = self._parse_alternation()
        if self.pos != len(self.pattern):
            raise RegexParseError(f"unexpected character at {self.pos}")
        return self.states, frag.start, frag.out

    def _parse_alternation(self) -> _Fragment:
        branches = [self._parse_concat()]
        while self._peek() == "|":
            self._next()
            branches.append(self._parse_concat())
        if len(branches) == 1:
            return branches[0]
        start = self._new_state()
        out = self._new_state()
        for branch in branches:
            self.states[start].epsilon.append(branch.start)
            self.states[branch.out].epsilon.append(out)
        return _Fragment(start, out)

    def _parse_concat(self) -> _Fragment:
        parts: list[_Fragment] = []
        peek = self._peek()
        while peek is not None and peek not in "|)":
            parts.append(self._parse_quantified())
            peek = self._peek()
        if not parts:
            s = self._new_state()
            return _Fragment(s, s)
        from itertools import pairwise

        for left, right in pairwise(parts):
            self.states[left.out].epsilon.append(right.start)
        return _Fragment(parts[0].start, parts[-1].out)

    def _parse_quantified(self) -> _Fragment:
        atom = self._parse_atom()
        quant = self._peek()
        if quant == "*":
            self._next()
            return self._star(atom)
        if quant == "+":
            self._next()
            return self._plus(atom)
        if quant == "?":
            self._next()
            return self._optional(atom)
        return atom

    def _star(self, atom: _Fragment) -> _Fragment:
        start = self._new_state()
        out = self._new_state()
        self.states[start].epsilon.extend([atom.start, out])
        self.states[atom.out].epsilon.extend([atom.start, out])
        return _Fragment(start, out)

    def _plus(self, atom: _Fragment) -> _Fragment:
        out = self._new_state()
        self.states[atom.out].epsilon.extend([atom.start, out])
        return _Fragment(atom.start, out)

    def _optional(self, atom: _Fragment) -> _Fragment:
        start = self._new_state()
        out = self._new_state()
        self.states[start].epsilon.extend([atom.start, out])
        self.states[atom.out].epsilon.append(out)
        return _Fragment(start, out)

    def _atom(self, char_class: CharClass) -> _Fragment:
        start = self._new_state()
        out = self._new_state()
        self.states[start].transitions.append((char_class, out))
        return _Fragment(start, out)

    def _parse_atom(self) -> _Fragment:
        ch = self._peek()
        if ch is None:
            raise RegexParseError("unexpected end of pattern")
        if ch == "(":
            self._next()
            frag = self._parse_alternation()
            if self._peek() != ")":
                raise RegexParseError("missing closing parenthesis")
            self._next()
            return frag
        if ch == "[":
            return self._atom(self._parse_class())
        if ch == ".":
            self._next()
            return self._atom(_ANY)
        if ch == "\\":
            self._next()
            return self._atom(self._parse_escape())
        if ch in ")|*+?":
            raise RegexParseError(f"unexpected metacharacter {ch!r}")
        self._next()
        return self._atom(_literal(ch))

    def _parse_escape(self) -> CharClass:
        if self._peek() is None:
            raise RegexParseError("dangling escape")
        esc = self._next()
        if esc in _ESCAPES:
            return _ESCAPES[esc]
        return _literal(esc)

    def _parse_class(self) -> CharClass:
        self._next()  # consume '['
        negated = False
        if self._peek() == "^":
            self._next()
            negated = True
        singles: set[str] = set()
        ranges: list[tuple[str, str]] = []
        while self._peek() is not None and self._peek() != "]":
            ch = self._next()
            if ch == "\\":
                esc = self._next()
                if esc in _ESCAPES:
                    cls = _ESCAPES[esc]
                    singles |= cls.singles
                    ranges += cls.ranges
                    continue
                ch = {"n": "\n", "t": "\t"}.get(esc, esc)
            if (
                self._peek() == "-"
                and self.pos + 1 < len(self.pattern)
                and self.pattern[self.pos + 1] != "]"
            ):
                self._next()  # consume '-'
                hi = self._next()
                ranges.append((ch, hi))
            else:
                singles.add(ch)
        if self._peek() != "]":
            raise RegexParseError("missing closing bracket")
        self._next()
        return CharClass(singles, ranges, negated=negated)


class Regex:
    """A compiled regular expression exposed as a lazy deterministic automaton."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self._states, self._start, self._accept = _Builder(pattern).build()
        self._closure_cache: dict[frozenset[int], frozenset[int]] = {}

    def _epsilon_closure(self, states: frozenset[int]) -> frozenset[int]:
        cached = self._closure_cache.get(states)
        if cached is not None:
            return cached
        stack = list(states)
        seen = set(states)
        while stack:
            current = stack.pop()
            for target in self._states[current].epsilon:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        result = frozenset(seen)
        self._closure_cache[states] = result
        return result

    @property
    def start_state(self) -> frozenset[int]:
        """Return the initial DFA state (epsilon closure of the NFA start)."""

        return self._epsilon_closure(frozenset({self._start}))

    def step(self, state: frozenset[int], ch: str) -> frozenset[int]:
        """Return the DFA state after consuming ``ch`` (empty set means dead)."""

        nxt: set[int] = set()
        for nfa_state in state:
            for char_class, target in self._states[nfa_state].transitions:
                if char_class.matches(ch):
                    nxt.add(target)
        if not nxt:
            return frozenset()
        return self._epsilon_closure(frozenset(nxt))

    def is_accepting(self, state: frozenset[int]) -> bool:
        """Return whether ``state`` accepts (the NFA accept state is reachable)."""

        return self._accept in state

    @staticmethod
    def is_dead(state: frozenset[int]) -> bool:
        """Return whether ``state`` is dead (no NFA states, cannot continue)."""

        return len(state) == 0

    def fullmatch(self, text: str) -> bool:
        """Return whether ``text`` is fully matched by the pattern."""

        state = self.start_state
        for ch in text:
            state = self.step(state, ch)
            if self.is_dead(state):
                return False
        return self.is_accepting(state)
