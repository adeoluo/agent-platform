"""Candidate extraction: turn a conversation turn into candidate facts and preferences.

Two extractors implement the :class:`Extractor` protocol:

* :class:`RuleExtractor` is the deterministic default. It matches first-person statements
  ("my favorite editor is Neovim", "I moved to Lisbon", "please answer in bullet points") and
  assigns a confidence from hedging and update cues. It is fast, offline, and predictable, but it
  only understands the phrasings its patterns cover.
* :class:`LLMExtractor` asks a language model (any :class:`agent_memory.provider.Provider`) for a
  JSON list of facts. Tests and continuous integration use the offline
  :class:`agent_memory.provider.FakeProvider`; production can use Claude.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_memory.provider import Provider
from agent_memory.text import normalize_value, split_sentences


@dataclass(frozen=True)
class CandidateFact:
    subject: str
    predicate: str
    object: str
    confidence: float


@dataclass(frozen=True)
class CandidatePreference:
    key: str
    value: str
    confidence: float


@dataclass
class Extraction:
    facts: list[CandidateFact] = field(default_factory=list)
    preferences: list[CandidatePreference] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.facts and not self.preferences


class Extractor(Protocol):
    """Extracts candidate memories from one turn of text spoken by ``role``."""

    def extract(self, text: str, role: str = "user") -> Extraction:
        """Return candidate facts and preferences stated in ``text``."""


_HEDGES = re.compile(
    r"\b(i think|i guess|maybe|perhaps|probably|might|not sure|kind of|sort of)\b", re.I
)
_UPDATES = re.compile(
    r"\b(actually|now|nowadays|these days|switched|changed|anymore|no longer|instead|lately|"
    r"moved|relocated|just joined|new)\b",
    re.I,
)
_STOP = re.compile(
    r"\s+(?:for|as|because|since|these days|nowadays|now|lately|instead|anymore|too|though|"
    r"and|but|which|when|ever since|these|at the moment|right now)\b.*$",
    re.I,
)

_VALUE = r"(?P<value>[A-Za-z0-9][\w+#.'\- ]*?)"
_TOPIC = r"(?P<topic>[a-z][a-z\- ]{1,30}?)"
_END = r"(?=\s*(?:[,.!?;]|$|\s(?:now|these days|nowadays|lately|instead|anymore|too)\b))"

_PREFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bmy (?:new |current |all-time )?(?:favorite|favourite|go-to|preferred) "
        + _TOPIC
        + r" (?:is|has become|became|is now|these days is) "
        + _VALUE
        + _END,
        re.I,
    ),
    re.compile(
        r"\bi(?:'ve| have)? (?:now |really |actually |definitely |mostly |still )*"
        r"(?:prefer|like using|use|am using|'m using|switched to|moved over to|changed to|"
        r"went with|go with|settled on) " + _VALUE + r" (?:for|as) (?:my )?" + _TOPIC + _END,
        re.I,
    ),
    re.compile(
        r"\bfor (?:my )?"
        + _TOPIC
        + r",? i(?:'ve| have)? (?:now |mostly |usually |always |actually )*"
        r"(?:prefer|use|go with|choose|pick|switched to|settled on|reach for) " + _VALUE + _END,
        re.I,
    ),
)

_FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lives_in", re.compile(r"\bi (?:live|am living|'m living|reside) in " + _VALUE + _END, re.I)),
    (
        "lives_in",
        re.compile(r"\bi(?:'ve| have)? (?:just )?(?:moved|relocated) to " + _VALUE + _END, re.I),
    ),
    (
        "works_at",
        re.compile(
            r"\bi(?: now)? (?:work|am working|'m working) (?:at|for) " + _VALUE + _END, re.I
        ),
    ),
    (
        "works_at",
        re.compile(
            r"\bi(?:'ve| have)? (?:just )?(?:joined|started at|started working at) "
            + _VALUE
            + _END,
            re.I,
        ),
    ),
    ("name", re.compile(r"\bmy name is " + _VALUE + _END, re.I)),
    ("name", re.compile(r"\bcall me " + _VALUE + _END, re.I)),
    ("allergic_to", re.compile(r"\bi(?: am|'m) allergic to " + _VALUE + _END, re.I)),
    ("speaks", re.compile(r"\bi (?:speak|am fluent in|'m fluent in) " + _VALUE + _END, re.I)),
    (
        "has_pet",
        re.compile(
            r"\bi have an? (?P<pet>dog|cat|parrot|rabbit|hamster) named (?P<value>\w+)", re.I
        ),
    ),
)

_INSTRUCTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "response_style",
        re.compile(
            r"\b(?:please )?(?:always )?(?:answer|respond|reply|write back) (?:to me )?(?:in|with) "
            + _VALUE
            + _END,
            re.I,
        ),
    ),
    (
        "units",
        re.compile(r"\b(?:please )?(?:always )?use (?P<value>metric|imperial) units\b", re.I),
    ),
    ("address_as", re.compile(r"\bplease (?:address|refer to) me as " + _VALUE + _END, re.I)),
)


def _clean(value: str) -> str:
    trimmed = _STOP.sub("", value.strip())
    return trimmed.strip().strip(".,!?;:\"'").strip()


def _clean_topic(topic: str) -> str:
    cleaned = normalize_value(_STOP.sub("", topic))
    return re.sub(r"^(?:my|our)\s+", "", cleaned)


def _confidence(sentence: str) -> float:
    if _HEDGES.search(sentence):
        return 0.5
    if _UPDATES.search(sentence):
        return 0.95
    return 0.9


class RuleExtractor:
    """Deterministic first-person statement extractor (the default)."""

    def __init__(self, subject: str = "user") -> None:
        self.subject = subject

    def extract(self, text: str, role: str = "user") -> Extraction:
        extraction = Extraction()
        if role != "user":
            return extraction
        for sentence in split_sentences(text):
            confidence = _confidence(sentence)
            self._preferences(sentence, confidence, extraction)
            self._facts(sentence, confidence, extraction)
            self._instructions(sentence, confidence, extraction)
        return extraction

    def _preferences(self, sentence: str, confidence: float, out: Extraction) -> None:
        for pattern in _PREFERENCE_PATTERNS:
            for match in pattern.finditer(sentence):
                topic = _clean_topic(match.group("topic"))
                value = _clean(match.group("value"))
                if topic and value:
                    out.facts.append(
                        CandidateFact(self.subject, f"prefers:{topic}", value, confidence)
                    )

    def _facts(self, sentence: str, confidence: float, out: Extraction) -> None:
        for predicate, pattern in _FACT_PATTERNS:
            for match in pattern.finditer(sentence):
                value = _clean(match.group("value"))
                if predicate == "has_pet":
                    value = f"{match.group('pet').lower()} named {value}"
                if value:
                    out.facts.append(CandidateFact(self.subject, predicate, value, confidence))

    def _instructions(self, sentence: str, confidence: float, out: Extraction) -> None:
        for key, pattern in _INSTRUCTION_PATTERNS:
            for match in pattern.finditer(sentence):
                value = _clean(match.group("value"))
                if value:
                    out.preferences.append(CandidatePreference(key, value, confidence))


EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable memories about the user from one conversation turn. Return only JSON "
    'of the form {"facts": [{"subject": "user", "predicate": str, "object": str, '
    '"confidence": float}], "preferences": [{"key": str, "value": str, "confidence": float}]}. '
    'Use predicate "prefers:<topic>" for stated preferences, and "lives_in", "works_at", '
    '"name" for biographical facts. Use preferences for instructions about how the assistant '
    "should respond. Lower confidence for hedged statements. Return empty lists if nothing "
    "durable was said."
)

TURN_START = "<turn>"
TURN_END = "</turn>"


def build_extraction_prompt(text: str, role: str) -> str:
    return f"Role: {role}\n{TURN_START}{text}{TURN_END}"


def _as_confidence(raw: Any) -> float:
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return 0.5


def parse_extraction(payload: str) -> Extraction:
    """Parse a model's JSON reply; malformed or partial replies yield what can be recovered."""
    start, end = payload.find("{"), payload.rfind("}")
    if start < 0 or end <= start:
        return Extraction()
    try:
        data = json.loads(payload[start : end + 1])
    except json.JSONDecodeError:
        return Extraction()
    if not isinstance(data, dict):
        return Extraction()
    extraction = Extraction()
    for item in data.get("facts", []) or []:
        if not isinstance(item, dict):
            continue
        subject, predicate, obj = item.get("subject"), item.get("predicate"), item.get("object")
        if all(isinstance(part, str) and part.strip() for part in (subject, predicate, obj)):
            extraction.facts.append(
                CandidateFact(
                    str(subject).strip(),
                    str(predicate).strip().lower(),
                    str(obj).strip(),
                    _as_confidence(item.get("confidence", 0.8)),
                )
            )
    for item in data.get("preferences", []) or []:
        if not isinstance(item, dict):
            continue
        key, value = item.get("key"), item.get("value")
        if isinstance(key, str) and isinstance(value, str) and key.strip() and value.strip():
            extraction.preferences.append(
                CandidatePreference(
                    key.strip().lower(), value.strip(), _as_confidence(item.get("confidence", 0.8))
                )
            )
    return extraction


class LLMExtractor:
    """Extraction delegated to a language model behind the :class:`Provider` interface."""

    def __init__(self, provider: Provider, *, user_turns_only: bool = True) -> None:
        self.provider = provider
        self.user_turns_only = user_turns_only
        self.failures = 0

    def extract(self, text: str, role: str = "user") -> Extraction:
        if self.user_turns_only and role != "user":
            return Extraction()
        reply = self.provider.complete(
            EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt(text, role)
        )
        extraction = parse_extraction(reply)
        if extraction.empty and reply.strip() and '"facts"' not in reply:
            self.failures += 1
        return extraction


def rule_based_reply(system: str, prompt: str) -> str:
    """Answer an extraction prompt the way the rule extractor would, as JSON (offline)."""
    del system
    start, end = prompt.find(TURN_START), prompt.rfind(TURN_END)
    text = prompt[start + len(TURN_START) : end] if 0 <= start < end else prompt
    role = "user"
    if prompt.startswith("Role: "):
        role = prompt.split("\n", 1)[0].removeprefix("Role: ").strip()
    extraction = RuleExtractor().extract(text, role)
    return json.dumps(
        {
            "facts": [
                {
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "confidence": fact.confidence,
                }
                for fact in extraction.facts
            ],
            "preferences": [
                {"key": item.key, "value": item.value, "confidence": item.confidence}
                for item in extraction.preferences
            ],
        }
    )
