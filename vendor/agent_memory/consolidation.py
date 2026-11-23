"""Consolidation: summarize old episodes into semantic memory, then forget by decay.

A consolidation pass does three things for one namespace:

1. Selects unconsolidated episodes older than ``consolidate_after_days``, grouped by session.
2. For each session, writes an extractive summary as a semantic fact
   ``(session <id>, summary, <sentences>)`` and, when extraction did not run on write, extracts
   and merges facts from the episodes. The episodes are then marked consolidated.
3. Applies exponential-decay forgetting (see :class:`agent_memory.settings.ForgettingSettings`)
   to consolidated episodes, and optionally to superseded fact history.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

from agent_memory.embedding import Embedder
from agent_memory.models import (
    ConsolidationReport,
    Episode,
    Fact,
    Memory,
    MemoryKind,
    Status,
    render_fact,
)
from agent_memory.settings import DAY_SECONDS, ForgettingSettings, MemorySettings
from agent_memory.store import MemoryStore
from agent_memory.text import content_tokens, split_sentences


def retention(memory: Memory, now: float, settings: ForgettingSettings) -> float:
    """Decayed strength of a memory; it is forgotten when this drops below the threshold."""
    last = memory.last_accessed if memory.last_accessed is not None else memory.created_at
    idle = max(now - last, 0.0)
    decay = 2.0 ** (-idle / (settings.half_life_days * DAY_SECONDS))
    boost = 1.0 + settings.access_boost * math.log1p(memory.access_count)
    return float(memory.importance * decay * boost)


def summarize(episodes: Sequence[Episode], max_sentences: int) -> str:
    """Pick the most informative user sentences (inverse sentence frequency), in order."""
    sentences: list[tuple[int, str]] = []
    for episode in episodes:
        if episode.role != "user":
            continue
        sentences.extend(
            (len(sentences), sentence) for sentence in split_sentences(episode.content)
        )
    if not sentences:
        sentences = [
            (index, episode.content) for index, episode in enumerate(episodes) if episode.content
        ]
    if not sentences:
        return ""
    document_frequency: Counter[str] = Counter()
    tokenized = [set(content_tokens(sentence)) for _, sentence in sentences]
    for tokens in tokenized:
        document_frequency.update(tokens)
    total = len(sentences)
    scored = []
    for (position, sentence), tokens in zip(sentences, tokenized, strict=True):
        weight = sum(math.log(1.0 + total / document_frequency[token]) for token in tokens)
        scored.append((weight / math.sqrt(len(tokens) + 1.0), position, sentence))
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
    chosen = sorted(best, key=lambda item: item[1])
    return " ".join(sentence for _, _, sentence in chosen)


def run_consolidation(
    store: MemoryStore,
    namespace: str,
    settings: MemorySettings,
    embedder: Embedder,
    extract: Callable[[Episode], list[str]],
    now: float,
) -> ConsolidationReport:
    """Run one pass; ``extract`` merges facts from an episode when write-time extraction is off."""
    report = ConsolidationReport()

    cutoff = now - settings.consolidate_after_days * DAY_SECONDS
    sessions: dict[str, list[Episode]] = defaultdict(list)
    for episode in store.unconsolidated_episodes(namespace, cutoff):
        sessions[episode.session_id].append(episode)

    for session_id, episodes in sessions.items():
        if not settings.extract_on_write:
            for episode in episodes:
                outcome = extract(episode)
                report.facts_written += len(outcome)
        summary = summarize(episodes, settings.summary_sentences)
        if summary:
            text = render_fact(f"session {session_id}", "summary", summary)
            store.insert_fact(
                namespace,
                subject=f"session {session_id}",
                predicate="summary",
                obj=summary,
                text=text,
                confidence=1.0,
                created_at=episodes[-1].created_at,
                importance=sum(episode.importance for episode in episodes) / len(episodes),
                embedding=embedder.embed(text),
            )
            report.facts_written += 1
        store.mark_consolidated(namespace, [episode.id for episode in episodes])
        report.sessions_summarized += 1
        report.episodes_consolidated += len(episodes)
        report.details.append(f"session {session_id}: {len(episodes)} episodes summarized")

    forgetting = settings.forgetting
    if forgetting.enabled:
        doomed_episodes = [
            item.id
            for item in store.list_memories(namespace, kind=MemoryKind.EPISODIC)
            if isinstance(item, Episode)
            and item.consolidated
            and retention(item, now, forgetting) < forgetting.threshold
        ]
        report.episodes_forgotten = store.delete(namespace, doomed_episodes)
        if forgetting.forget_superseded_facts:
            history = store.list_memories(
                namespace,
                kind=MemoryKind.SEMANTIC,
                statuses=[Status.SUPERSEDED, Status.CONTESTED],
            )
            doomed_facts = [
                item.id
                for item in history
                if isinstance(item, Fact)
                and retention(item, now, forgetting) < forgetting.threshold
            ]
            report.facts_forgotten = store.delete(namespace, doomed_facts)
    return report
