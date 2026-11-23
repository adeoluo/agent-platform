"""Deterministic synthetic multi-session dialogue dataset with temporal ground truth.

Each synthetic user holds preferences on several topics (editor, drink, cuisine, ...), a home
city, and an employer. Over a sequence of sessions spaced one to four days apart the user states
these facts, later changes some of them (sometimes twice), and also produces realistic
distractors: other people's preferences, questions that mention candidate values, tentative
plans that are not commitments, and small talk. An assistant turn follows every user turn.

Questions ask for the user's current value of an attribute (which must reflect every update) or
the value held before the most recent change (which requires temporal history). Some questions
use paraphrases of the topic name ("coding language" for "programming language") so lexical
overlap with the evidence is not guaranteed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

TOPICS: dict[str, tuple[str, ...]] = {
    "editor": ("Vim", "Neovim", "Emacs", "VS Code", "Sublime Text", "Helix", "Zed"),
    "drink": ("green tea", "black coffee", "espresso", "oat latte", "matcha", "kombucha"),
    "programming language": ("Python", "Rust", "Go", "TypeScript", "Kotlin", "Haskell"),
    "music genre": ("jazz", "techno", "indie rock", "classical", "hip hop", "ambient"),
    "cuisine": ("Thai", "Ethiopian", "Mexican", "Japanese", "Lebanese", "Peruvian"),
    "sport": ("climbing", "tennis", "running", "swimming", "cycling", "basketball"),
    "operating system": ("Linux", "macOS", "Windows", "FreeBSD"),
    "note-taking app": ("Obsidian", "Notion", "Logseq", "Apple Notes", "Bear"),
}

TOPIC_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "editor": ("text editor", "code editor"),
    "drink": ("beverage", "drink of choice"),
    "programming language": ("coding language", "language for programming"),
    "music genre": ("kind of music", "style of music"),
    "cuisine": ("type of food", "food"),
    "sport": ("way to exercise", "physical activity"),
    "operating system": ("computer platform", "operating system"),
    "note-taking app": ("notes tool", "app for notes"),
}

CITIES = ("Lisbon", "Tokyo", "Nairobi", "Toronto", "Berlin", "Austin", "Seoul", "Melbourne")
COMPANIES = ("Acme Robotics", "Globex", "Initech", "Umbrella Labs", "Hooli", "Stark Industries")

INITIAL_TEMPLATES = (
    "My favorite {topic} is {value}.",
    "I prefer {value} for my {topic}.",
    "For {topic}, I usually go with {value}.",
)
UPDATE_TEMPLATES = (
    "Actually, I switched to {value} for my {topic}.",
    "These days my favorite {topic} is {value}.",
    "My new favorite {topic} is {value}.",
    "For {topic}, I have now settled on {value}.",
)
DISTRACTOR_TEMPLATES = (
    "My coworker keeps telling me {value} is the best {topic}.",
    "What do you think about {value} as a {topic}?",
    "I might try {value} as a {topic} at some point, not sure yet.",
    "My sister swears by {value} for her {topic}.",
    "I read an article comparing {value} with other options for {topic}.",
)
SMALL_TALK = (
    "The weather has been strange this week.",
    "Can you help me draft an email to my landlord?",
    "I have a long day of meetings tomorrow.",
    "Remind me how compound interest works.",
    "I finally finished that book I was reading.",
    "What is a good way to stay focused in the afternoon?",
    "My team shipped a big release yesterday.",
    "Can you summarize the news about electric cars?",
)
ASSISTANT_REPLIES = (
    "Thanks for sharing that.",
    "Got it. Let me know if you want suggestions.",
    "That sounds good. Anything else on your mind?",
    "Understood. I can help with that.",
)


@dataclass(frozen=True)
class Turn:
    user_id: str
    session_index: int
    session_id: str
    timestamp: float
    role: str
    content: str


@dataclass(frozen=True)
class Question:
    user_id: str
    text: str
    predicate: str
    mode: str
    answer: str
    candidates: tuple[str, ...]
    paraphrased: bool


@dataclass
class UserLog:
    user_id: str
    turns: list[Turn] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    end_time: float = 0.0


@dataclass(frozen=True)
class Dataset:
    users: tuple[UserLog, ...]
    seed: int

    @property
    def questions(self) -> list[Question]:
        return [question for user in self.users for question in user.questions]

    @property
    def turns(self) -> list[Turn]:
        return [turn for user in self.users for turn in user.turns]


START_TIME = 1_700_000_000.0
DAY = 86_400.0


def _value_sequence(
    rng: random.Random, values: tuple[str, ...], sessions: int
) -> list[tuple[int, str]]:
    """Session indices and values for one attribute: an initial value and up to two changes."""
    chosen = rng.sample(values, k=min(3, len(values)))
    changes = 0
    roll = rng.random()
    if roll < 0.35:
        changes = 1
    elif roll < 0.6:
        changes = 2
    first = rng.randrange(0, max(1, sessions // 3))
    points = sorted(rng.sample(range(first + 1, sessions), k=min(changes, sessions - first - 1)))
    return [(first, chosen[0])] + [(point, chosen[i + 1]) for i, point in enumerate(points)]


def generate_user(
    rng: random.Random, user_id: str, sessions: int, topics_per_user: int = 6
) -> UserLog:
    log = UserLog(user_id)
    topics = rng.sample(sorted(TOPICS), k=topics_per_user)
    plan: dict[int, list[str]] = {index: [] for index in range(sessions)}
    timelines: dict[str, list[tuple[int, str]]] = {}

    for topic in topics:
        timeline = _value_sequence(rng, TOPICS[topic], sessions)
        timelines[f"prefers:{topic}"] = timeline
        for order, (session, value) in enumerate(timeline):
            template = rng.choice(INITIAL_TEMPLATES if order == 0 else UPDATE_TEMPLATES)
            plan[session].append(template.format(topic=topic, value=value))

    for predicate, pool, initial, update in (
        ("lives_in", CITIES, "I live in {value}.", "I just moved to {value}."),
        ("works_at", COMPANIES, "I work at {value}.", "I just joined {value}."),
    ):
        timeline = _value_sequence(rng, pool, sessions)
        timelines[predicate] = timeline
        for order, (session, value) in enumerate(timeline):
            plan[session].append((initial if order == 0 else update).format(value=value))

    clock = START_TIME + rng.randrange(0, 5) * DAY
    for session in range(sessions):
        clock += rng.randint(1, 4) * DAY
        user_turns = list(plan[session])
        for _ in range(rng.randint(2, 4)):
            topic = rng.choice(sorted(TOPICS))
            value = rng.choice(TOPICS[topic])
            user_turns.append(rng.choice(DISTRACTOR_TEMPLATES).format(topic=topic, value=value))
        user_turns.extend(rng.sample(SMALL_TALK, k=rng.randint(1, 3)))
        if rng.random() < 0.4:
            user_turns.append(f"My friend from {rng.choice(CITIES)} is visiting next month.")
        rng.shuffle(user_turns)
        session_id = f"{user_id}-s{session:03d}"
        moment = clock
        for content in user_turns:
            log.turns.append(Turn(user_id, session, session_id, moment, "user", content))
            moment += 45.0
            reply = rng.choice(ASSISTANT_REPLIES)
            log.turns.append(Turn(user_id, session, session_id, moment, "assistant", reply))
            moment += 45.0
    log.end_time = clock + DAY

    for predicate, timeline in timelines.items():
        values = [value for _, value in timeline]
        candidates = _candidates(predicate)
        log.questions.append(_question(rng, user_id, predicate, "current", values[-1], candidates))
        if len(values) > 1:
            log.questions.append(
                _question(rng, user_id, predicate, "previous", values[-2], candidates)
            )
    return log


def _candidates(predicate: str) -> tuple[str, ...]:
    if predicate == "lives_in":
        return CITIES
    if predicate == "works_at":
        return COMPANIES
    return TOPICS[predicate.split(":", 1)[1]]


def _question(
    rng: random.Random,
    user_id: str,
    predicate: str,
    mode: str,
    answer: str,
    candidates: tuple[str, ...],
) -> Question:
    paraphrased = False
    if predicate == "lives_in":
        text = (
            "Where does the user live now?"
            if mode == "current"
            else "Where did the user live before the most recent move?"
        )
    elif predicate == "works_at":
        text = (
            "Where does the user work now?"
            if mode == "current"
            else "Where did the user work before the most recent job change?"
        )
    else:
        topic = predicate.split(":", 1)[1]
        name = topic
        if rng.random() < 0.4:
            name = rng.choice(TOPIC_PARAPHRASES[topic])
            paraphrased = name != topic
        text = (
            f"What is the user's current preferred {name}?"
            if mode == "current"
            else f"Which {name} did the user prefer before the most recent change?"
        )
    return Question(user_id, text, predicate, mode, answer, candidates, paraphrased)


def generate_dataset(users: int = 12, sessions: int = 20, seed: int = 7) -> Dataset:
    rng = random.Random(seed)
    logs = tuple(generate_user(rng, f"user{index:02d}", sessions) for index in range(users))
    return Dataset(logs, seed)
