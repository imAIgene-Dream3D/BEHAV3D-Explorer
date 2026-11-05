"""
BEHAV3D assistant — the step plan pinned above the chat input (pure logic).

The assistant service is stateless and canned replies never reach the model, so
the dock remembers the steps a reply recommended and resends them with every
request. No Qt or network here; ``AssistantDock`` owns one :class:`PlanState`.

A reply declares a plan with a line reading ``**Plan**`` above a numbered list.
The model forgets that marker about half the time, but numbered lists without it
are often explanations. So an unmarked list only counts when it comes from the
model, has three or more items, and every item starts with a verb or a pipeline
module. Canned replies must always use the marker.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

MAX_PLAN_STEPS = 10
MAX_OUTCOMES = 6
_MIN_UNMARKED_STEPS = 3
_TITLE_LIMIT = 90
_DETAIL_LIMIT = 240
_SAME_PLAN_OVERLAP = 0.5

_PLAN_MARKER = re.compile(r"^\s*(?:#{1,4}\s+)?(?:\*\*)?plan:?(?:\*\*)?:?\s*$", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^ ?\d{1,2}[.)]\s+(\S.*)$")
_STEP_VERBS = frozenset({
    "add", "adjust", "apply", "build", "check", "choose", "compare", "configure",
    "confirm", "create", "define", "enter", "export", "extract", "fill", "filter",
    "go", "inspect", "load", "measure", "merge", "move", "navigate", "open", "pick",
    "proceed", "rename", "review", "run", "save", "segment", "select", "set", "start",
    "switch", "track", "train", "tune", "use", "verify",
})
_STEP_MODULES = (
    "active killing", "analysis", "backprojection", "behavioral state",
    "behavioural state", "data preparation", "death dynamics", "feature extraction",
    "filtering", "interaction", "invasiveness", "metadata", "population dynamics",
    "segmentation", "state trajectory", "tracking", "visualization",
)

_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_ORDINAL = re.compile(
    r"\b(?:step|task|item|number)\s*#?\s*(\d{1,2})\b|#(\d{1,2})\b|\bdo\s+(\d{1,2})\b|"
    rf"\b({'|'.join(_ORDINAL_WORDS)})\s+(?:one|step|thing|task|item|part)\b"
)
_ACK = (
    r"^(?:(?:ok(?:ay)?|yes|yep|yeah|great|good|nice|thanks|thank you|perfect|"
    r"alright|sure|cool)[\s,.!-]+)*"
)
_QUESTION = re.compile(
    _ACK + r"(?:what|which|why|how|when|where|who|is|are|does|explain|describe)\b"
)
_DONE = re.compile(
    _ACK + r"(?:(?:it'?s|that'?s|all|i'?m|we'?re|i am)\s+)?"
    r"(?:done|finished|completed)\b"
)
_NEXT = re.compile(
    _ACK + r"(?:let'?s\s+)?(?:(?:do|go to|start)\s+)?(?:the\s+)?"
    r"(?:next(?:\s+(?:step|one|task))?|continue|carry on|keep going|go on|move on|"
    r"proceed)\s*(?:please|then|thanks)?\s*[.!?]*$"
)
_NAVIGATION_VERB = re.compile(r"^(?:open|go to|navigate to|switch to)\s+")
_NAVIGATION_FILLER = frozenset({"the", "tab", "view", "panel", "module", "step"})


def _plain(text: str) -> str:
    """Drop inline Markdown so a step reads cleanly in a plain-text row."""
    without_links = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", str(text or ""))
    return " ".join(re.sub(r"[*`]", "", without_links).split())


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def step_title(item: str) -> str:
    """Return the short label for a list item, keeping its full text elsewhere."""
    lead = re.match(r"\*\*(.+?)\*\*\s*(?:[—–:-]\s|$)", item)
    if lead:
        return _shorten(_plain(lead.group(1)), _TITLE_LIMIT)
    sentence = re.split(r"(?<=[.!?])\s", _plain(item), maxsplit=1)[0].rstrip(".")
    return _shorten(re.split(r"\s[—–]\s", sentence, maxsplit=1)[0], _TITLE_LIMIT)


def _list_items(lines: list[str]) -> list[str]:
    """Collect the top-level items of the numbered list that starts ``lines``."""
    items: list[str] = []
    after_blank = False
    for line in lines:
        match = _LIST_ITEM.match(line)
        if match:
            items.append(match.group(1).strip())
            after_blank = False
        elif not line.strip():
            after_blank = True
        elif line[:1] in (" ", "\t"):
            continue  # nested list or wrapped text: belongs to the current item
        elif items and not after_blank:
            items[-1] += " " + line.strip()
        else:
            break
    return items


def _reads_as_step(item: str) -> bool:
    text = _plain(item).lower()
    words = text.split()
    return bool(words) and (
        words[0].strip(":,") in _STEP_VERBS or text.startswith(_STEP_MODULES)
    )


def extract_plan(
    text: str, *, canned: bool = False,
) -> Optional[tuple[list[str], bool]]:
    """Return ``(items, marked)`` when a reply declares a plan, otherwise ``None``."""
    lines = str(text or "").splitlines()
    marker = next(
        (index for index, line in enumerate(lines) if _PLAN_MARKER.match(line)), None
    )
    if marker is not None:
        rest = lines[marker + 1:]
        while rest and not rest[0].strip():
            rest = rest[1:]
        if rest and _LIST_ITEM.match(rest[0]):
            items = _list_items(rest)
            if 2 <= len(items) <= MAX_PLAN_STEPS:
                return items, True
        return None
    if canned:
        return None
    start = next(
        (index for index, line in enumerate(lines) if _LIST_ITEM.match(line)), None
    )
    if start is None:
        return None
    items = _list_items(lines[start:])
    if (
        _MIN_UNMARKED_STEPS <= len(items) <= MAX_PLAN_STEPS
        and all(_reads_as_step(item) for item in items)
    ):
        return items, False
    return None


def parse_plan_command(text: str) -> Optional[tuple[str, Optional[int]]]:
    """Read a typed message as ``("focus"|"next"|"done", step_index)`` or ``None``.

    Questions change nothing, including "what's next?": the model answers those
    from the plan it receives.
    """
    normalized = " ".join(str(text or "").lower().replace("’", "'").split())
    match = _ORDINAL.search(normalized)
    number = None
    if match:
        digits = match.group(1) or match.group(2) or match.group(3)
        number = int(digits) if digits else _ORDINAL_WORDS[match.group(4)]
    index = number - 1 if number else None
    if _DONE.match(normalized):
        return "done", index
    if _QUESTION.match(normalized):
        return None
    if index is not None:
        return "focus", index
    if _NEXT.match(normalized):
        return "next", None
    return None


def record_outcome(outcomes: list[str], text: str) -> list[str]:
    """Append what happened in the interface, keeping only the latest entries."""
    entry = " ".join(str(text or "").split())
    return ([*outcomes, entry] if entry else list(outcomes))[-MAX_OUTCOMES:]


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(first: str, second: str) -> float:
    a, b = _word_set(first), _word_set(second)
    return len(a & b) / len(a | b) if a | b else 0.0


@dataclass
class PlanStep:
    title: str
    detail: str
    done: bool = False


class PlanState:
    """The steps being worked through and which one is in focus."""

    def __init__(self):
        self.steps: list[PlanStep] = []
        self.focus = 0
        self.started = False    # the focus step has been begun, not merely listed
        self.applied = False    # a change was applied while on the focus step
        self.targeted = False   # the message being sent named a step
        self.offer: Optional[list[str]] = None

    @property
    def active(self) -> bool:
        return any(not step.done for step in self.steps)

    @property
    def done_count(self) -> int:
        return sum(step.done for step in self.steps)

    def clear(self):
        self.__init__()

    def status(self, index: int) -> str:
        if self.steps[index].done:
            return "done"
        return "current" if index == self.focus else "pending"

    def _replace(self, items: list[str]):
        self.clear()
        self.steps = [
            PlanStep(step_title(item), _shorten(_plain(item), _DETAIL_LIMIT))
            for item in items
        ]

    def _restates(self, items: list[str]) -> bool:
        titles = [step.title for step in self.steps]
        matches = sum(
            any(_overlap(step_title(item), title) >= _SAME_PLAN_OVERLAP for title in titles)
            for item in items
        )
        return matches >= len(items) * _SAME_PLAN_OVERLAP

    def adopt(self, items: list[str], marked: bool):
        """Take a plan from a reply. An unfinished plan is never replaced silently:
        a list given while working on a step is usually that step's sub-steps."""
        if not self.active:
            self._replace(items)
        elif marked and not self._restates(items):
            self.offer = items

    def accept_offer(self):
        if self.offer:
            self._replace(self.offer)

    def _lowest_pending(self) -> int:
        return next((i for i, step in enumerate(self.steps) if not step.done), 0)

    def _move_focus(self, index: int, *, started: bool):
        self.focus = index
        self.started = started
        self.applied = False

    def complete(self, index: Optional[int] = None):
        """Mark a step done, then focus the lowest step still pending."""
        index = self.focus if index is None else index
        if 0 <= index < len(self.steps):
            self.steps[index].done = True
            if index == self.focus:
                self._move_focus(self._lowest_pending(), started=False)

    def advance(self) -> Optional[int]:
        """Start the focus step, or finish it and start the next one.

        Returns the 1-based step now in progress, or ``None`` when none remain.
        """
        if self.active and self.started:
            self.complete()
        if not self.active:
            return None
        self._move_focus(self.focus, started=True)
        self.targeted = True
        return self.focus + 1

    def handle_message(self, text: str):
        """Update the plan from a typed message. Ordinals only move the focus."""
        if not self.steps:
            return
        command = parse_plan_command(text)
        if command is None:
            return
        name, index = command
        if name == "next":
            self.advance()
        elif index is not None and not 0 <= index < len(self.steps):
            return
        elif name == "done":
            self.complete(index)
            self.targeted = True
        else:
            self.steps[index].done = False
            self._move_focus(index, started=True)
            self.targeted = True

    def toggle(self, index: int):
        """Flip a row by hand; this is local and sends nothing to the model."""
        if self.steps[index].done:
            self.steps[index].done = False
            if self.steps[self.focus].done:
                self._move_focus(index, started=False)
        else:
            self.complete(index)

    def note_applied(self):
        self.started = True
        self.applied = True

    def note_navigation(self, label: str):
        """Follow an applied navigation, which also covers canned turns."""
        needle = " ".join(str(label or "").lower().split())
        order = [self.focus, *range(len(self.steps))] if self.steps else []
        for index in order:
            step = self.steps[index]
            if needle and not step.done and needle in step.detail.lower():
                self._move_focus(index, started=True)
                if self._is_navigation_only(step, needle):
                    self.complete(index)
                return

    @staticmethod
    def _is_navigation_only(step: PlanStep, needle: str) -> bool:
        title = step.title.lower()
        if not _NAVIGATION_VERB.match(title):
            return False
        rest = _NAVIGATION_VERB.sub("", title).replace(needle, " ")
        return not set(rest.split()) - _NAVIGATION_FILLER

    def action_label(self) -> str:
        """Name what the panel's button does next."""
        if not self.active:
            return ""
        if not self.started:
            return f"Start step {self.focus + 1}"
        following = next(
            (i for i, step in enumerate(self.steps) if not step.done and i != self.focus),
            None,
        )
        return "Finish" if following is None else f"Continue to step {following + 1}"

    def to_context(self) -> Optional[dict]:
        """Serialize for ``assistant_session.plan``."""
        if not self.steps:
            return None
        return {
            "steps": [
                {
                    "number": index + 1,
                    "title": step.title,
                    "detail": step.detail,
                    "status": self.status(index),
                }
                for index, step in enumerate(self.steps)
            ],
            "targeted": self.targeted,
        }

    def finish_request_setup(self):
        """Forget per-message state once the request has been serialized."""
        self.targeted = False
        self.offer = None
        if self.steps and not self.active:
            self.clear()
