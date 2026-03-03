#!/usr/bin/env python3
"""
Poetry Hub Agent
~~~~~~~~~~~~~~~~
A collaborative poetry agent that participates in the AI Poetry Hub game.
Designed to be deployed as a Railway worker.

Game flow (per round):
  1. Composition  — agents post 4 lines, one at a time
  2. Feedback     — agents post FEEDBACK: messages about the poem
  3. Finalization — the first-line agent posts FINAL: and calls /control/reset

Environment variables:
  OPENAI_API_KEY      Required. OpenAI API key for generating poetry.
  POET_ID             Optional. One of: shakespeare, dickinson, hughes, rumi,
                      basho, plath, neruda, angelou. Defaults to random.
  AGENT_NAME          Optional. Override the agent name sent on registration.
  AGENT_PROFILE       Optional. Override the one-sentence style description.
  HUB_URL             Optional. Hub base URL (default: production Railway URL).
"""

import os
import time
import random
import logging

import openai
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HUB_BASE_URL = os.getenv(
    "HUB_URL", "https://poetry-hub-production.up.railway.app"
).rstrip("/")

POETS: dict[str, dict[str, str]] = {
    "shakespeare": {
        "name": "shakespeare-agent",
        "profile": (
            "I write in the voice of William Shakespeare—iambic pentameter, "
            "rich metaphor, and Elizabethan diction."
        ),
    },
    "dickinson": {
        "name": "dickinson-agent",
        "profile": (
            "I write in the voice of Emily Dickinson—slant rhyme, compressed "
            "lyric intensity, and dashes that pause the breath."
        ),
    },
    "hughes": {
        "name": "hughes-agent",
        "profile": (
            "I write in the voice of Langston Hughes—jazz rhythms, vernacular "
            "speech, and themes of identity and resilience."
        ),
    },
    "rumi": {
        "name": "rumi-agent",
        "profile": (
            "I write in the voice of Rumi—mystical imagery, yearning for the "
            "divine beloved, and Sufi sensibility."
        ),
    },
    "basho": {
        "name": "basho-agent",
        "profile": (
            "I write in the voice of Matsuo Bashō—haiku-inspired brevity, "
            "vivid natural imagery, and seasonal awareness."
        ),
    },
    "plath": {
        "name": "plath-agent",
        "profile": (
            "I write in the voice of Sylvia Plath—confessional intensity, "
            "stark imagery, and psychological depth."
        ),
    },
    "neruda": {
        "name": "neruda-agent",
        "profile": (
            "I write in the voice of Pablo Neruda—sensuous imagery, elemental "
            "metaphors, and a passionate, elemental voice."
        ),
    },
    "angelou": {
        "name": "angelou-agent",
        "profile": (
            "I write in the voice of Maya Angelou—bold affirmation, resilience, "
            "and a celebration of human dignity."
        ),
    },
}

# Fallback lines (used when no Anthropic key is available)
FALLBACK_LINES = [
    "The moon casts silver light on sleeping stones",
    "In dreams we find the paths we dare not walk",
    "Time flows like rivers to the distant sea",
    "And whispers fade beneath the winter stars",
]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class PoetryAgent:
    def __init__(self) -> None:
        poet_id = os.getenv("POET_ID", random.choice(list(POETS.keys())))
        poet = POETS.get(poet_id, random.choice(list(POETS.values())))

        self.poet_id = poet_id
        self.agent_name: str = os.getenv("AGENT_NAME", poet["name"])
        self.agent_profile: str = os.getenv("AGENT_PROFILE", poet["profile"])

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.llm: openai.OpenAI | None = (
            openai.OpenAI(api_key=api_key, timeout=30.0) if api_key else None
        )
        if not self.llm:
            logger.warning("No OPENAI_API_KEY found — using fallback lines.")

        logger.info("Agent: %s (%s)", self.agent_name, self.poet_id)
        logger.info("Profile: %s", self.agent_profile)

    # ------------------------------------------------------------------
    # Hub API helpers
    # ------------------------------------------------------------------
    def _get(self, path: str, **kwargs) -> dict:
        r = requests.get(f"{HUB_BASE_URL}{path}", timeout=15, **kwargs)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: dict | None = None, **kwargs) -> dict:
        r = requests.post(
            f"{HUB_BASE_URL}{path}", json=json, timeout=15, **kwargs
        )
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {}

    def register(self) -> None:
        self._post(
            "/agents/register",
            json={"name": self.agent_name, "profile": self.agent_profile},
        )
        logger.info("Registered with hub as '%s'", self.agent_name)

    def get_state(self) -> dict:
        return self._get("/state")

    def post_line(self, text: str) -> None:
        self._post("/posts", json={"agent_name": self.agent_name, "text": text})
        logger.info("Posted: %s", text[:120])

    def control(self, action: str) -> None:
        self._post(f"/control/{action}")
        logger.info("Hub control: %s", action)

    def wait_for_hub_running(self) -> dict:
        """Block until is_running is True; return the state."""
        while True:
            state = self.get_state()
            if state.get("is_running"):
                return state
            logger.info("Hub not running — waiting 10 s…")
            time.sleep(10)

    # ------------------------------------------------------------------
    # Feed parsing
    # ------------------------------------------------------------------
    @staticmethod
    def parse_feed(feed: list[dict]):
        """
        Returns:
          poem_lines      — list of (agent_name, text) for non-FEEDBACK, non-FINAL posts
          feedback_msgs   — list of (agent_name, text) for FEEDBACK: posts
          first_line_agent — agent_name of the first poem line, or None
        """
        poem_lines: list[tuple[str, str]] = []
        feedback_msgs: list[tuple[str, str]] = []
        first_line_agent: str | None = None

        for post in feed:
            text = post.get("text", "")
            agent = post.get("agent_name", "")
            if text.startswith("FINAL:"):
                pass  # ignore for counting
            elif text.startswith("FEEDBACK:"):
                feedback_msgs.append((agent, text))
            else:
                poem_lines.append((agent, text))
                if first_line_agent is None:
                    first_line_agent = agent

        return poem_lines, feedback_msgs, first_line_agent

    @staticmethod
    def _agent_names(agents: list) -> list[str]:
        names: list[str] = []
        for a in agents:
            if isinstance(a, dict):
                names.append(a.get("name", ""))
            elif isinstance(a, str):
                names.append(a)
        return [n for n in names if n]

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------
    def _openai(self, prompt: str, max_tokens: int = 150) -> str:
        assert self.llm is not None
        resp = self.llm.chat.completions.create(
            model="gpt-4o",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    def generate_poem_line(self, existing: list[str], line_num: int) -> str:
        if not self.llm:
            return FALLBACK_LINES[(line_num - 1) % len(FALLBACK_LINES)]

        context = (
            "\n".join(existing) if existing else "(you are starting a brand-new poem)"
        )
        prompt = (
            f"You are {self.agent_name}, writing in the style of {self.poet_id}.\n"
            f"Your voice: {self.agent_profile}\n\n"
            f"You are collaborating on a four-line poem. Lines written so far:\n"
            f"{context}\n\n"
            f"Write EXACTLY ONE new line of poetry that continues this poem.\n"
            f"Rules:\n"
            f"- Match the thematic and stylistic direction of existing lines\n"
            f"- Keep your distinctive poetic voice\n"
            f"- Output a single line only — no line breaks, no quotes, no numbering\n"
            f"- Be evocative and original"
        )
        raw = self._openai(prompt, max_tokens=80)
        # Take only the first line in case the model drifts
        return raw.splitlines()[0].strip()

    def generate_feedback(self, poem_lines: list[str]) -> str:
        if not self.llm:
            return "FEEDBACK: A beautiful poem with evocative imagery and strong cohesion."

        poem_text = "\n".join(poem_lines)
        prompt = (
            f"You are {self.agent_name}, a poet reviewing a collaborative poem.\n"
            f"Your voice: {self.agent_profile}\n\n"
            f"The four-line poem:\n{poem_text}\n\n"
            f"Write 1–2 sentences of constructive feedback.\n"
            f"Start your response with 'FEEDBACK:' (no prefix, no line breaks).\n"
            f"Focus on imagery, theme, or suggest one specific improvement."
        )
        raw = self._openai(prompt, max_tokens=120)
        line = raw.splitlines()[0].strip()
        if not line.startswith("FEEDBACK:"):
            line = "FEEDBACK: " + line
        return line

    def generate_final_poem(
        self, poem_lines: list[str], feedback_msgs: list[str]
    ) -> str:
        if not self.llm:
            return "FINAL:\n" + "\n".join(poem_lines)

        poem_text = "\n".join(poem_lines)
        feedback_text = "\n".join(feedback_msgs) or "(no feedback received)"
        prompt = (
            f"You are {self.agent_name}, finalizing a collaborative poem.\n"
            f"Your voice: {self.agent_profile}\n\n"
            f"Original four-line poem:\n{poem_text}\n\n"
            f"Feedback from collaborators:\n{feedback_text}\n\n"
            f"Revise the poem, incorporating the feedback where it improves the work.\n"
            f"Output format — start with 'FINAL:' on its own line, "
            f"then exactly four lines, one per line:\n\n"
            f"FINAL:\n<line 1>\n<line 2>\n<line 3>\n<line 4>"
        )
        raw = self._openai(prompt, max_tokens=200)
        # Normalise: ensure it starts with FINAL:
        if "FINAL:" not in raw:
            lines = [l.strip() for l in raw.splitlines() if l.strip()][:4]
            return "FINAL:\n" + "\n".join(lines)
        idx = raw.index("FINAL:")
        return raw[idx:].strip()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.register()

        # feedback_phase_start: wall-clock time when we first detected 4 poem
        # lines in the current round (used to enforce the ~20 s fallback).
        feedback_phase_start: float | None = None

        while True:
            try:
                state = self.wait_for_hub_running()
                feed: list[dict] = state.get("posts", [])
                agents: list = state.get("agents", [])

                poem_lines, feedback_msgs, first_line_agent = self.parse_feed(feed)
                n = min(len(poem_lines), 4)  # clamp: >4 lines treated as 4 (composition over)
                last_author = feed[-1]["agent_name"] if feed else None
                i_wrote_first = first_line_agent == self.agent_name

                # ── Any FINAL: already posted? ──────────────────────────
                final_posted = any(
                    p.get("text", "").startswith("FINAL:") for p in feed
                )
                if final_posted:
                    logger.info("FINAL poem posted — waiting for reset…")
                    feedback_phase_start = None
                    time.sleep(5)
                    continue

                # ── COMPOSITION PHASE (< 4 poem lines) ─────────────────
                if n < 4:
                    feedback_phase_start = None  # reset for next feedback phase

                    # Don't respond to ourselves
                    if last_author == self.agent_name:
                        logger.info("My post was last — waiting 3 s…")
                        time.sleep(3)
                        continue

                    # Pacing between lines; re-fetch state to guard against races
                    time.sleep(2)
                    fresh_state = self.wait_for_hub_running()
                    fresh_lines, _, _ = self.parse_feed(fresh_state.get("posts", []))
                    if len(fresh_lines) >= 4:
                        continue  # another agent posted line 4 while we waited

                    new_line = self.generate_poem_line(
                        [t for _, t in poem_lines], n + 1
                    )
                    self.post_line(new_line)

                # ── FEEDBACK PHASE (exactly 4 poem lines) ───────────────
                elif n == 4:
                    if feedback_phase_start is None:
                        feedback_phase_start = time.time()

                    # Post my feedback if I haven't yet
                    my_feedbacks = [t for a, t in feedback_msgs if a == self.agent_name]
                    if not my_feedbacks:
                        time.sleep(2)
                        fresh_state = self.wait_for_hub_running()
                        # Guard: skip if FINAL: was posted while we waited
                        if any(
                            p.get("text", "").startswith("FINAL:")
                            for p in fresh_state.get("posts", [])
                        ):
                            continue
                        fb = self.generate_feedback([t for _, t in poem_lines])
                        self.post_line(fb)
                        continue

                    # If I wrote the first line, decide when to finalize
                    if i_wrote_first:
                        agent_names = self._agent_names(agents)
                        agents_with_feedback = {a for a, _ in feedback_msgs}
                        all_responded = set(agent_names) <= agents_with_feedback
                        elapsed = time.time() - feedback_phase_start

                        if all_responded or elapsed >= 20:
                            logger.info("Generating final poem (elapsed=%.0f s)…", elapsed)
                            time.sleep(2)
                            fresh_state = self.wait_for_hub_running()
                            # Guard: skip if another agent already posted FINAL:
                            if any(
                                p.get("text", "").startswith("FINAL:")
                                for p in fresh_state.get("posts", [])
                            ):
                                feedback_phase_start = None
                                continue

                            final = self.generate_final_poem(
                                [t for _, t in poem_lines],
                                [t for _, t in feedback_msgs],
                            )
                            self.post_line(final)

                            logger.info("Waiting 20 s before reset…")
                            time.sleep(20)
                            self.control("reset")
                            feedback_phase_start = None
                            logger.info("Round complete — starting new round.")
                        else:
                            logger.info(
                                "Waiting for all feedback (%.0f / 20 s)…", elapsed
                            )
                            time.sleep(5)
                    else:
                        # Not the first-line agent — wait, but take over if
                        # first-line agent appears dead (no FINAL: after 60 s)
                        elapsed = (
                            time.time() - feedback_phase_start
                            if feedback_phase_start is not None
                            else 0
                        )
                        if elapsed >= 60:
                            logger.info(
                                "First-line agent appears dead (%.0f s) — taking over finalization…",
                                elapsed,
                            )
                            time.sleep(2)
                            fresh_state = self.wait_for_hub_running()
                            fresh_posts = fresh_state.get("posts", [])
                            if any(
                                p.get("text", "").startswith("FINAL:")
                                for p in fresh_posts
                            ):
                                feedback_phase_start = None
                                continue  # first-line agent recovered; wait for reset
                            final = self.generate_final_poem(
                                [t for _, t in poem_lines],
                                [t for _, t in feedback_msgs],
                            )
                            self.post_line(final)
                            logger.info("Backup finalizer: waiting 20 s before reset…")
                            time.sleep(20)
                            self.control("reset")
                            feedback_phase_start = None
                            logger.info("Backup round complete — starting new round.")
                        else:
                            time.sleep(5)

                else:
                    # Unexpected state
                    logger.warning("Unexpected poem_lines count: %d — waiting…", n)
                    time.sleep(5)

            except requests.RequestException as exc:
                logger.error("Network error: %s — retrying in 5 s", exc)
                time.sleep(5)
            except Exception as exc:
                logger.error("Unexpected error: %s", exc, exc_info=True)
                time.sleep(5)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PoetryAgent().run()
