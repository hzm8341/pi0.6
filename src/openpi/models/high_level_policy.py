"""High-level policy (πHL) for π0.6-MEM.

πHL runs at a low frequency (once per subtask) and is responsible for:
  1. Selecting the next subtask instruction.
  2. Updating the compressed language memory.

The policy uses the PaliGemma VLM with a chain-of-thought JSON prompt that
outputs both a subtask and the updated memory in a single call.

Key design decision (from the MEM paper):
  - On subtask *failure* the memory is NOT updated.  This prevents the
    training-inference distribution shift caused by repeated failed attempts
    cluttering the context.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_HL_PROMPT_TEMPLATE = """\
You are controlling a robot to complete a task.

Task goal: {task_goal}

Memory of completed steps so far:
{language_memory}

Based on the current camera image and the memory above, determine:
1. The NEXT subtask for the robot to execute (a short, concrete action).
2. An UPDATED memory that captures everything completed so far (compress where
   possible, omit failed attempts).

Respond with valid JSON only – no markdown fences:
{{
  "subtask": "<next robot action, e.g. pick up the bowl>",
  "updated_memory": "<compressed past-tense summary of completed steps>"
}}"""

_COMPRESSION_RULES = """
Memory compression rules:
- Keep only information needed for future steps.
- Merge similar successive actions (e.g. "grabbed bowl, grabbed plate" →
  "grabbed the dishes").
- Do NOT record failed attempts.
- Use first-person past tense.
- Stay under 200 words.
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HighLevelPolicyConfig:
    """Configuration for the high-level policy."""

    # Trigger a πHL update every N low-level steps.
    subtask_trigger_steps: int = 50
    # Also trigger immediately when a subtask-completion signal is received.
    use_completion_detection: bool = True
    # Maximum tokens allocated to the subtask output.
    max_subtask_tokens: int = 64
    # Maximum tokens allocated to the updated memory output.
    max_memory_tokens: int = 256


# ---------------------------------------------------------------------------
# HighLevelPolicy
# ---------------------------------------------------------------------------

class HighLevelPolicy:
    """Manages subtask scheduling and language-memory updates at inference time.

    Parameters
    ----------
    vlm_inference_fn:
        Callable with signature ``(image, prompt, max_tokens) -> str``.
        ``image`` is a uint8 numpy array of shape ``(H, W, 3)``.
    tokenizer:
        Object with an ``encode(text: str) -> list[int]`` method.
    config:
        ``HighLevelPolicyConfig`` instance.
    """

    def __init__(
        self,
        vlm_inference_fn: Callable[[np.ndarray, str, int], str],
        tokenizer: Any,
        config: HighLevelPolicyConfig | None = None,
    ) -> None:
        self._vlm = vlm_inference_fn
        self._tokenizer = tokenizer
        self.config = config or HighLevelPolicyConfig()

        # Runtime state (reset at the start of each episode)
        self._task_goal: str = ""
        self._language_memory: str = ""
        self._current_subtask: str = ""
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, task_goal: str) -> None:
        """Start a new episode."""
        self._task_goal = task_goal
        self._language_memory = ""
        self._current_subtask = ""
        self._step_count = 0
        logger.info("[HL] Reset.  Goal: %s", task_goal)

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------

    def should_update(self, subtask_completed: bool = False) -> bool:
        """Return True if πHL should run this step."""
        if subtask_completed and self.config.use_completion_detection:
            return True
        self._step_count += 1
        return (self._step_count % self.config.subtask_trigger_steps) == 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        observation_image: np.ndarray,
        subtask_success: bool = True,
    ) -> tuple[str, str]:
        """Query the VLM and update internal state.

        Parameters
        ----------
        observation_image:
            Current base-camera image, uint8 (H, W, 3).
        subtask_success:
            Whether the *previous* subtask completed successfully.
            If False the language memory is NOT updated (prevents distribution
            shift; see MEM paper Section III-B).

        Returns
        -------
        (new_subtask, current_memory)
        """
        prompt = (
            _HL_PROMPT_TEMPLATE.format(
                task_goal=self._task_goal,
                language_memory=self._language_memory or "(none yet)",
            )
            + _COMPRESSION_RULES
        )

        max_tokens = self.config.max_subtask_tokens + self.config.max_memory_tokens
        raw = self._vlm(observation_image, prompt, max_tokens)

        # Parse JSON output
        new_subtask = self._current_subtask
        new_memory = self._language_memory
        try:
            parsed = json.loads(raw.strip())
            new_subtask = parsed.get("subtask", new_subtask)
            new_memory = parsed.get("updated_memory", new_memory)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("[HL] Failed to parse VLM output (%s).  Raw: %.120s", exc, raw)

        # KEY: only advance memory on success
        if subtask_success:
            self._language_memory = new_memory
            logger.info("[HL] Memory updated: %.80s…", new_memory)
        else:
            logger.info("[HL] Subtask failed – memory NOT updated.")

        self._current_subtask = new_subtask
        logger.info("[HL] Next subtask: %s", new_subtask)

        return new_subtask, self._language_memory

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def tokenize_memory(self, memory: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Tokenise the language memory to fixed-length arrays.

        Returns
        -------
        token_ids : int32 array of shape (max_memory_tokens,)
        token_mask: bool  array of shape (max_memory_tokens,)
        """
        max_len = self.config.max_memory_tokens
        text = memory if memory is not None else self._language_memory

        if not text:
            return (
                np.zeros(max_len, dtype=np.int32),
                np.zeros(max_len, dtype=bool),
            )

        ids = self._tokenizer.encode(text)[:max_len]
        pad = max_len - len(ids)
        token_ids = np.array(ids + [0] * pad, dtype=np.int32)
        token_mask = np.array([True] * len(ids) + [False] * pad, dtype=bool)
        return token_ids, token_mask

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_subtask(self) -> str:
        return self._current_subtask

    @property
    def language_memory(self) -> str:
        return self._language_memory
