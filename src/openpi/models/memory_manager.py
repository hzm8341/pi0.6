"""Language-memory management for π0.6-MEM.

Provides:
- ``MemoryLabel`` – dataclass for training annotations.
- ``MemoryDataGenerator`` – offline tool that calls an LLM to produce
  (m_t → m_{t+1}) training pairs from robot episode subtask sequences.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_GENERATION_PROMPT = """\
You are generating training labels for a robot memory system.

Given the completed subtasks listed below, write a **compressed** memory summary
that contains only information useful for *future* steps.

Rules:
1. Use first-person past tense ("I placed …", "I retrieved …").
2. Compress similar actions ("picked up bowl, plate" → "picked up the dishes").
3. OMIT failed attempts entirely.
4. Keep the summary under {max_length} characters.

Completed subtasks (successful only):
{history}

Compressed memory summary:"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MemoryLabel:
    """Single training annotation for the language-memory mechanism."""

    episode_id: str
    timestep: int
    subtask_instruction: str
    subtask_success: bool
    memory_before: str   # m_t  (input to high-level policy)
    memory_after: str    # m_{t+1} (target output)


@dataclasses.dataclass
class MemoryGenerationConfig:
    """Configuration for offline memory-label generation."""

    max_memory_length: int = 512   # max characters in generated summary
    temperature: float = 0.3
    max_tokens: int = 256


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MemoryDataGenerator:
    """Offline generator of language-memory training labels.

    Usage::

        generator = MemoryDataGenerator(llm_client)
        labels = generator.generate_labels_for_episode("ep001", subtasks)
        generator.save_labels(labels, "/path/to/labels.jsonl")

    ``llm_client`` must expose a ``generate(prompt, *, temperature, max_tokens)``
    method that returns a string.
    """

    def __init__(
        self,
        llm_client: Any,
        config: MemoryGenerationConfig | None = None,
    ) -> None:
        self.llm = llm_client
        self.config = config or MemoryGenerationConfig()

    # ------------------------------------------------------------------
    def generate_labels_for_episode(
        self,
        episode_id: str,
        subtasks: list[dict],
    ) -> list[MemoryLabel]:
        """Generate (m_t → m_{t+1}) pairs for every step of an episode.

        Args:
            episode_id: Unique string identifier for the episode.
            subtasks:   List of dicts with keys ``instruction`` (str) and
                        ``success`` (bool, default True).

        Returns:
            One ``MemoryLabel`` per subtask step.
        """
        labels: list[MemoryLabel] = []
        current_memory = ""

        for t, subtask in enumerate(subtasks):
            # Build history of *successful* steps up to (not including) step t
            successful = [
                f"Step {i + 1}: {s['instruction']}"
                for i, s in enumerate(subtasks[:t])
                if s.get("success", True)
            ]

            if not successful:
                new_memory = ""
            else:
                prompt = _GENERATION_PROMPT.format(
                    max_length=self.config.max_memory_length,
                    history="\n".join(successful),
                )
                new_memory = self.llm.generate(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ).strip()

            labels.append(
                MemoryLabel(
                    episode_id=episode_id,
                    timestep=t,
                    subtask_instruction=subtask["instruction"],
                    subtask_success=subtask.get("success", True),
                    memory_before=current_memory,
                    memory_after=new_memory,
                )
            )

            # Only advance memory on success (prevents distribution shift)
            if subtask.get("success", True):
                current_memory = new_memory

        return labels

    # ------------------------------------------------------------------
    def save_labels(self, labels: list[MemoryLabel], output_path: str) -> None:
        """Write labels to a JSON-Lines file."""
        with open(output_path, "w", encoding="utf-8") as fh:
            for label in labels:
                fh.write(json.dumps(dataclasses.asdict(label), ensure_ascii=False) + "\n")
        logger.info("Saved %d memory labels to %s", len(labels), output_path)

    @staticmethod
    def load_labels(input_path: str) -> list[MemoryLabel]:
        """Load labels from a JSON-Lines file."""
        labels: list[MemoryLabel] = []
        with open(input_path, encoding="utf-8") as fh:
            for line in fh:
                labels.append(MemoryLabel(**json.loads(line.strip())))
        return labels
