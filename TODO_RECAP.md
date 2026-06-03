# RECAP TODO

This file tracks the parts of the requested RECAP route that are not yet fully implemented.

## Offline RECAP

- [ ] Implement optional true LeRobot in-place dataset write-back.
  - Current status: `scripts/label_recap_advantage.py` writes sidecar files:
    `recap_labels.jsonl` and `lerobot_fields.npz`; `DataConfig.recap_fields_path`
    can merge `lerobot_fields.npz` into training samples at dataloader time.
  - Needed: optional permanent integration into the actual LeRobot dataset frame
    table or metadata format used by the target dataset.
- [ ] Train a learned lightweight value model.
  - Current status: offline labeling uses a deterministic progress-based value proxy.
  - Needed: add a small trainable value model that consumes state and/or image
    embeddings, trains with return targets, saves checkpoints, and supports
    batched inference.
- [ ] Add an evaluation harness comparing `pi05_recap` against normal `pi05` SFT.
  - Current status: `scripts/eval_recap_episodes.py` summarizes episode action/static
    statistics before training.
  - Needed metrics: model rollout success rate, average completion time, failure
    type counts, and label distribution.
- [ ] Connect `scripts/recap_train.py` to real subprocesses.
  - Current status: it logs the RECAP stages.
  - Needed: call label generation, VLA training, evaluation, and report generation.

## Rollout and HITL

- [ ] Implement `src/openpi/rollout/recap_env.py`.
  - Interface: `reset()`, `step(action)`, `success()`, `close()`.
- [ ] Implement `src/openpi/rollout/episode_recorder.py`.
  - Record observation, policy action, human action, executed action,
    success/failure, timeout, and intervention flags.
- [ ] Implement `src/openpi/rollout/intervention.py`.
  - Support keyboard, SpaceMouse, WebSocket, or platform-specific teleoperation callbacks.
- [ ] Implement `src/openpi/rollout/safety.py`.
  - Add velocity limits, joint/workspace limits, force thresholds, action
    smoothing, timeout stop, and emergency stop hooks.
- [ ] Implement success/failure labelers for the target task.
  - Start with manual labels, then add task-specific automatic checks where reliable.

## Paper-Level π0.6

- [ ] Upgrade the VLA backbone toward paper-level π0.6 scale.
  - Target: Gemma 3 4B base VLM and larger action expert.
  - This requires checkpoint availability, memory planning, training config
    changes, and validation on the target hardware.
- [ ] Add full KI joint objective coverage for RECAP.
  - Current status: RECAP loss wraps flow matching only.
  - Needed: include sub-task text prediction and FAST/discrete action likelihood where the model path supports them.
- [ ] Implement the full vision-language distributional value function.
  - Target: image + robot state + language input, 201 value bins,
    cross-entropy training, checkpoint save/load, and inference.
