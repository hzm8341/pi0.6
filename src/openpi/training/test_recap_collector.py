import numpy as np

from openpi.training.recap_collector import ReCAPEpisode, assign_advantage_labels


def test_assign_advantage_labels_marks_positive_above_quantile():
    episode = ReCAPEpisode(
        observations=[{"state": np.array([0.0])}, {"state": np.array([1.0])}],
        actions=[np.array([0.0]), np.array([1.0])],
        rewards=np.array([-0.2, 0.0], dtype=np.float32),
        success=True,
        is_human_intervention=np.array([False, True]),
    )
    labeled = assign_advantage_labels([episode], advantages=[np.array([-0.5, 0.5])], positive_fraction=0.5)
    assert labeled[0].advantage_indicator.tolist() == [False, True]


def test_assign_advantage_labels_treats_positive_fraction_as_target_fraction():
    episode = ReCAPEpisode(
        observations=[{"state": np.array([idx])} for idx in range(5)],
        actions=[np.array([idx]) for idx in range(5)],
        rewards=np.zeros(5, dtype=np.float32),
        success=True,
        is_human_intervention=np.zeros(5, dtype=bool),
    )

    labeled = assign_advantage_labels(
        [episode],
        advantages=[np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)],
        positive_fraction=0.4,
    )

    assert labeled[0].advantage_indicator.tolist() == [False, False, False, True, True]


def test_assign_advantage_labels_forces_human_interventions_positive_and_records_source():
    episode = ReCAPEpisode(
        observations=[{"state": np.array([idx])} for idx in range(4)],
        actions=[np.array([idx]) for idx in range(4)],
        rewards=np.zeros(4, dtype=np.float32),
        success=False,
        is_human_intervention=np.array([False, True, False, False]),
    )

    labeled = assign_advantage_labels(
        [episode],
        advantages=[np.array([0.0, -10.0, 2.0, 3.0], dtype=np.float32)],
        positive_fraction=0.25,
    )

    assert labeled[0].advantage_indicator.tolist() == [False, True, False, True]
    assert labeled[0].label_source.tolist() == ["negative", "human", "negative", "value"]
