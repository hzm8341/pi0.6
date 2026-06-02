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
    labeled = assign_advantage_labels([episode], advantages=[np.array([-0.5, 0.5])], positive_quantile=0.5)
    assert labeled[0].advantage_indicator.tolist() == [False, True]
