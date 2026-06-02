import numpy as np

from openpi.models.value_function import (
    compute_advantage,
    compute_empirical_returns,
    compute_episode_rewards,
    return_to_bin,
)


def test_rewards_success_and_failure_terminal_values():
    success = compute_episode_rewards(success=True, episode_length=5, max_episode_length=10)
    failure = compute_episode_rewards(success=False, episode_length=5, max_episode_length=10)
    assert success[-1] == 0.0
    assert failure[-1] < success[-1]


def test_empirical_returns_are_monotonic_for_step_penalty():
    rewards = compute_episode_rewards(success=True, episode_length=5, max_episode_length=10)
    returns = compute_empirical_returns(rewards)
    assert returns[0] <= returns[-1]


def test_return_to_bin_clips_to_configured_range():
    bins = return_to_bin(np.array([-2.0, -0.5, 0.5]), num_bins=201, value_min=-1.0, value_max=0.0)
    assert bins[0] == 0
    assert bins[-1] == 200


def test_compute_advantage_shape():
    adv = compute_advantage(
        rewards_sum=np.array([-0.1, -0.2], dtype=np.float32),
        v_current=np.array([-0.5, -0.4], dtype=np.float32),
        v_next=np.array([-0.2, -0.1], dtype=np.float32),
    )
    assert adv.shape == (2,)
