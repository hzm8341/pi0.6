import numpy as np

from openpi.models.model import Observation, preprocess_observation
from openpi.models.pi0_config import Pi0Config, ReCAPConfig
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.transforms import TokenizeReCAPAdvantage


def _data(batch: int = 2) -> dict:
    return {
        "image": {
            "base_0_rgb": np.zeros((batch, 224, 224, 3), dtype=np.float32),
            "left_wrist_0_rgb": np.zeros((batch, 224, 224, 3), dtype=np.float32),
            "right_wrist_0_rgb": np.zeros((batch, 224, 224, 3), dtype=np.float32),
        },
        "image_mask": {
            "base_0_rgb": np.ones(batch, dtype=bool),
            "left_wrist_0_rgb": np.ones(batch, dtype=bool),
            "right_wrist_0_rgb": np.ones(batch, dtype=bool),
        },
        "state": np.zeros((batch, 32), dtype=np.float32),
        "tokenized_prompt": np.zeros((batch, 48), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((batch, 48), dtype=bool),
    }


def test_recap_config_defaults_disabled():
    config = Pi0Config()
    assert isinstance(config.recap, ReCAPConfig)
    assert config.recap.enabled is False
    assert config.recap.advantage_dropout_prob == 0.1


def test_observation_parses_and_preprocesses_recap_fields():
    data = _data()
    data["advantage_indicator"] = np.array([True, False])
    data["use_advantage"] = np.array([True, True])
    data["is_human_intervention"] = np.array([False, True])
    data["tokenized_advantage_positive"] = np.ones((2, 4), dtype=np.int32)
    data["tokenized_advantage_negative"] = np.full((2, 4), 2, dtype=np.int32)
    data["tokenized_advantage_mask"] = np.ones((2, 4), dtype=bool)

    obs = Observation.from_dict(data)
    processed = preprocess_observation(None, obs, train=False)

    assert processed.advantage_indicator.shape == (2,)
    assert processed.use_advantage.shape == (2,)
    assert processed.is_human_intervention.shape == (2,)
    assert processed.tokenized_advantage_positive.shape == (2, 4)
    assert processed.tokenized_advantage_negative.shape == (2, 4)
    assert processed.tokenized_advantage_mask.shape == (2, 4)


def test_tokenize_recap_advantage_adds_fixed_token_fields():
    transform = TokenizeReCAPAdvantage(PaligemmaTokenizer(max_len=16), max_len=8)
    item = transform({"advantage_indicator": np.array(True)})

    assert item["tokenized_advantage_positive"].shape == (8,)
    assert item["tokenized_advantage_negative"].shape == (8,)
    assert item["tokenized_advantage_mask"].shape == (8,)
    assert item["use_advantage"].dtype == np.bool_
