import numpy as np
from wintermute.adversarial.environment import AsmMutationEnv, EnvConfig
from wintermute.adversarial.oracle import TieredOracle


class TestAsmMutationEnv:
    def setup_method(self):
        self.vocab = {
            "<PAD>": 0, "nop": 1, "push_eax": 2, "pop_eax": 3,
            "xor_eax_eax": 4, "sub_eax_eax": 5, "inc_eax": 6, "dec_eax": 7,
        }
        sample_pool = [
            (np.array([2, 4, 6, 2, 4, 6, 0, 0], dtype=np.int32), 1, "test_family"),
        ]
        oracle = TieredOracle(vocab_size=len(self.vocab))
        self.env = AsmMutationEnv(
            config=EnvConfig(max_seq_length=8, vocab_size=len(self.vocab), n_action_types=4),
            defender_fn=lambda tokens: 0.9,  # mock: always says malicious
            oracle_fn=lambda orig, mut: oracle.validate(orig, mut),
            sample_pool=sample_pool,
            vocab=self.vocab,
        )

    def test_reset_returns_correct_shape(self):
        obs, info = self.env.reset(seed=42)
        assert obs.shape == (8 + 8 + 4,)  # tokens + mask + meta

    def test_step_returns_five_tuple(self):
        self.env.reset(seed=42)
        result = self.env.step(np.array([0, 1]))  # NOP insertion at pos 1
        assert len(result) == 5  # obs, reward, terminated, truncated, info

    def test_episode_terminates(self):
        self.env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 100:
            obs, reward, terminated, truncated, info = self.env.step(
                np.array([0, steps % 6])
            )
            done = terminated or truncated
            steps += 1
        assert done
