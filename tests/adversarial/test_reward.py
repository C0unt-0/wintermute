from wintermute.adversarial.reward import compute_reward, RewardConfig

class TestReward:
    def test_evasion_gives_bonus(self):
        r = compute_reward(0.9, 0.3, oracle_valid=True, budget_remaining=0.8)
        assert r >= 10.0

    def test_oracle_rejection_is_negative(self):
        r = compute_reward(0.9, 0.9, oracle_valid=False, budget_remaining=1.0)
        assert r == -5.0

    def test_confidence_drop_is_positive(self):
        r = compute_reward(0.9, 0.7, oracle_valid=True, budget_remaining=0.9)
        assert r > 0

    def test_no_progress_is_negative(self):
        r = compute_reward(0.9, 0.91, oracle_valid=True, budget_remaining=0.9)
        assert r < 0
