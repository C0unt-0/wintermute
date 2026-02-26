import numpy as np
from wintermute.adversarial.oracle import TieredOracle


class TestTieredOracle:
    def setup_method(self):
        self.oracle = TieredOracle(vocab_size=100, pad_id=0)

    def test_valid_mutation(self):
        orig = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
        mut = np.array([1, 5, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
        assert self.oracle.validate(orig, mut).valid

    def test_rejects_out_of_range(self):
        orig = np.array([1, 2, 3, 0, 0], dtype=np.int32)
        mut = np.array([1, 999, 3, 0, 0], dtype=np.int32)
        assert not self.oracle.validate(orig, mut).valid

    def test_rejects_budget_exceeded(self):
        orig = np.ones(100, dtype=np.int32)
        mut = np.ones(100, dtype=np.int32) * 2  # 100% changed
        assert not self.oracle.validate(orig, mut).valid

    def test_rejects_all_padding(self):
        orig = np.array([1, 2, 3, 0, 0], dtype=np.int32)
        mut = np.zeros(5, dtype=np.int32)
        assert not self.oracle.validate(orig, mut).valid
