import numpy as np
import pytest
from wintermute.adversarial.actions.code_actions import (
    apply_action,
    instruction_substitution,
)


@pytest.fixture(autouse=True)
def _clear_substitution_cache():
    """Clear the cached reverse vocab before each test to avoid stale state."""
    if hasattr(instruction_substitution, "_id_to_op"):
        del instruction_substitution._id_to_op


@pytest.fixture
def vocab():
    """Minimal vocab for testing mutations."""
    return {
        "<PAD>": 0, "nop": 1, "push_eax": 2, "pop_eax": 3,
        "mov_eax_eax": 4, "xor_eax_eax": 5, "sub_eax_eax": 6,
        "inc_eax": 7, "dec_eax": 8, "add_eax_1": 9,
        "xchg_eax_eax": 10, "push_ebx": 11, "pop_ebx": 12,
    }


class TestNopInsertion:
    def test_inserts_nop(self, vocab):
        tokens = np.array([2, 5, 7, 0, 0], dtype=np.int32)  # 3 real tokens + padding
        mutated, ok = apply_action(tokens, 0, 1, vocab)
        assert ok
        assert mutated[1] == vocab["nop"]
        assert mutated.shape == tokens.shape

    def test_rejects_padding_position(self, vocab):
        tokens = np.array([2, 5, 0, 0, 0], dtype=np.int32)
        _, ok = apply_action(tokens, 0, 4, vocab)
        assert not ok


class TestInstructionSubstitution:
    def test_swaps_xor_to_sub(self, vocab):
        tokens = np.array([2, 5, 7, 0, 0], dtype=np.int32)  # xor_eax_eax at pos 1
        mutated, ok = apply_action(tokens, 1, 1, vocab)
        assert ok
        assert mutated[1] == vocab["sub_eax_eax"]


class TestDeadCodeInjection:
    def test_inserts_push_pop_pair(self, vocab):
        tokens = np.array([2, 5, 7, 0, 0, 0, 0, 0], dtype=np.int32)
        mutated, ok = apply_action(tokens, 2, 1, vocab)
        assert ok
        # First matching pair is push_eax/pop_eax since both are in vocab
        assert mutated[1] == vocab["push_eax"]
        assert mutated[2] == vocab["pop_eax"]
