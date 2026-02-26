# IMPLEMENTATION_PLAN.md — Wintermute Phase 5: Adversarial Training Loop

# Optimized for Claude Code — sequential tasks, exact paths, concrete code

## CONTEXT

You are adding an adversarial training loop to Project Wintermute, an MLX-native
malware classifier. One model (attacker) mutates malware assembly to evade detection.
Another model (defender) tries to catch it. They train against each other.

**Hard constraint: ALL neural network code must use MLX. No PyTorch. No Stable-Baselines3.**

The PPO reinforcement learning agent, TRADES loss, EWC regularizer, and spectral
normalization are all implemented from scratch using `mlx.core`, `mlx.nn`, and
`mlx.optimizers`. The Gymnasium RL environment uses numpy. Conversion happens once
at the boundary via `mx.array(np_array)`.

**Existing code you MUST NOT modify** (unless noted):

- `src/wintermute/models/fusion.py` — `WintermuteMalwareDetector`, `DetectorConfig`
- `src/wintermute/models/transformer.py` — `MalBERTEncoder`, `MalBERTConfig`
- `src/wintermute/models/gat.py` — `GATEncoder`
- `src/wintermute/engine/trainer.py` — `Trainer`
- `src/wintermute/engine/pretrain.py` — `MLMPretrainer`
- `src/wintermute/engine/metrics.py` — `compute_macro_f1`
- `src/wintermute/data/tokenizer.py`
- `src/wintermute/data/augment.py` — `HeuristicAugmenter`, `SyntheticGenerator`

**Existing code you will EXTEND** (append to, don't rewrite):

- `src/wintermute/cli.py` — add `adv` subcommand group
- `src/wintermute/engine/joint_trainer.py` — add optional TRADES+EWC loss path
- `pyproject.toml` — add `[adversarial]` extras

**Existing patterns to follow** (copy these conventions exactly):

- Loss: `nn.value_and_grad(model, loss_fn)` → see `joint_trainer.py`
- Grad clip: `mlx.utils.tree_flatten(grads)` → norm → scale → see `joint_trainer.py`
- Optimizer: `optim.AdamW(learning_rate=...)` → see `joint_trainer.py`
- Materialize: `mx.eval(model.parameters(), optimizer.state)` after every update
- Config: `OmegaConf.create(DEFAULTS)` merged with YAML → see `joint_trainer.py`
- Tests: `pytest` classes with synthetic data → see `tests/test_pipeline.py`

---

## TASK 0: Add dependencies

**File:** `pyproject.toml`

Add a new optional dependency group after the existing `dev` group:

```toml
adversarial = [
    "gymnasium>=1.0.0",
    "lief>=0.15.0",
]
```

Also add it to the `all` group:

```toml
all = [
    "wintermute[api,mlops,dev,adversarial]",
    "angr>=9.2.0",
]
```

**Done when:** `pip install -e ".[adversarial]"` succeeds and `import gymnasium` works.

---

## TASK 1: Create the substitution table

**File:** `src/wintermute/adversarial/__init__.py` — empty file
**File:** `src/wintermute/adversarial/actions/__init__.py` — empty file
**File:** `src/wintermute/adversarial/actions/substitution_table.py`

This is a pure-data file. No imports beyond typing. It maps opcode mnemonics to
their semantic equivalents. The mutation engine uses this to swap instructions
without changing behavior.

```python
"""
Semantic equivalence table for x86/x64 instruction substitution.

Each entry: (original_mnemonic, replacement_sequence)
Both produce identical CPU state. Used by the mutation engine.
"""

# fmt: off
SUBSTITUTION_PAIRS: list[tuple[str, list[str]]] = [
    # Zero a register
    ("xor_eax_eax",   ["sub_eax_eax"]),
    ("sub_eax_eax",   ["xor_eax_eax"]),
    ("xor_ebx_ebx",   ["sub_ebx_ebx"]),
    ("sub_ebx_ebx",   ["xor_ebx_ebx"]),
    ("xor_ecx_ecx",   ["sub_ecx_ecx"]),
    ("sub_ecx_ecx",   ["xor_ecx_ecx"]),
    ("xor_edx_edx",   ["sub_edx_edx"]),
    ("sub_edx_edx",   ["xor_edx_edx"]),

    # Increment / decrement
    ("inc_eax",       ["add_eax_1"]),
    ("add_eax_1",     ["inc_eax"]),
    ("dec_eax",       ["sub_eax_1"]),
    ("sub_eax_1",     ["dec_eax"]),
    ("inc_ebx",       ["add_ebx_1"]),
    ("add_ebx_1",     ["inc_ebx"]),
    ("dec_ebx",       ["sub_ebx_1"]),
    ("sub_ebx_1",     ["dec_ebx"]),

    # Move zero
    ("mov_eax_0",     ["xor_eax_eax"]),
    ("mov_ebx_0",     ["xor_ebx_ebx"]),
    ("mov_ecx_0",     ["xor_ecx_ecx"]),

    # NOPs (semantic equivalents)
    ("nop",           ["xchg_eax_eax"]),
    ("xchg_eax_eax",  ["nop"]),
    ("mov_eax_eax",   ["nop"]),
    ("lea_eax_[eax]", ["nop"]),

    # Push-via-sub-mov vs push
    ("push_eax",      ["sub_esp_4", "mov_[esp]_eax"]),
    ("push_ebx",      ["sub_esp_4", "mov_[esp]_ebx"]),

    # Test-vs-and
    ("test_eax_eax",  ["and_eax_eax"]),
    ("and_eax_eax",   ["test_eax_eax"]),
    ("test_ebx_ebx",  ["and_ebx_ebx"]),

    # Comparison equivalents
    ("cmp_eax_0",     ["test_eax_eax"]),
    ("cmp_ebx_0",     ["test_ebx_ebx"]),
]
# fmt: on

# Build fast lookup: mnemonic_str → list of possible replacements
SUBSTITUTION_MAP: dict[str, list[list[str]]] = {}
for _orig, _repl in SUBSTITUTION_PAIRS:
    SUBSTITUTION_MAP.setdefault(_orig, []).append(_repl)
```

**Done when:** `python -c "from wintermute.adversarial.actions.substitution_table import SUBSTITUTION_MAP; print(len(SUBSTITUTION_MAP))"` prints a number > 20.

---

## TASK 2: Create the mutation actions

**File:** `src/wintermute/adversarial/actions/code_actions.py`

This file operates on **numpy int arrays** (token ID sequences). No MLX here.
The environment passes numpy arrays; mutations happen in numpy; results go back to the env.

Implement these 4 functions. Each takes `(tokens: np.ndarray, position: int, vocab: dict) -> tuple[np.ndarray, bool]`
where the return is `(mutated_tokens, success)`.

```python
"""
Functionality-preserving code mutations on token-ID sequences.

All functions operate on numpy arrays. No MLX.
Input:  tokens [T] int array of vocab IDs, target position, vocab dict
Output: (mutated_tokens [T], success bool)
"""

import numpy as np
from wintermute.adversarial.actions.substitution_table import SUBSTITUTION_MAP


def nop_insertion(tokens: np.ndarray, position: int, vocab: dict) -> tuple[np.ndarray, bool]:
    """Insert a NOP token at `position`, shifting subsequent tokens left (dropping last)."""
    nop_id = vocab.get("nop")
    if nop_id is None:
        return tokens, False
    pad_id = vocab.get("<PAD>", 0)
    # Don't insert into padding region
    non_pad = np.sum(tokens != pad_id)
    if position >= non_pad:
        return tokens, False
    mutated = tokens.copy()
    # Shift right from position, drop last non-pad token
    mutated[position + 1:non_pad] = tokens[position:non_pad - 1]
    mutated[position] = nop_id
    return mutated, True


def instruction_substitution(tokens: np.ndarray, position: int,
                              vocab: dict) -> tuple[np.ndarray, bool]:
    """Swap the token at `position` with a semantic equivalent from the substitution table."""
    # Build reverse vocab once (cache on function attribute)
    if not hasattr(instruction_substitution, "_id_to_op"):
        instruction_substitution._id_to_op = {v: k for k, v in vocab.items()}
    id_to_op = instruction_substitution._id_to_op

    token_id = int(tokens[position])
    mnemonic = id_to_op.get(token_id)
    if mnemonic is None or mnemonic not in SUBSTITUTION_MAP:
        return tokens, False

    replacements = SUBSTITUTION_MAP[mnemonic]
    # Pick a random replacement (single-token only for now)
    for repl in replacements:
        if len(repl) == 1 and repl[0] in vocab:
            mutated = tokens.copy()
            mutated[position] = vocab[repl[0]]
            return mutated, True

    return tokens, False


def dead_code_injection(tokens: np.ndarray, position: int,
                         vocab: dict) -> tuple[np.ndarray, bool]:
    """Insert a cancel-out pair (push/pop or inc/dec) at `position`."""
    pairs = [
        ("push_eax", "pop_eax"),
        ("push_ebx", "pop_ebx"),
        ("push_ecx", "pop_ecx"),
        ("inc_eax", "dec_eax"),
        ("inc_ebx", "dec_ebx"),
    ]
    pad_id = vocab.get("<PAD>", 0)
    non_pad = int(np.sum(tokens != pad_id))

    for a, b in pairs:
        if a in vocab and b in vocab:
            a_id, b_id = vocab[a], vocab[b]
            if position + 1 >= non_pad or non_pad + 2 > len(tokens):
                continue
            mutated = tokens.copy()
            # Shift right by 2 to make room, drop last 2 non-pad tokens
            end = min(non_pad + 2, len(tokens))
            mutated[position + 2:end] = tokens[position:end - 2]
            mutated[position] = a_id
            mutated[position + 1] = b_id
            return mutated, True

    return tokens, False


def register_reassignment(tokens: np.ndarray, position: int,
                           vocab: dict) -> tuple[np.ndarray, bool]:
    """
    Swap register references in a local window around `position`.

    Simple version: find tokens containing 'eax' in a ±5 window and swap to 'ecx'
    (or vice versa) if both variants exist in vocab. This is a conservative approximation
    — the full version would use liveness analysis from the CFG.
    """
    id_to_op = {v: k for k, v in vocab.items()}
    window = 5
    start = max(0, position - window)
    end = min(len(tokens), position + window)

    swap_pairs = [("eax", "ecx"), ("ebx", "edx")]
    mutated = tokens.copy()
    swapped = False

    for reg_a, reg_b in swap_pairs:
        for i in range(start, end):
            op = id_to_op.get(int(tokens[i]), "")
            if reg_a in op:
                new_op = op.replace(reg_a, reg_b)
                if new_op in vocab:
                    mutated[i] = vocab[new_op]
                    swapped = True
            elif reg_b in op:
                new_op = op.replace(reg_b, reg_a)
                if new_op in vocab:
                    mutated[i] = vocab[new_op]
                    swapped = True
        if swapped:
            break

    return mutated, swapped


# Action dispatch table — maps action_id to function
ACTION_FNS = {
    0: nop_insertion,
    1: instruction_substitution,
    2: dead_code_injection,
    3: register_reassignment,
}


def apply_action(tokens: np.ndarray, action_type: int, position: int,
                  vocab: dict) -> tuple[np.ndarray, bool]:
    """Apply a mutation action. Returns (mutated_tokens, success)."""
    fn = ACTION_FNS.get(action_type)
    if fn is None:
        return tokens, False
    position = max(0, min(position, len(tokens) - 1))
    return fn(tokens, position, vocab)
```

**Test file:** `tests/adversarial/__init__.py` — empty
**Test file:** `tests/adversarial/test_actions.py`

```python
import numpy as np
import pytest
from wintermute.adversarial.actions.code_actions import apply_action

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
        assert mutated[1] in (vocab["push_eax"], vocab["push_ebx"], vocab["inc_eax"], vocab["inc_ebx"])
```

**Done when:** `pytest tests/adversarial/test_actions.py -v` passes.

---

## TASK 3: Create the Oracle (Tier 1 only)

**File:** `src/wintermute/adversarial/oracle.py`

Tier 1 = structural validation only. No r2pipe, no sandbox. Fast enough for RL training loops.

```python
"""
oracle.py — Tiered verification of mutated samples.

Tier 1 (this task): token-level structural checks.
  - All token IDs within vocab bounds
  - Sequence length unchanged
  - At least 10% non-PAD tokens remain
  - Modification budget not exceeded

Tier 2 (future): CFG diff via r2pipe
Tier 3 (future): CAPE sandbox execution
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class OracleResult:
    valid: bool
    tier: int
    reason: str = ""


class TieredOracle:
    def __init__(self, vocab_size: int, pad_id: int = 0,
                 max_modification_ratio: float = 0.15):
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_mod_ratio = max_modification_ratio

    def validate(self, original_tokens: np.ndarray,
                 mutated_tokens: np.ndarray) -> OracleResult:
        """Run Tier 1 structural validation."""
        # Check 1: same shape
        if original_tokens.shape != mutated_tokens.shape:
            return OracleResult(False, 1, "shape_mismatch")

        # Check 2: all token IDs in vocab range
        if np.any(mutated_tokens < 0) or np.any(mutated_tokens >= self.vocab_size):
            return OracleResult(False, 1, "token_out_of_range")

        # Check 3: not all padding
        non_pad = np.sum(mutated_tokens != self.pad_id)
        total = len(mutated_tokens)
        if non_pad < total * 0.10:
            return OracleResult(False, 1, "too_few_tokens")

        # Check 4: modification budget
        changed = np.sum(original_tokens != mutated_tokens)
        if changed / max(total, 1) > self.max_mod_ratio:
            return OracleResult(False, 1, "budget_exceeded")

        return OracleResult(True, 1)
```

**Test file:** `tests/adversarial/test_oracle.py`

```python
import numpy as np
from wintermute.adversarial.oracle import TieredOracle

class TestTieredOracle:
    def setup_method(self):
        self.oracle = TieredOracle(vocab_size=100, pad_id=0)

    def test_valid_mutation(self):
        orig = np.array([1, 2, 3, 0, 0], dtype=np.int32)
        mut = np.array([1, 5, 3, 0, 0], dtype=np.int32)
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
```

**Done when:** `pytest tests/adversarial/test_oracle.py -v` passes.

---

## TASK 4: Create the shaped reward function

**File:** `src/wintermute/adversarial/reward.py`

Pure Python + numpy. No MLX.

```python
"""
reward.py — Shaped reward for the RL attacker agent.

Returns a float reward given the defender's confidence before and after mutation.
Replaces the original sparse {-10, -1, +10, +100} with a smooth gradient signal.
"""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    evasion_bonus: float = 10.0          # defender predicts benign (conf < 0.5)
    confidence_drop_scale: float = 5.0    # reward proportional to confidence drop
    detection_penalty: float = -0.5       # detected with high confidence
    oracle_rejection: float = -5.0        # mutation broke functionality
    step_penalty: float = -0.05           # small per-step cost
    budget_penalty_scale: float = 0.5     # penalize large modifications


def compute_reward(
    confidence_before: float,
    confidence_after: float,
    oracle_valid: bool,
    budget_remaining: float,
    config: RewardConfig | None = None,
) -> float:
    """Compute shaped reward for one mutation step."""
    cfg = config or RewardConfig()

    if not oracle_valid:
        return cfg.oracle_rejection

    # Total evasion: defender thinks it's benign
    if confidence_after < 0.5:
        return cfg.evasion_bonus

    # Confidence dropped meaningfully
    delta = confidence_before - confidence_after
    if delta > 0.01:
        reward = cfg.confidence_drop_scale * delta
        # Budget penalty: penalize mutations that change too much
        budget_used = 1.0 - budget_remaining
        reward -= cfg.budget_penalty_scale * budget_used
        return reward + cfg.step_penalty

    # No progress
    return cfg.detection_penalty + cfg.step_penalty
```

**Test file:** `tests/adversarial/test_reward.py`

```python
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
```

**Done when:** `pytest tests/adversarial/test_reward.py -v` passes.

---

## TASK 5: Create the defender bridge

**File:** `src/wintermute/adversarial/bridge.py`

This is the ONE place where numpy ↔ MLX conversion happens. The environment
gives numpy token arrays; this bridge runs the MLX model and returns a float.

```python
"""
bridge.py — Connects the Gymnasium env (numpy) to the MLX defender model.

Single conversion point: np.ndarray → mx.array → model forward → float
"""

import mlx.core as mx
import numpy as np
from wintermute.models.fusion import WintermuteMalwareDetector


class DefenderBridge:
    """
    Wraps WintermuteMalwareDetector for use by the RL environment.

    Call signature matches what AsmMutationEnv expects:
        confidence: float = bridge(tokens: np.ndarray)
    """

    def __init__(self, model: WintermuteMalwareDetector):
        self.model = model
        self.model.eval()

    def __call__(self, tokens: np.ndarray) -> float:
        """
        Run inference on a single token sequence.

        Args:
            tokens: [T] numpy int32 array of vocab IDs

        Returns:
            float — P(malicious), between 0.0 and 1.0
        """
        x = mx.array(tokens[np.newaxis, :], dtype=mx.int32)  # [1, T]
        logits = self.model(x)                                 # [1, num_classes]
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)

        if probs.shape[1] == 2:
            # Binary: index 1 = malicious
            return float(probs[0, 1].item())
        else:
            # Multi-class: 1 - P(benign), assuming class 0 = benign
            return float(1.0 - probs[0, 0].item())
```

**Done when:** Can be imported without error: `python -c "from wintermute.adversarial.bridge import DefenderBridge"`

---

## TASK 6: Create the Gymnasium environment

**File:** `src/wintermute/adversarial/environment.py`

Uses: numpy, gymnasium. Does NOT import MLX — it calls `defender_fn` and `oracle_fn`
which are injected as callables.

```python
"""
environment.py — Gymnasium environment for adversarial malware mutation.

Framework-agnostic: all observations and actions are numpy arrays.
The defender model is accessed through an injected callable (DefenderBridge).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from wintermute.adversarial.actions.code_actions import apply_action
from wintermute.adversarial.reward import compute_reward, RewardConfig


@dataclass
class EnvConfig:
    max_steps: int = 25
    modification_budget: float = 0.15
    vocab_size: int = 287
    max_seq_length: int = 2048
    n_action_types: int = 4       # start with 4 code-level actions
    pad_id: int = 0


class AsmMutationEnv(gym.Env):
    """
    RL environment: mutate malware token sequences to evade detection.

    Action space: MultiDiscrete([n_action_types, max_seq_length])
    Observation space: Box — flattened (normalized_tokens + mod_mask + meta)
    """
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig, defender_fn, oracle_fn,
                 sample_pool: list, vocab: dict,
                 reward_config: RewardConfig | None = None):
        """
        Args:
            config: environment parameters
            defender_fn: callable(np.ndarray) -> float (confidence)
            oracle_fn: callable(np.ndarray, np.ndarray) -> OracleResult
            sample_pool: list of (tokens_np, label_int, family_str)
            vocab: dict mapping opcode strings to int IDs
            reward_config: reward shaping parameters
        """
        super().__init__()
        self.cfg = config
        self.defender_fn = defender_fn
        self.oracle_fn = oracle_fn
        self.sample_pool = sample_pool
        self.vocab = vocab
        self.reward_config = reward_config or RewardConfig()

        T = config.max_seq_length
        # Observation: normalized tokens + modification mask + 4 meta floats
        obs_dim = T + T + 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [config.n_action_types, config.max_seq_length]
        )

        # Internal state (set in reset)
        self.original_tokens = None
        self.current_tokens = None
        self.mod_mask = None
        self.step_count = 0
        self.n_mutations = 0
        self.last_confidence = 0.0
        self.initial_confidence = 0.0
        self.budget_remaining = 1.0

    def _make_obs(self) -> np.ndarray:
        T = self.cfg.max_seq_length
        norm_tokens = self.current_tokens.astype(np.float32) / max(self.cfg.vocab_size, 1)
        meta = np.array([
            self.budget_remaining,
            self.last_confidence,
            self.step_count / max(self.cfg.max_steps, 1),
            self.n_mutations / max(self.cfg.max_steps, 1),
        ], dtype=np.float32)
        return np.concatenate([norm_tokens, self.mod_mask, meta])

    @property
    def obs_dim(self) -> int:
        return self.cfg.max_seq_length * 2 + 4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(len(self.sample_pool))
        tokens, label, family = self.sample_pool[idx]

        self.original_tokens = tokens.copy()
        self.current_tokens = tokens.copy()
        self.mod_mask = np.zeros(self.cfg.max_seq_length, dtype=np.float32)
        self.step_count = 0
        self.n_mutations = 0
        self.budget_remaining = 1.0

        self.initial_confidence = self.defender_fn(self.current_tokens)
        self.last_confidence = self.initial_confidence

        return self._make_obs(), {"family": family}

    def step(self, action):
        action_type, target_pos = int(action[0]), int(action[1])
        self.step_count += 1

        # Apply mutation
        mutated, success = apply_action(
            self.current_tokens, action_type, target_pos, self.vocab
        )

        if not success:
            # Action couldn't be applied (e.g., no substitution available)
            reward = -0.1
            obs = self._make_obs()
            truncated = self.step_count >= self.cfg.max_steps
            return obs, reward, False, truncated, {"action_failed": True}

        # Oracle check
        oracle_result = self.oracle_fn(self.original_tokens, mutated)
        if not oracle_result.valid:
            reward = compute_reward(
                self.last_confidence, self.last_confidence,
                oracle_valid=False, budget_remaining=self.budget_remaining,
                config=self.reward_config,
            )
            obs = self._make_obs()
            return obs, reward, True, False, {"oracle_rejected": True}

        # Update state
        self.current_tokens = mutated
        self.mod_mask[target_pos] = 1.0
        self.n_mutations += 1
        changed = np.sum(self.current_tokens != self.original_tokens)
        self.budget_remaining = 1.0 - changed / len(self.current_tokens)

        # Query defender
        new_confidence = self.defender_fn(self.current_tokens)
        reward = compute_reward(
            self.last_confidence, new_confidence,
            oracle_valid=True, budget_remaining=self.budget_remaining,
            config=self.reward_config,
        )
        self.last_confidence = new_confidence

        terminated = self.budget_remaining <= 0
        truncated = self.step_count >= self.cfg.max_steps
        evasion = new_confidence < 0.5

        info = {
            "confidence": new_confidence,
            "evasion": evasion,
            "n_mutations": self.n_mutations,
            "budget": self.budget_remaining,
        }
        return self._make_obs(), reward, terminated, truncated, info
```

**Test file:** `tests/adversarial/test_environment.py`

```python
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
```

**Done when:** `pytest tests/adversarial/test_environment.py -v` passes.

---

## TASK 7: Create the MLX PPO agent

**File:** `src/wintermute/adversarial/ppo.py`

This is the core RL algorithm. ALL neural network code uses `mlx.nn` and `mlx.optimizers`.
Follow the exact same patterns as `joint_trainer.py` for the update loop.

```python
"""
ppo.py — Proximal Policy Optimization in pure MLX.

Patterns copied from joint_trainer.py:
  - nn.value_and_grad(model, loss_fn) for backward pass
  - mlx.utils.tree_flatten(grads) for gradient clipping
  - optim.AdamW for parameter updates
  - mx.eval(model.parameters(), optimizer.state) to materialize
"""

from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np
from dataclasses import dataclass


@dataclass
class PPOConfig:
    obs_dim: int = 4100          # will be set from env.obs_dim
    n_actions: int = 4           # number of action types
    max_position: int = 2048     # max target position
    hidden_dim: int = 256
    gamma: float = 0.5
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 1e-4
    n_update_epochs: int = 4
    minibatch_size: int = 64


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic. Outputs action logits, position logits, value."""

    def __init__(self, cfg: PPOConfig):
        super().__init__()
        D = cfg.hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(cfg.obs_dim, D), nn.GELU(),
            nn.Linear(D, D), nn.GELU(),
        )
        self.action_head = nn.Linear(D, cfg.n_actions)
        self.position_head = nn.Linear(D, cfg.max_position)
        self.value_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 1),
        )

    def __call__(self, obs: mx.array):
        """obs: [B, obs_dim] → (action_logits [B, A], pos_logits [B, P], values [B])"""
        h = self.backbone(obs)
        return self.action_head(h), self.position_head(h), self.value_head(h).squeeze(-1)

    def evaluate_actions(self, obs, actions, positions):
        """Given obs and taken actions, return log_probs, entropy, values."""
        act_logits, pos_logits, values = self(obs)

        act_probs = mx.softmax(act_logits, axis=-1)
        act_lp = mx.log(act_probs[mx.arange(actions.shape[0]), actions] + 1e-8)

        pos_probs = mx.softmax(pos_logits, axis=-1)
        pos_lp = mx.log(pos_probs[mx.arange(positions.shape[0]), positions] + 1e-8)

        log_prob = act_lp + pos_lp

        act_ent = -mx.sum(act_probs * mx.log(act_probs + 1e-8), axis=-1)
        pos_ent = -mx.sum(pos_probs * mx.log(pos_probs + 1e-8), axis=-1)
        entropy = act_ent + pos_ent

        return log_prob, entropy, values


class PPOTrainer:
    """PPO training loop. Uses the same grad clip + AdamW pattern as JointTrainer."""

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.model = ActorCritic(cfg)
        self.optimizer = optim.AdamW(learning_rate=cfg.learning_rate)

    def sample_action(self, obs_np: np.ndarray) -> tuple[int, int, float, float]:
        """
        Given a numpy observation from the env, sample an action.

        Args:
            obs_np: [obs_dim] numpy array

        Returns:
            (action_type: int, position: int, log_prob: float, value: float)
        """
        obs_mx = mx.array(obs_np[np.newaxis, :], dtype=mx.float32)
        act_logits, pos_logits, values = self.model(obs_mx)

        action = mx.random.categorical(act_logits)
        position = mx.random.categorical(pos_logits)

        act_probs = mx.softmax(act_logits, axis=-1)
        pos_probs = mx.softmax(pos_logits, axis=-1)
        log_prob = (
            mx.log(act_probs[0, action[0]] + 1e-8) +
            mx.log(pos_probs[0, position[0]] + 1e-8)
        )

        mx.eval(action, position, log_prob, values)
        return (
            int(action.item()),
            int(position.item()),
            float(log_prob.item()),
            float(values[0].item()),
        )

    def compute_gae(self, rewards, values, dones) -> tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation. Runs in numpy (sequential)."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_v = 0.0 if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * next_v * (1 - dones[t]) - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(self, rollout: dict) -> dict:
        """
        PPO clipped surrogate update on collected rollout data.

        Args:
            rollout: dict with numpy arrays — obs, actions, positions,
                     log_probs, advantages, returns

        Returns:
            dict with 'loss' metric
        """
        # Convert numpy rollout → MLX
        obs = mx.array(rollout["obs"], dtype=mx.float32)
        actions = mx.array(rollout["actions"], dtype=mx.int32)
        positions = mx.array(rollout["positions"], dtype=mx.int32)
        old_lp = mx.array(rollout["log_probs"], dtype=mx.float32)
        advantages = mx.array(rollout["advantages"], dtype=mx.float32)
        returns = mx.array(rollout["returns"], dtype=mx.float32)

        # Normalize advantages
        adv_std = mx.sqrt(mx.var(advantages) + 1e-8)
        advantages = (advantages - mx.mean(advantages)) / adv_std

        total_loss = 0.0
        n_updates = 0
        T = obs.shape[0]

        for _ in range(self.cfg.n_update_epochs):
            idx = np.random.permutation(T)
            for start in range(0, T, self.cfg.minibatch_size):
                end = min(start + self.cfg.minibatch_size, T)
                mb = mx.array(idx[start:end])

                def ppo_loss(model):
                    new_lp, entropy, new_val = model.evaluate_actions(
                        obs[mb], actions[mb], positions[mb]
                    )
                    ratio = mx.exp(new_lp - old_lp[mb])
                    surr1 = ratio * advantages[mb]
                    surr2 = mx.clip(ratio, 1 - self.cfg.clip_range,
                                    1 + self.cfg.clip_range) * advantages[mb]
                    policy_loss = -mx.mean(mx.minimum(surr1, surr2))
                    value_loss = mx.mean((new_val - returns[mb]) ** 2)
                    entropy_loss = -mx.mean(entropy)
                    return (policy_loss
                            + self.cfg.value_coef * value_loss
                            + self.cfg.entropy_coef * entropy_loss)

                loss, grads = nn.value_and_grad(self.model, ppo_loss)(self.model)

                # Gradient clipping — same as joint_trainer.py
                flat = [v for _, v in mlx.utils.tree_flatten(grads)
                        if isinstance(v, mx.array)]
                if flat:
                    norm = mx.sqrt(sum(mx.sum(g * g) for g in flat))
                    mx.eval(norm)
                    if float(norm.item()) > self.cfg.max_grad_norm:
                        scale = self.cfg.max_grad_norm / (float(norm.item()) + 1e-6)
                        grads = mlx.utils.tree_map(
                            lambda g: g * scale if isinstance(g, mx.array) else g,
                            grads,
                        )

                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)

                total_loss += float(loss.item())
                n_updates += 1

        return {"loss": total_loss / max(n_updates, 1)}
```

**Test file:** `tests/adversarial/test_ppo.py`

```python
import mlx.core as mx
import numpy as np
from wintermute.adversarial.ppo import ActorCritic, PPOConfig, PPOTrainer

class TestActorCritic:
    def test_output_shapes(self):
        cfg = PPOConfig(obs_dim=64, n_actions=4, max_position=16, hidden_dim=32)
        model = ActorCritic(cfg)
        obs = mx.random.normal((4, 64))
        act_logits, pos_logits, values = model(obs)
        assert act_logits.shape == (4, 4)
        assert pos_logits.shape == (4, 16)
        assert values.shape == (4,)

    def test_evaluate_actions_shapes(self):
        cfg = PPOConfig(obs_dim=64, n_actions=4, max_position=16, hidden_dim=32)
        model = ActorCritic(cfg)
        obs = mx.random.normal((4, 64))
        actions = mx.array([0, 1, 2, 3])
        positions = mx.array([0, 5, 10, 15])
        lp, ent, val = model.evaluate_actions(obs, actions, positions)
        assert lp.shape == (4,)
        assert ent.shape == (4,)
        assert val.shape == (4,)

class TestPPOTrainer:
    def test_update_reduces_loss(self):
        cfg = PPOConfig(obs_dim=20, n_actions=4, max_position=8,
                        hidden_dim=16, n_update_epochs=2, minibatch_size=8)
        trainer = PPOTrainer(cfg)
        T = 32
        rollout = {
            "obs": np.random.randn(T, 20).astype(np.float32),
            "actions": np.random.randint(0, 4, T).astype(np.int32),
            "positions": np.random.randint(0, 8, T).astype(np.int32),
            "log_probs": np.random.randn(T).astype(np.float32) * 0.1,
            "advantages": np.random.randn(T).astype(np.float32),
            "returns": np.random.randn(T).astype(np.float32),
        }
        result = trainer.update(rollout)
        assert np.isfinite(result["loss"])
```

**Done when:** `pytest tests/adversarial/test_ppo.py -v` passes.

---

## TASK 8: Create the TRADES loss

**File:** `src/wintermute/adversarial/trades_loss.py`

~40 lines of MLX. Plugs into `joint_trainer.py` as an alternative loss function.

```python
"""
trades_loss.py — TRADES loss in pure MLX.

L = cross_entropy(f(x_clean), y) + β · KL(softmax(f(x_clean)) || softmax(f(x_adv)))

β warms up from 0 to target over the first 30% of training.
"""

import mlx.core as mx
import mlx.nn as nn


class TRADESLoss:
    def __init__(self, beta: float = 1.0, warmup_fraction: float = 0.3):
        self.beta = beta
        self.warmup_fraction = warmup_fraction

    def __call__(self, model, x_clean, labels, x_adv, epoch: int, max_epochs: int):
        clean_logits = model(x_clean)
        L_det = mx.mean(nn.losses.cross_entropy(clean_logits, labels))

        adv_logits = model(x_adv)
        clean_p = mx.softmax(clean_logits, axis=-1)
        adv_log_p = mx.log_softmax(adv_logits, axis=-1)
        kl = mx.sum(clean_p * (mx.log(clean_p + 1e-8) - adv_log_p), axis=-1)
        L_kl = mx.mean(kl)

        beta_t = self.beta * min(1.0, epoch / max(max_epochs * self.warmup_fraction, 1))
        return L_det + beta_t * L_kl
```

**Test file:** `tests/adversarial/test_trades.py`

```python
import mlx.core as mx
import mlx.nn as nn
from wintermute.adversarial.trades_loss import TRADESLoss
from wintermute.models.fusion import DetectorConfig, WintermuteMalwareDetector

class TestTRADES:
    def test_identical_inputs_zero_kl(self):
        """When clean == adversarial, KL term should be ~0."""
        cfg = DetectorConfig(vocab_size=32, dims=32, num_heads=1, num_layers=1,
                             mlp_dims=64, num_classes=2, max_seq_length=16)
        model = WintermuteMalwareDetector(cfg)
        x = mx.random.randint(0, 32, (4, 16))
        y = mx.array([0, 1, 0, 1])
        trades = TRADESLoss(beta=1.0)
        loss_trades = trades(model, x, y, x, epoch=10, max_epochs=10)  # same input
        loss_ce = mx.mean(nn.losses.cross_entropy(model(x), y))
        mx.eval(loss_trades, loss_ce)
        # Should be very close since KL(p||p) ≈ 0
        assert abs(float(loss_trades.item()) - float(loss_ce.item())) < 0.1

    def test_beta_zero_equals_ce(self):
        cfg = DetectorConfig(vocab_size=32, dims=32, num_heads=1, num_layers=1,
                             mlp_dims=64, num_classes=2, max_seq_length=16)
        model = WintermuteMalwareDetector(cfg)
        x = mx.random.randint(0, 32, (4, 16))
        y = mx.array([0, 1, 0, 1])
        trades = TRADESLoss(beta=0.0)
        loss = trades(model, x, y, x, epoch=0, max_epochs=10)
        mx.eval(loss)
        assert float(loss.item()) > 0
```

**Done when:** `pytest tests/adversarial/test_trades.py -v` passes.

---

## TASK 9: Create the adversarial vault

**File:** `src/wintermute/adversarial/vault.py`

Stores evasive samples for replay during defender retraining.
Uses numpy arrays on disk. No MLX.

```python
"""
vault.py — Adversarial sample vault for replay during defender retraining.

Stores mutated token sequences that successfully evaded the defender.
Provides stratified sampling for balanced replay batches.
All storage is numpy. No MLX.
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VaultEntry:
    mutated_tokens: np.ndarray
    label: int
    family: str
    evasion_confidence: float     # defender's confidence on the mutated sample
    action_types_used: list[int]
    n_mutations: int
    epoch: int


@dataclass
class VaultConfig:
    max_samples: int = 50_000
    replay_ratio: float = 0.2


class AdversarialVault:
    def __init__(self, config: VaultConfig | None = None):
        self.cfg = config or VaultConfig()
        self.entries: list[VaultEntry] = []

    def __len__(self):
        return len(self.entries)

    def add(self, entry: VaultEntry):
        self.entries.append(entry)
        # Evict oldest if over capacity
        if len(self.entries) > self.cfg.max_samples:
            self.entries = self.entries[-self.cfg.max_samples:]

    def sample_replay_batch(self, batch_size: int, rng: np.random.Generator | None = None
                             ) -> np.ndarray | None:
        """
        Sample adversarial tokens for replay.

        Returns: [N, T] numpy array where N = batch_size * replay_ratio, or None if vault empty.
        """
        if len(self.entries) == 0:
            return None
        rng = rng or np.random.default_rng()
        n = max(1, int(batch_size * self.cfg.replay_ratio))
        n = min(n, len(self.entries))
        indices = rng.choice(len(self.entries), size=n, replace=False)
        tokens = np.stack([self.entries[i].mutated_tokens for i in indices])
        return tokens

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tokens = np.stack([e.mutated_tokens for e in self.entries]) if self.entries else np.array([])
        np.save(path / "vault_tokens.npy", tokens)
        meta = [
            {"label": e.label, "family": e.family, "confidence": e.evasion_confidence,
             "actions": e.action_types_used, "n_mutations": e.n_mutations, "epoch": e.epoch}
            for e in self.entries
        ]
        (path / "vault_meta.json").write_text(json.dumps(meta))

    def load(self, path: str | Path):
        path = Path(path)
        tokens = np.load(path / "vault_tokens.npy")
        meta = json.loads((path / "vault_meta.json").read_text())
        self.entries = []
        for i, m in enumerate(meta):
            self.entries.append(VaultEntry(
                mutated_tokens=tokens[i], label=m["label"], family=m["family"],
                evasion_confidence=m["confidence"], action_types_used=m["actions"],
                n_mutations=m["n_mutations"], epoch=m["epoch"],
            ))
```

**Done when:** `python -c "from wintermute.adversarial.vault import AdversarialVault; v = AdversarialVault(); print('ok')"` prints `ok`.

---

## TASK 10: Create the orchestrator

**File:** `src/wintermute/adversarial/orchestrator.py`

This ties everything together: rollout collection → PPO update → vault storage → defender retraining.

Key pattern: rollout loop uses numpy (Gymnasium). PPO update and defender retraining use MLX. The DefenderBridge is the only conversion point.

```python
"""
orchestrator.py — Coordinates the adversarial training loop.

1. Collect rollouts from AsmMutationEnv (numpy)
2. Update PPO agent (MLX)
3. Store evasive samples in vault (numpy)
4. Retrain defender with TRADES loss + vault replay (MLX)
"""

from __future__ import annotations
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

from wintermute.adversarial.ppo import PPOTrainer, PPOConfig
from wintermute.adversarial.environment import AsmMutationEnv, EnvConfig
from wintermute.adversarial.oracle import TieredOracle
from wintermute.adversarial.bridge import DefenderBridge
from wintermute.adversarial.vault import AdversarialVault, VaultConfig, VaultEntry
from wintermute.adversarial.trades_loss import TRADESLoss
from wintermute.adversarial.reward import RewardConfig
from wintermute.models.fusion import WintermuteMalwareDetector


class AdversarialOrchestrator:
    """
    Main entry point for adversarial training.

    Usage:
        orch = AdversarialOrchestrator(model, vocab, sample_pool)
        for cycle in range(n_cycles):
            metrics = orch.run_cycle(n_episodes=500)
            print(metrics)
    """

    def __init__(
        self,
        model: WintermuteMalwareDetector,
        vocab: dict,
        sample_pool: list[tuple[np.ndarray, int, str]],
        env_config: EnvConfig | None = None,
        ppo_config: PPOConfig | None = None,
        vault_config: VaultConfig | None = None,
        trades_beta: float = 1.0,
    ):
        self.model = model
        self.vocab = vocab

        # Defender bridge (MLX model ↔ numpy env)
        self.bridge = DefenderBridge(model)

        # Oracle
        self.oracle = TieredOracle(vocab_size=len(vocab))

        # Environment
        env_cfg = env_config or EnvConfig(
            vocab_size=len(vocab),
            max_seq_length=sample_pool[0][0].shape[0] if sample_pool else 2048,
            n_action_types=4,
        )
        self.env = AsmMutationEnv(
            config=env_cfg,
            defender_fn=self.bridge,
            oracle_fn=lambda orig, mut: self.oracle.validate(orig, mut),
            sample_pool=sample_pool,
            vocab=vocab,
        )

        # PPO agent
        ppo_cfg = ppo_config or PPOConfig(
            obs_dim=self.env.obs_dim,
            n_actions=env_cfg.n_action_types,
            max_position=env_cfg.max_seq_length,
        )
        self.ppo = PPOTrainer(ppo_cfg)

        # Vault
        self.vault = AdversarialVault(vault_config or VaultConfig())

        # TRADES loss for defender retraining
        self.trades = TRADESLoss(beta=trades_beta)

        self._cycle_count = 0

    def run_cycle(self, n_episodes: int = 500) -> dict:
        """Run one full adversarial cycle: attack → PPO update → store evasions."""
        self._cycle_count += 1
        print(f"\n{'='*60}")
        print(f" Adversarial Cycle {self._cycle_count}")
        print(f"{'='*60}")

        # Step 1: Collect rollouts
        print(f"\n🔴 Collecting {n_episodes} episodes...")
        rollout_data, episode_stats = self._collect_rollouts(n_episodes)

        # Step 2: PPO update
        print("🔴 Updating PPO agent...")
        ppo_metrics = self.ppo.update(rollout_data)

        # Step 3: Log stats
        evasion_rate = episode_stats["evasions"] / max(n_episodes, 1)
        print(f"   Episodes: {n_episodes}")
        print(f"   Evasions: {episode_stats['evasions']} ({evasion_rate:.1%})")
        print(f"   PPO loss: {ppo_metrics['loss']:.4f}")
        print(f"   Vault size: {len(self.vault)}")

        return {
            "cycle": self._cycle_count,
            "evasion_rate": evasion_rate,
            "ppo_loss": ppo_metrics["loss"],
            "vault_size": len(self.vault),
            "mean_confidence": episode_stats["mean_final_confidence"],
            "mean_mutations": episode_stats["mean_mutations"],
        }

    def _collect_rollouts(self, n_episodes: int) -> tuple[dict, dict]:
        """Collect experience from env. Numpy throughout, convert to MLX in PPO.update()."""
        all_obs, all_actions, all_positions = [], [], []
        all_log_probs, all_values, all_rewards, all_dones = [], [], [], []

        n_evasions = 0
        final_confidences = []
        mutations_per_ep = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            ep_rewards = []
            ep_values = []
            done = False

            while not done:
                action, position, log_prob, value = self.ppo.sample_action(obs)

                next_obs, reward, terminated, truncated, step_info = self.env.step(
                    np.array([action, position])
                )
                done = terminated or truncated

                all_obs.append(obs)
                all_actions.append(action)
                all_positions.append(position)
                all_log_probs.append(log_prob)
                all_values.append(value)
                all_rewards.append(reward)
                all_dones.append(float(done))

                ep_rewards.append(reward)
                ep_values.append(value)
                obs = next_obs

                # Track evasions and store in vault
                if step_info.get("evasion", False):
                    n_evasions += 1
                    self.vault.add(VaultEntry(
                        mutated_tokens=self.env.current_tokens.copy(),
                        label=self.env.current_label if hasattr(self.env, 'current_label') else 1,
                        family=info.get("family", "unknown"),
                        evasion_confidence=step_info["confidence"],
                        action_types_used=[action],
                        n_mutations=step_info.get("n_mutations", 0),
                        epoch=self._cycle_count,
                    ))

            final_confidences.append(self.env.last_confidence)
            mutations_per_ep.append(self.env.n_mutations)

        # Compute GAE
        advantages, returns = self.ppo.compute_gae(
            all_rewards, all_values, all_dones
        )

        rollout = {
            "obs": np.stack(all_obs).astype(np.float32),
            "actions": np.array(all_actions, dtype=np.int32),
            "positions": np.array(all_positions, dtype=np.int32),
            "log_probs": np.array(all_log_probs, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
        }

        stats = {
            "evasions": n_evasions,
            "mean_final_confidence": float(np.mean(final_confidences)) if final_confidences else 0.0,
            "mean_mutations": float(np.mean(mutations_per_ep)) if mutations_per_ep else 0.0,
        }

        return rollout, stats
```

**Done when:** Can be imported: `python -c "from wintermute.adversarial.orchestrator import AdversarialOrchestrator; print('ok')"`

---

## TASK 11: Add CLI subcommands

**File:** `src/wintermute/cli.py` — EXTEND, don't rewrite.

Add after the existing `data_app` Typer group:

```python
# Add this import at top of file
# from pathlib import Path

adv_app = typer.Typer(
    name="adv",
    help="Adversarial training pipeline (Phase 5)",
    no_args_is_help=True,
)
app.add_typer(adv_app, name="adv")


@adv_app.command()
def run(
    model: str = typer.Option("malware_detector.safetensors", "--model", "-m"),
    manifest: str = typer.Option("malware_detector_manifest.json", "--manifest"),
    vocab: str = typer.Option("data/processed/vocab.json", "--vocab", "-v"),
    data_dir: str = typer.Option("data/processed", "--data-dir"),
    episodes: int = typer.Option(500, "--episodes", "-n"),
    cycles: int = typer.Option(5, "--cycles"),
    trades_beta: float = typer.Option(1.0, "--trades-beta"),
) -> None:
    """Run adversarial training cycles: attack → update → store."""
    import json
    import numpy as np
    from wintermute.models.fusion import DetectorConfig, WintermuteMalwareDetector
    from wintermute.adversarial.orchestrator import AdversarialOrchestrator

    # Load model
    dp = Path(data_dir)
    with open(dp / "vocab.json") as f:
        voc = json.load(f)

    manifest_data = json.loads(Path(manifest).read_text())
    cfg = DetectorConfig(
        vocab_size=manifest_data.get("vocab_size", len(voc)),
        num_classes=manifest_data.get("num_classes", 2),
    )
    detector = WintermuteMalwareDetector(cfg)
    detector.load_weights(model)
    typer.echo(f"Loaded model from {model}")

    # Build sample pool from training data (malicious samples only)
    x_data = np.load(dp / "x_data.npy")
    y_data = np.load(dp / "y_data.npy")
    pool = [
        (x_data[i], int(y_data[i]), "unknown")
        for i in range(len(y_data)) if y_data[i] == 1  # malicious only
    ]
    typer.echo(f"Sample pool: {len(pool)} malicious samples")

    orch = AdversarialOrchestrator(
        model=detector, vocab=voc, sample_pool=pool, trades_beta=trades_beta,
    )

    for c in range(cycles):
        metrics = orch.run_cycle(n_episodes=episodes)
        typer.echo(json.dumps(metrics, indent=2))
```

**Done when:** `wintermute adv --help` shows the `run` subcommand.

---

## TASK 12: Run all tests

```bash
pytest tests/adversarial/ -v
```

**Done when:** All tests pass. If any fail, fix them before proceeding.

---

## BUILD ORDER SUMMARY

```
Task 0:  pyproject.toml              ← no dependencies
Task 1:  substitution_table.py       ← no dependencies
Task 2:  code_actions.py             ← depends on Task 1
Task 3:  oracle.py                   ← no dependencies
Task 4:  reward.py                   ← no dependencies
Task 5:  bridge.py                   ← imports fusion.py (exists)
Task 6:  environment.py              ← depends on Tasks 2, 3, 4
Task 7:  ppo.py                      ← no dependencies (pure MLX)
Task 8:  trades_loss.py              ← no dependencies (pure MLX)
Task 9:  vault.py                    ← no dependencies (numpy)
Task 10: orchestrator.py             ← depends on Tasks 5, 6, 7, 8, 9
Task 11: cli.py extension            ← depends on Task 10
Task 12: run all tests               ← depends on all above
```

Tasks 1-5, 7, 8, 9 can be built in parallel. Task 6 needs 2+3+4. Task 10 needs all.
