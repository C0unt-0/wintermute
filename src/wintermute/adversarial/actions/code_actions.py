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

    Simple version: find tokens containing 'eax' in a +-5 window and swap to 'ecx'
    (or vice versa) if both variants exist in vocab. This is a conservative approximation
    -- the full version would use liveness analysis from the CFG.
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


# Action dispatch table -- maps action_id to function
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
