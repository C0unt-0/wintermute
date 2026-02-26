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
