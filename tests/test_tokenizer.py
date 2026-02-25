"""
test_tokenizer.py — Tests for wintermute.data.tokenizer
"""

import numpy as np
import pytest

from wintermute.data.tokenizer import (
    build_vocabulary,
    detect_asm_format,
    encode_sequence,
    read_bazaar_asm,
)


class TestBuildVocabulary:
    def test_includes_special_tokens(self):
        stoi = build_vocabulary([["mov", "push"]])
        assert stoi["<PAD>"] == 0
        assert stoi["<UNK>"] == 1

    def test_sorted_opcodes(self):
        stoi = build_vocabulary([["push", "mov", "add"]])
        keys = list(stoi.keys())
        # After special tokens, opcodes should be sorted
        assert keys[2:] == ["add", "mov", "push"]

    def test_unique_ids(self):
        stoi = build_vocabulary([["mov", "push", "mov", "xor"]])
        ids = list(stoi.values())
        assert len(ids) == len(set(ids)), "All IDs must be unique"

    def test_empty_input(self):
        stoi = build_vocabulary([])
        assert len(stoi) == 2  # PAD + UNK only

    def test_multiple_lists(self):
        stoi = build_vocabulary([["mov", "push"], ["xor", "mov"]])
        assert "mov" in stoi
        assert "push" in stoi
        assert "xor" in stoi


class TestEncodeSequence:
    def test_truncation(self):
        opcodes = ["mov"] * 100
        stoi = {"<PAD>": 0, "<UNK>": 1, "mov": 2}
        result = encode_sequence(opcodes, stoi, max_len=10)
        assert result.shape == (10,)
        assert all(result == 2)

    def test_padding(self):
        opcodes = ["mov", "push"]
        stoi = {"<PAD>": 0, "<UNK>": 1, "mov": 2, "push": 3}
        result = encode_sequence(opcodes, stoi, max_len=5)
        assert result.shape == (5,)
        assert result[0] == 2
        assert result[1] == 3
        assert all(result[2:] == 0)  # padded

    def test_unknown_tokens(self):
        opcodes = ["mov", "unknown_opcode"]
        stoi = {"<PAD>": 0, "<UNK>": 1, "mov": 2}
        result = encode_sequence(opcodes, stoi, max_len=5)
        assert result[0] == 2
        assert result[1] == 1  # UNK

    def test_dtype(self):
        stoi = {"<PAD>": 0, "<UNK>": 1, "mov": 2}
        result = encode_sequence(["mov"], stoi, max_len=5)
        assert result.dtype == np.int32


class TestReadBazaarAsm:
    def test_basic_read(self, tmp_path):
        asm_file = tmp_path / "sample.asm"
        asm_file.write_text("mov\npush\nxor\n")
        result = read_bazaar_asm(str(asm_file))
        assert result == ["mov", "push", "xor"]

    def test_skips_blank_lines(self, tmp_path):
        asm_file = tmp_path / "sample.asm"
        asm_file.write_text("mov\n\npush\n\n")
        result = read_bazaar_asm(str(asm_file))
        assert result == ["mov", "push"]

    def test_missing_file(self):
        result = read_bazaar_asm("/nonexistent/file.asm")
        assert result == []


class TestDetectAsmFormat:
    def test_ida_format(self, tmp_path):
        asm_file = tmp_path / "ida.asm"
        asm_file.write_text(".text:00401000 55       push    ebp\n")
        assert detect_asm_format(str(asm_file)) == "ida"

    def test_bazaar_format(self, tmp_path):
        asm_file = tmp_path / "bazaar.asm"
        asm_file.write_text("push\nmov\nxor\n")
        assert detect_asm_format(str(asm_file)) == "bazaar"
