# tests/test_disassembler.py
from unittest.mock import MagicMock, patch
import json
from wintermute.data.disassembler import DisassemblyResult, HeadlessDisassembler


def _make_r2_mock():
    r2 = MagicMock()
    funcs = [{"offset": 0x1000}]
    blocks = [
        {"offset": 0x1000, "ops": [{"disasm": "push ebp"}, {"disasm": "mov esp, ebp"}], "jump": 0x1010},
        {"offset": 0x1010, "ops": [{"disasm": "ret"}]},
    ]
    r2.cmd.side_effect = lambda cmd: (
        json.dumps(funcs) if "aflj" in cmd
        else json.dumps([{"blocks": blocks}]) if "agj" in cmd
        else ""
    )
    return r2


class TestDisassemblyResult:
    def test_defaults(self):
        r = DisassemblyResult(extraction_failed=True)
        assert r.sequence == [] and r.n_nodes == 0

class TestHeadlessDisassembler:
    @patch("wintermute.data.disassembler.r2pipe")
    def test_sequence_extracted(self, mock_r2pipe):
        mock_r2pipe.open.return_value = _make_r2_mock()
        result = HeadlessDisassembler("fake.exe").extract()
        assert not result.extraction_failed
        assert "push" in result.sequence

    @patch("wintermute.data.disassembler.r2pipe")
    def test_edge_index_populated(self, mock_r2pipe):
        mock_r2pipe.open.return_value = _make_r2_mock()
        result = HeadlessDisassembler("fake.exe").extract()
        src, dst = result.edge_index
        assert len(src) > 0 and len(src) == len(dst)

    @patch("wintermute.data.disassembler.r2pipe")
    def test_node_limit_fails(self, mock_r2pipe):
        many = [{"offset": i, "ops": [{"disasm": "nop"}]} for i in range(5001)]
        r2 = MagicMock()
        r2.cmd.side_effect = lambda cmd: (
            json.dumps([{"offset": 0}]) if "aflj" in cmd
            else json.dumps([{"blocks": many}]) if "agj" in cmd
            else ""
        )
        mock_r2pipe.open.return_value = r2
        result = HeadlessDisassembler("big.exe", max_nodes=5000).extract()
        assert result.extraction_failed

    @patch("wintermute.data.disassembler.r2pipe")
    def test_exception_fails(self, mock_r2pipe):
        mock_r2pipe.open.side_effect = Exception("crash")
        result = HeadlessDisassembler("bad.exe").extract()
        assert result.extraction_failed

    @patch("wintermute.data.disassembler.r2pipe")
    def test_timeout_fails(self, mock_r2pipe):
        from concurrent.futures import TimeoutError as FuturesTimeoutError
        mock_r2pipe.open.return_value = MagicMock()
        with patch("concurrent.futures.Future.result", side_effect=FuturesTimeoutError()):
            result = HeadlessDisassembler("slow.exe", timeout=1).extract()
        assert result.extraction_failed
