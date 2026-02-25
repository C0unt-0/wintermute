# src/wintermute/data/disassembler.py
from __future__ import annotations
import json, logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
import r2pipe

logger = logging.getLogger(__name__)
MAX_NODES = 5000
DEFAULT_TIMEOUT = 30


@dataclass
class DisassemblyResult:
    sequence: list[str] = field(default_factory=list)
    edge_index: tuple[list[int], list[int]] = field(default_factory=lambda: ([], []))
    node_opcodes: list[list[str]] = field(default_factory=list)
    n_nodes: int = 0
    n_edges: int = 0
    extraction_failed: bool = False


class HeadlessDisassembler:
    """
    Extracts opcode sequence + CFG from a binary using Radare2.
    Replaces both data/cfg.py (angr) and data/extractor.py.
    """
    def __init__(self, binary_path: str, timeout: int = DEFAULT_TIMEOUT, max_nodes: int = MAX_NODES):
        self.binary_path = binary_path
        self.timeout = timeout
        self.max_nodes = max_nodes

    def extract(self) -> DisassemblyResult:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self._run)
            try:
                return fut.result(timeout=self.timeout)
            except FuturesTimeoutError:
                logger.warning("Timeout for %s", self.binary_path)
                return DisassemblyResult(extraction_failed=True)
            except Exception as e:
                logger.warning("Failed %s: %s", self.binary_path, e)
                return DisassemblyResult(extraction_failed=True)

    def _run(self) -> DisassemblyResult:
        r2 = r2pipe.open(self.binary_path, flags=["-q", "-2"])
        try:
            r2.cmd("aaa")
            sequence, src_nodes, dst_nodes, node_opcodes = [], [], [], []
            node_id_map: dict[int, int] = {}

            for func in json.loads(r2.cmd("aflj") or "[]"):
                func_data = json.loads(r2.cmd(f"agj @ {func['offset']}") or "[]")
                if not func_data:
                    continue
                for block in func_data[0].get("blocks", []):
                    offset = block.get("offset")
                    if offset not in node_id_map:
                        node_id_map[offset] = len(node_id_map)
                    idx = node_id_map[offset]
                    while len(node_opcodes) <= idx:
                        node_opcodes.append([])
                    ops = [op["disasm"].split()[0] for op in block.get("ops", []) if op.get("disasm")]
                    node_opcodes[idx] = ops
                    sequence.extend(ops)
                    for key in ("jump", "fail"):
                        tgt = block.get(key)
                        if tgt is not None:
                            if tgt not in node_id_map:
                                node_id_map[tgt] = len(node_id_map)
                            src_nodes.append(idx)
                            dst_nodes.append(node_id_map[tgt])

            n = len(node_id_map)
            if n > self.max_nodes:
                logger.warning("CFG has %d nodes > limit %d for %s", n, self.max_nodes, self.binary_path)
                return DisassemblyResult(sequence=sequence, extraction_failed=True, n_nodes=n)

            while len(node_opcodes) < n:
                node_opcodes.append([])

            return DisassemblyResult(
                sequence=sequence,
                edge_index=(src_nodes, dst_nodes),
                node_opcodes=node_opcodes,
                n_nodes=n,
                n_edges=len(src_nodes),
                extraction_failed=False,
            )
        finally:
            r2.quit()
