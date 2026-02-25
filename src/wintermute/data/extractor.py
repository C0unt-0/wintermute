"""
src/wintermute/data/extractor.py — Headless Binary Disassembly Engine

Uses Radare2 (via r2pipe) to programmatically open raw executables (.exe/.elf),
trace execution paths, and output the exact tensors the ML models expect:
  - A linear opcode sequence for the Transformer (MalBERT)
  - A Control Flow Graph (CFG) for the Graph Neural Network (GNN)
"""

import r2pipe
import json
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class HeadlessDisassembler:
    """Extracts ML features directly from raw executable binaries using Radare2."""

    def __init__(self, binary_path: str):
        self.binary_path = binary_path
        # Open binary in quiet mode (-q), suppress stderr (-2)
        self.r2 = r2pipe.open(self.binary_path, flags=["-q", "-2"])

    def extract_features(self) -> tuple[str, nx.DiGraph]:
        """
        Performs advanced analysis on the binary and returns:
          - sequence (str): space-joined opcode mnemonics for the sequence model
          - cfg (nx.DiGraph): Control Flow Graph for the GNN model
        """
        logger.info(f"Analyzing {self.binary_path}...")
        # 'aaa' tells Radare2 to perform advanced analysis on all functions
        self.r2.cmd("aaa")

        sequence: list[str] = []
        cfg: nx.DiGraph = nx.DiGraph()

        # 'aflj': Analyze Functions List JSON
        functions = json.loads(self.r2.cmd("aflj") or "[]")

        for func in functions:
            # 'agj': Analyze Graph JSON (returns basic blocks and branch edges)
            func_data = json.loads(
                self.r2.cmd(f"agj @ {func['offset']}") or "[]"
            )
            if not func_data:
                continue

            for block in func_data[0].get("blocks", []):
                block_id = block.get("offset")
                block_ops: list[str] = []

                # 1. Extract sequential instructions for MalBERT.
                # We strip memory addresses/immediates to prevent overfitting.
                # e.g. 'mov eax, 0x1' -> 'mov'
                for op in block.get("ops", []):
                    opcode = op.get("disasm", "").split()[0]
                    if opcode:
                        sequence.append(opcode)
                        block_ops.append(opcode)

                # 2. Build the CFG for the GNN
                cfg.add_node(
                    block_id,
                    features=block_ops,
                    size=block.get("size", 0),
                )

                # Add branching edges
                if "jump" in block:  # True branch
                    cfg.add_edge(block_id, block["jump"], type="jump")
                if "fail" in block:  # False branch
                    cfg.add_edge(block_id, block["fail"], type="fail")

        self.r2.quit()
        return " ".join(sequence), cfg
