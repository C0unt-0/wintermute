"""
cfg.py — Wintermute Control Flow Graph (CFG) Extraction

Uses `angr` to disassemble PE binaries and extract logical execution flow.
Produces graph-structured data (nodes, edges, node features) for GNN training.

Responsibilities:
    - Load binary via angr
    - Generate CFGFast
    - Extract Basic Blocks as nodes
    - Extract opcode sequences per node
    - Extract adjacency matrix (edges)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Silence angr/cle/pyvex logging unless necessary
logging.getLogger("angr").setLevel(logging.ERROR)
logging.getLogger("cle").setLevel(logging.ERROR)
logging.getLogger("pyvex").setLevel(logging.ERROR)

import angr
import numpy as np
from omegaconf import OmegaConf

from wintermute.data.tokenizer import (
    load_data_config, 
    SKIP_MNEMONICS
)

# ---------------------------------------------------------------------------
# Core Extractor
# ---------------------------------------------------------------------------

class CFGExtractor:
    """
    Extracts Control Flow Graphs from executable binaries using angr.
    """

    def __init__(self, config_path: str | Path | None = None):
        self.cfg = load_data_config(config_path)

    def extract_cfg(self, filepath: str | Path) -> dict | None:
        """
        Extract CFG from a single binary.

        Returns
        -------
        dict or None:
            - "nodes": List of opcode sequences (lists of strings)
            - "edges": Adjacency list (list of tuples)
            - "node_addresses": List of start addresses (for debugging)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"  [ERROR] {filepath} does not exist")
            return None

        try:
            # Load binary without libraries (static analysis focus)
            project = angr.Project(str(filepath), auto_load_libs=False)
            
            # Generate Fast CFG (much faster than Accurate, sufficient for ML)
            # resolve_indirect_jumps=True is a bit slower but captures more edges
            cfg_fast = project.analyses.CFGFast(resolve_indirect_jumps=True)
            
            graph = cfg_fast.graph
            
            # 1. Mapping nodes to indices
            angr_nodes = list(graph.nodes())
            node_to_idx = {node: i for i, node in enumerate(angr_nodes)}
            
            # 2. Extract node features (opcode sequences per basic block)
            node_opcodes = []
            node_addrs = []
            
            for node in angr_nodes:
                opcodes = []
                try:
                    # node.block might be None for some auxiliary nodes
                    if node.block:
                        # block.capstone contains disassembled instructions
                        for insn in node.block.capstone.insns:
                            mnemonic = insn.mnemonic.lower()
                            if mnemonic not in SKIP_MNEMONICS:
                                opcodes.append(mnemonic)
                except Exception:
                    # Block disassembly might fail if angr misidentified code
                    pass
                
                node_opcodes.append(opcodes)
                node_addrs.append(node.addr)

            # 3. Extract edges
            edges = []
            for u, v in graph.edges():
                edges.append((node_to_idx[u], node_to_idx[v]))

            return {
                "nodes": node_opcodes,
                "edges": edges,
                "node_addresses": node_addrs
            }

        except Exception as e:
            print(f"  [SKIP] {filepath}: angr error — {e}")
            return None

# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------

def process_binary_to_graph(filepath: str | Path, extractor: CFGExtractor) -> dict | None:
    """Convenience wrapper for single file extraction."""
    return extractor.extract_cfg(filepath)
