"""Memory-efficient loading for trusted HeatWise routing assets."""

from __future__ import annotations

import pickle
from pathlib import Path

import osmnx as ox


def routing_graph_exists(path: str | Path) -> bool:
    """Return true when either GraphML or its trusted binary sidecar exists."""
    graph_path = Path(path)
    return graph_path.exists() or graph_path.with_suffix(".pickle").exists()


def load_routing_graph(path: str | Path):
    """Prefer a trusted binary sidecar, with GraphML as a portable fallback."""
    graph_path = Path(path)
    binary_path = graph_path.with_suffix(".pickle")
    if binary_path.exists():
        with binary_path.open("rb") as source:
            return pickle.load(source)
    return ox.load_graphml(graph_path)
