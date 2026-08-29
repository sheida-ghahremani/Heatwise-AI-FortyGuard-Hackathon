"""Build trusted, low-memory binary sidecars for prepared routing graphs."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import osmnx as ox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graphs", nargs="+")
    args = parser.parse_args()
    for value in args.graphs:
        source = Path(value)
        graph = ox.load_graphml(source)
        output = source.with_suffix(".pickle")
        with output.open("wb") as target:
            pickle.dump(graph, target, protocol=5)
        print(f"Saved {output}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")


if __name__ == "__main__":
    main()
