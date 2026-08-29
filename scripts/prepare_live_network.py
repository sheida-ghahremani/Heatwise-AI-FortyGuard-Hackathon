"""Join cached FortyGuard and local canopy data to an OSMnx walk graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import osmnx as ox

from heatwise.data_integration import apply_canopy_shade, assign_heatmap_to_graph, load_fortyguard_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--heatmap", required=True)
    parser.add_argument("--vegetation")
    parser.add_argument("--output", default="data/networks/college_station_walk_live.graphml")
    args = parser.parse_args()

    graph = ox.load_graphml(args.graph)
    heatmap = load_fortyguard_heatmap(args.heatmap)
    graph = assign_heatmap_to_graph(graph, heatmap)
    if args.vegetation:
        graph = apply_canopy_shade(graph, args.vegetation)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, output)
    shaded = sum(float(data.get("shade_fraction", 0)) >= 0.7 for *_, data in graph.edges(data=True))
    print(f"Saved {output}: {len(graph.nodes)} nodes, {len(graph.edges)} edges, {shaded} shaded edges")


if __name__ == "__main__":
    main()
