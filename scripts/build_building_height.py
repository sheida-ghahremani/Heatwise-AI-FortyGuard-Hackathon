#!/usr/bin/env python3
"""Derive a 1 m building-height raster from footprints and raw LiDAR CHM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import box


NODATA = -9999.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-chm", type=Path, required=True)
    parser.add_argument("--buildings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with rasterio.open(args.raw_chm) as src:
        raw = src.read(1, masked=True)
        profile = src.profile.copy()
        bounds = src.bounds
        raster_crs = src.crs
        transform = src.transform
        height, width = src.height, src.width

    # Spatial filtering in the source CRS avoids loading all 44k city footprints.
    source_crs = gpd.read_file(args.buildings, rows=1).crs
    tx = Transformer.from_crs(raster_crs, source_crs, always_xy=True)
    corners = [tx.transform(x, y) for x in (bounds.left, bounds.right) for y in (bounds.bottom, bounds.top)]
    source_bbox = (
        min(x for x, _ in corners), min(y for _, y in corners),
        max(x for x, _ in corners), max(y for _, y in corners),
    )
    buildings = gpd.read_file(args.buildings, bbox=source_bbox).to_crs(raster_crs)
    study_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    buildings = buildings[buildings.intersects(study_box)].copy()
    buildings["geometry"] = buildings.geometry.intersection(study_box)
    buildings = buildings[(~buildings.geometry.is_empty) & buildings.geometry.is_valid].reset_index(drop=True)

    labels = rasterize(
        ((geom, idx + 1) for idx, geom in enumerate(buildings.geometry)),
        out_shape=(height, width), transform=transform, fill=0,
        all_touched=False, dtype="int32",
    )
    raw_values = raw.filled(np.nan)
    obstacle = np.full((height, width), NODATA, dtype="float32")
    derived, fallback = [], 0

    for idx in range(len(buildings)):
        footprint = labels == idx + 1
        values = raw_values[footprint]
        values = values[np.isfinite(values) & (values >= 2.5) & (values <= 60)]
        if values.size:
            # Upper-quartile roof return is robust to ground/gap cells at edges.
            building_height = float(np.percentile(values, 75))
        else:
            feature = str(buildings.iloc[idx].get("FEATURE_TY", "")).upper()
            building_height = 4.5 if "RESIDENTIAL" in feature else 7.0
            fallback += 1
        building_height = float(np.clip(building_height, 2.5, 60))
        obstacle[footprint] = building_height
        derived.append(building_height)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.update(
        dtype="float32", nodata=NODATA, compress="deflate", predictor=3,
        tiled=True, blockxsize=256, blockysize=256,
    )
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(obstacle, 1)
        dst.set_band_description(1, "Building height above ground")
        dst.update_tags(
            source_footprints=str(args.buildings),
            source_height="2018 USGS 3DEP LiDAR raw CHM",
            method="75th percentile of 1 m LiDAR height cells per footprint",
            vertical_units="metres",
        )

    qa = {
        "building_footprints_in_aoi": len(buildings),
        "lidar_derived_heights": len(buildings) - fallback,
        "fallback_heights": fallback,
        "median_building_height_m": float(np.median(derived)),
        "p95_building_height_m": float(np.percentile(derived, 95)),
        "maximum_building_height_m": float(np.max(derived)),
        "note": "Source building_* attributes were all zero; heights were derived from LiDAR within each footprint.",
    }
    qa_path = args.output.with_name(args.output.stem + "_QA.json")
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
