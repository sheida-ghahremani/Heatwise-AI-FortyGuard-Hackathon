#!/usr/bin/env python3
"""Build 1 m DTM/DSM/CHM rasters for the HeatWise TAMU study area.

The 2018 Red River LAZ delivery has only class 1 (unclassified) and class 2
(ground), so vegetation cannot be selected by LAS class.  We build a surface
model from the highest class 1/2 returns, subtract a class-2 terrain model,
and use the College Station UTC polygon as the vegetation mask.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box


CRS = "EPSG:6343"
NODATA = -9999.0


def fill_nearest(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill missing raster cells using their nearest valid neighbor."""
    if not valid.any():
        raise RuntimeError("Raster has no valid cells to interpolate")
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[tuple(indices)]


def write_tif(path: Path, data: np.ndarray, transform, description: str, resolution_m: float) -> None:
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": CRS,
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
        dst.set_band_description(1, description)
        dst.update_tags(
            source="USGS 3DEP Red River 3Area B2 2018 LiDAR",
            horizontal_crs=CRS,
            vertical_datum="NAVD88 Geoid12B",
            vertical_units="metres",
            cell_size_metres=f"{resolution_m:g}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--laz-dir", type=Path, required=True)
    parser.add_argument("--canopy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--west", type=float, default=-96.352)
    parser.add_argument("--east", type=float, default=-96.332)
    parser.add_argument("--south", type=float, default=30.602)
    parser.add_argument("--north", type=float, default=30.626)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--prefix", default="TAMU")
    args = parser.parse_args()

    files = sorted(args.laz_dir.glob("*.laz"))
    if not files:
        raise FileNotFoundError(f"No LAZ files found in {args.laz_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tx = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    corners = [
        tx.transform(lon, lat)
        for lon in (args.west, args.east)
        for lat in (args.south, args.north)
    ]
    minx = min(p[0] for p in corners)
    maxx = max(p[0] for p in corners)
    miny = min(p[1] for p in corners)
    maxy = max(p[1] for p in corners)
    res = args.resolution
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    maxx = minx + width * res
    maxy = miny + height * res
    transform = from_origin(minx, maxy, res, res)
    size = width * height

    ground_sum = np.zeros(size, dtype="float64")
    ground_count = np.zeros(size, dtype="uint32")
    surface_max = np.full(size, -np.inf, dtype="float64")
    class_counts: Counter[int] = Counter()
    used_points = 0

    for path in files:
        with laspy.open(path) as reader:
            for pts in reader.chunk_iterator(1_000_000):
                x = np.asarray(pts.x)
                y = np.asarray(pts.y)
                z = np.asarray(pts.z)
                cls = np.asarray(pts.classification, dtype="uint8")
                inside = (x >= minx) & (x < maxx) & (y >= miny) & (y < maxy)
                if not inside.any():
                    continue
                x, y, z, cls = x[inside], y[inside], z[inside], cls[inside]
                used_points += len(z)
                class_counts.update(map(int, cls))
                col = ((x - minx) / res).astype("int64")
                row = ((maxy - y) / res).astype("int64")
                idx = row * width + col

                ground = cls == 2
                if ground.any():
                    np.add.at(ground_sum, idx[ground], z[ground])
                    np.add.at(ground_count, idx[ground], 1)

                surface = (cls == 1) | (cls == 2)
                if surface.any():
                    np.maximum.at(surface_max, idx[surface], z[surface])

    ground_valid = ground_count > 0
    dtm = np.zeros(size, dtype="float64")
    dtm[ground_valid] = ground_sum[ground_valid] / ground_count[ground_valid]
    dtm = fill_nearest(dtm.reshape(height, width), ground_valid.reshape(height, width))

    surface_valid = np.isfinite(surface_max)
    surface_grid = surface_max.reshape(height, width)
    surface_grid = fill_nearest(surface_grid, surface_valid.reshape(height, width))
    dsm = np.maximum(surface_grid, dtm)
    chm = np.clip(dsm - dtm, 0, 60)

    # Read and clip the highly detailed UTC polygons only after the LiDAR pass.
    canopy = gpd.read_file(args.canopy).to_crs(CRS)
    study_box = box(minx, miny, maxx, maxy)
    canopy = canopy[canopy.intersects(study_box)].copy()
    canopy["geometry"] = canopy.geometry.intersection(study_box)
    canopy = canopy[~canopy.geometry.is_empty]
    canopy_mask = rasterize(
        ((geom, 1) for geom in canopy.geometry),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    canopy_chm = np.where(canopy_mask & (chm >= 1.0), chm, NODATA)

    resolution_label = f"{res:g}m"
    write_tif(args.output_dir / f"{args.prefix}_DTM_{resolution_label}.tif", dtm, transform, "Terrain elevation", res)
    write_tif(args.output_dir / f"{args.prefix}_DSM_{resolution_label}.tif", dsm, transform, "Top surface elevation", res)
    write_tif(args.output_dir / f"{args.prefix}_CHM_raw_{resolution_label}.tif", chm, transform, "Raw height above ground", res)
    write_tif(
        args.output_dir / f"{args.prefix}_CanopyHeight_{resolution_label}.tif",
        canopy_chm,
        transform,
        "Tree canopy height above ground",
        res,
    )

    valid_heights = canopy_chm[canopy_chm != NODATA]
    report = {
        "laz_files": [p.name for p in files],
        "study_bounds_wgs84": [args.west, args.south, args.east, args.north],
        "study_bounds_epsg6343": [minx, miny, maxx, maxy],
        "raster_shape": [height, width],
        "resolution_m": res,
        "points_inside_aoi": used_points,
        "classification_counts_inside_aoi": dict(sorted(class_counts.items())),
        "ground_cell_coverage_fraction": float(ground_valid.mean()),
        "surface_cell_coverage_fraction": float(surface_valid.mean()),
        "canopy_mask_fraction": float(canopy_mask.mean()),
        "canopy_height_valid_cells": int(valid_heights.size),
        "canopy_height_median_m": float(np.median(valid_heights)) if valid_heights.size else None,
        "canopy_height_p95_m": float(np.percentile(valid_heights, 95)) if valid_heights.size else None,
        "method_note": (
            "The source lacks LAS vegetation classes 3-5. DSM uses maximum class 1/2 "
            "elevation; DTM uses mean class-2 ground elevation; CHM is masked by the "
            "College Station UTC polygon and heights below 1 m are excluded."
        ),
    }
    (args.output_dir / f"{args.prefix}_CanopyHeight_QA.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
