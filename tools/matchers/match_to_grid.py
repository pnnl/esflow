"""
Match observation gauges to E3SM model grid cells.

Uses gauge coordinates and drainage area to find the best matching
model grid cell. Produces a matched_gauges CSV with model indices.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param
from core.e3sm import find_e3sm_files, open_e3sm_dataset
from core.spatial import match_point_to_grid

SPEC = ToolSpec(
    name='match_to_grid',
    description='Match observation gauges to E3SM model grid cells using '
                'coordinates and drainage area. Produces matched_gauges CSV '
                'with model grid indices for time series extraction.',
    inputs={
        'gauge_metadata': Param('path', required=True,
                                description='Path to gauge_metadata.csv'),
        'data_dir': Param('path', required=True,
                          description='Base directory containing E3SM case output'),
        'case_name': Param('str', required=True,
                           description='E3SM case name'),
        'component': Param('str', required=True,
                           description='Model component (mosart, elm, eam)'),
    },
    outputs={
        'matched_file': {'type': 'csv', 'description': 'Matched gauges CSV with model grid indices'},
        'n_matched': {'type': 'int', 'description': 'Number of successfully matched gauges'},
    },
)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    gauge_metadata = config['gauge_metadata']
    data_dir = config['data_dir']
    case_name = config['case_name']
    component = config['component']
    output_dir = Path(config['output_dir'])

    # Load metadata
    meta = pd.read_csv(gauge_metadata)
    meta['gauge_id'] = meta['gauge_id'].astype(str)
    print(f"  Gauges to match: {len(meta)}")

    # Open one model file to get grid
    files = find_e3sm_files(data_dir, case_name, component, years=[])
    ds = open_e3sm_dataset(files[:1])
    print(f"  Model grid: {ds.attrs.get('mesh_type', 'unknown')}")

    # Match each gauge
    results = []
    for _, row in meta.iterrows():
        match = match_point_to_grid(
            obs_lat=row['lat'],
            obs_lon=row['lon'],
            obs_area_km2=row.get('area_km2'),
            ds=ds,
        )

        if match is not None:
            results.append({
                'gauge_id': row['gauge_id'],
                'lat': row['lat'],
                'lon': row['lon'],
                'model_lat': match['lat_model'],
                'model_lon': match['lon_model'],
                'lat_idx': match['lat_idx'],
                'lon_idx': match['lon_idx'],
            })

    ds.close()

    matched_df = pd.DataFrame(results)
    print(f"  Matched: {len(matched_df)} of {len(meta)}")

    out_path = output_dir / 'matched_gauges.csv'
    matched_df.to_csv(out_path, index=False)

    return {
        'matched_file': str(out_path),
        'n_matched': len(matched_df),
    }
