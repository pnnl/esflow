"""
Load observation gauge metadata from CSV.

Reads a gauge_metadata.csv with columns: gauge_id, lat, lon, area_km2, river_name.
Validates required columns and passes the file through.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esflow_tool, ToolSpec, Param

SPEC = ToolSpec(
    name='load_obs_metadata',
    description='Load observation gauge metadata CSV. '
                'Validates required columns (gauge_id, lat, lon, area_km2, river_name) '
                'and passes the file to downstream tools.',
    inputs={
        'metadata_file': Param('path', required=True,
                               description='Path to gauge_metadata.csv'),
    },
    outputs={
        'metadata_file': {'type': 'csv', 'description': 'Validated gauge metadata CSV'},
        'n_gauges': {'type': 'int', 'description': 'Number of gauges'},
    },
)


@esflow_tool(SPEC)
def run(config: dict) -> dict:
    metadata_file = config['metadata_file']

    df = pd.read_csv(metadata_file)
    print(f"  Loaded {len(df)} gauges from {metadata_file}")

    # Validate required columns
    required = ['gauge_id', 'lat', 'lon', 'area_km2', 'river_name']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Ensure gauge_id is string
    df['gauge_id'] = df['gauge_id'].astype(str)

    # Save validated copy to output_dir
    output_dir = Path(config['output_dir'])
    out_path = output_dir / 'gauge_metadata.csv'
    df.to_csv(out_path, index=False)

    print(f"  Columns: {list(df.columns)}")
    print(f"  Gauges: {len(df)}")

    return {
        'metadata_file': str(out_path),
        'n_gauges': len(df),
    }
