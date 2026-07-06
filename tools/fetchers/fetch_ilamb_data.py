"""
Fetch observation datasets from the ILAMB data server.

Downloads a NetCDF file for a given variable and dataset from
https://www.ilamb.org/ILAMB-Data/. Caches locally to avoid re-downloading.
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import esmflow_tool, ToolSpec, Param

# Registry of known ILAMB datasets: (variable, dataset) -> relative URL path
ILAMB_DATASETS = {
    # Precipitation
    ('pr', 'GPCPv2.3'): 'pr/GPCPv2.3/pr.nc',
    ('pr', 'GPCCv2018'): 'pr/GPCCv2018/pr.nc',
    ('pr', 'CMAPv1904'): 'pr/CMAPv1904/pr.nc',
    # Evapotranspiration  (NetCDF variable inside these files is "et")
    ('evspsbl', 'MODIS'): 'evspsbl/MODIS/et_0.5x0.5.nc',
    ('evspsbl', 'GLEAMv3.3a'): 'evspsbl/GLEAMv3.3a/et.nc',
    # Runoff
    ('mrro', 'LORA'): 'mrro/LORA/LORA.nc',
    ('mrro', 'Dai'): 'mrro/Dai/runoff.nc',
    # LAI
    ('lai', 'MODIS'): 'lai/MODIS/lai_0.5x0.5.nc',
    ('lai', 'AVH15C1'): 'lai/AVH15C1/lai.nc',
    # GPP
    ('gpp', 'FLUXCOM'): 'gpp/FLUXCOM/gpp.nc',
    ('gpp', 'FLUXNET2015'): 'gpp/FLUXNET2015/gpp.nc',
    # Surface air temperature
    ('tas', 'CRU4.02'): 'tas/CRU4.02/tas.nc',
    # Terrestrial water storage
    ('twsa', 'GRACE'): 'twsa/GRACE/twsa_0.5x0.5.nc',
    # Snow water equivalent
    ('swe', 'CanSISE'): 'swe/CanSISE/swe.nc',
    # Net radiation
    ('rns', 'CERESed4.2'): 'rns/CERESed4.2/rns.nc',
    # Latent heat
    ('hfls', 'FLUXCOM'): 'hfls/FLUXCOM/le.nc',
    # Sensible heat
    ('hfss', 'FLUXCOM'): 'hfss/FLUXCOM/sh.nc',
}

ILAMB_BASE_URL = 'https://www.ilamb.org/ILAMB-Data/DATA'

SPEC = ToolSpec(
    name='fetch_ilamb_data',
    description=(
        'Fetch an observation dataset from the ILAMB data server '
        '(https://www.ilamb.org/ILAMB-Data/). Downloads a gridded NetCDF file '
        'for a given variable and dataset. Caches locally so re-runs skip the download. '
        'Available variable/dataset pairs: '
        'pr (GPCPv2.3, GPCCv2018, CMAPv1904), '
        'evspsbl (MODIS, GLEAMv3.3a) — NOTE: the NetCDF variable inside ET files is "et" not "evspsbl", '
        'mrro (LORA, Dai), '
        'lai (MODIS, AVH15C1), '
        'gpp (FLUXCOM, FLUXNET2015), '
        'tas (CRU4.02), '
        'twsa (GRACE), '
        'swe (CanSISE), '
        'rns (CERESed4.2), '
        'hfls (FLUXCOM), '
        'hfss (FLUXCOM).'
    ),
    inputs={
        'variable': Param('str', required=True,
                          description='CMIP variable name (e.g., pr, evspsbl, mrro, lai, gpp, tas, twsa, swe)'),
        'dataset': Param('str', required=True,
                         description='Dataset name (e.g., GPCPv2.3, GLEAMv3.3a, LORA, MODIS, CRU4.02)'),
    },
    outputs={
        'data_file': {'type': 'netcdf', 'description': 'Downloaded observation NetCDF file'},
    },
)


@esmflow_tool(SPEC)
def run(config: dict) -> dict:
    variable = config['variable']
    dataset = config['dataset']
    output_dir = Path(config['output_dir'])

    key = (variable, dataset)
    if key not in ILAMB_DATASETS:
        available = [f"{v}/{d}" for v, d in sorted(ILAMB_DATASETS.keys())]
        raise ValueError(
            f"Unknown variable/dataset pair: {variable}/{dataset}. "
            f"Available: {available}"
        )

    rel_path = ILAMB_DATASETS[key]
    url = f"{ILAMB_BASE_URL}/{rel_path}"
    filename = f"{variable}_{dataset}.nc"
    out_path = output_dir / filename

    # Check cache
    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Cached: {out_path} ({size_mb:.1f} MB)")
        return {'data_file': str(out_path)}

    # Download
    print(f"  Downloading: {url}")
    print(f"  Destination: {out_path}")

    try:
        urllib.request.urlretrieve(url, str(out_path))
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Downloaded: {size_mb:.1f} MB")
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}")

    return {'data_file': str(out_path)}
