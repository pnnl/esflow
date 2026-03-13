"""
Spatial operations for ESM analysis.

Includes:
- Grid cell matching (nearest neighbor, area-based)
- River network tracing (upstream/downstream)
- Region extraction
- Coordinate transformations
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Tuple, Union


def find_nearest_cell(
    target_lat: float,
    target_lon: float,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    mesh_type: str = 'latlon'
) -> Tuple[int, ...]:
    """
    Find nearest grid cell to target coordinates.

    Args:
        target_lat: Target latitude
        target_lon: Target longitude
        lat_grid: Grid latitudes (1D or 2D)
        lon_grid: Grid longitudes (1D or 2D)
        mesh_type: 'latlon' for structured, 'unstructured' for unstructured

    Returns:
        Tuple of indices (lat_idx, lon_idx) for latlon or (cell_idx,) for unstructured
    """
    # Normalize longitude to -180 to 180
    target_lon = ((target_lon + 180) % 360) - 180

    if mesh_type == 'unstructured':
        # 1D arrays of cell centers
        lat_1d = np.asarray(lat_grid).flatten()
        lon_1d = np.asarray(lon_grid).flatten()
        lon_1d = ((lon_1d + 180) % 360) - 180

        # Haversine distance
        dlat = np.radians(lat_1d - target_lat)
        dlon = np.radians(lon_1d - target_lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(target_lat)) * \
            np.cos(np.radians(lat_1d)) * np.sin(dlon/2)**2
        dist = 2 * np.arcsin(np.sqrt(a))

        idx = np.argmin(dist)
        return (idx,)

    else:
        # Structured lat-lon grid
        lat_1d = np.asarray(lat_grid).flatten() if lat_grid.ndim == 1 else lat_grid[:, 0]
        lon_1d = np.asarray(lon_grid).flatten() if lon_grid.ndim == 1 else lon_grid[0, :]
        lon_1d = ((lon_1d + 180) % 360) - 180

        lat_idx = np.argmin(np.abs(lat_1d - target_lat))
        lon_idx = np.argmin(np.abs(lon_1d - target_lon))

        return (int(lat_idx), int(lon_idx))


def match_point_to_grid(
    obs_lat: float,
    obs_lon: float,
    obs_area_km2: Optional[float],
    ds: xr.Dataset,
    search_radius: int = 1,
    area_tolerance: float = 0.2,  # 20% tolerance: <10% good, 10-20% fair, >20% reject
    area_var: str = 'areatotal2'
) -> Optional[Dict]:
    """
    Match an observation point to model grid using location and drainage area.

    Args:
        obs_lat: Observation latitude
        obs_lon: Observation longitude
        obs_area_km2: Observation drainage area in km²
        ds: Model dataset with coordinates and drainage area
        search_radius: Search radius in grid cells
        area_tolerance: Maximum allowed area difference (fraction)
        area_var: Variable name for drainage area

    Returns:
        Dict with match info or None if no match found
    """
    mesh_type = ds.attrs.get('mesh_type', 'latlon')

    # Get coordinate arrays
    if 'lat' in ds.coords:
        lat_grid = ds['lat'].values
        lon_grid = ds['lon'].values
    elif 'lat' in ds.data_vars:
        lat_grid = ds['lat'].values
        lon_grid = ds['lon'].values
    else:
        raise ValueError("Cannot find lat/lon coordinates in dataset")

    # Find nearest cell
    nearest_idx = find_nearest_cell(obs_lat, obs_lon, lat_grid, lon_grid, mesh_type)

    if mesh_type == 'unstructured':
        cell_idx = nearest_idx[0]
        model_lat = float(lat_grid.flat[cell_idx])
        model_lon = float(lon_grid.flat[cell_idx])

        # Get drainage area if available
        if area_var in ds and obs_area_km2 is not None:
            model_area_m2 = float(ds[area_var].values.flat[cell_idx])
            model_area_km2 = model_area_m2 / 1e6
            area_error = abs(model_area_km2 - obs_area_km2) / obs_area_km2

            if area_error > area_tolerance:
                return None
        else:
            model_area_km2 = np.nan
            area_error = np.nan

        return {
            'lat_model': model_lat,
            'lon_model': model_lon,
            'lat_idx': cell_idx,
            'lon_idx': cell_idx,  # Same for unstructured
            'area_model_km2': model_area_km2,
            'area_error_pct': area_error * 100 if not np.isnan(area_error) else np.nan,
            'match_quality': 'good' if area_error < 0.1 else 'fair',  # <10% good, 10-20% fair
        }

    else:
        # Structured grid
        lat_idx, lon_idx = nearest_idx

        # Search in neighborhood for best area match
        best_match = None
        best_area_error = float('inf')

        nj, ni = lat_grid.shape if lat_grid.ndim == 2 else (len(lat_grid), len(lon_grid))

        for di in range(-search_radius, search_radius + 1):
            for dj in range(-search_radius, search_radius + 1):
                i = lat_idx + di
                j = lon_idx + dj

                if i < 0 or i >= nj or j < 0 or j >= ni:
                    continue

                if lat_grid.ndim == 2:
                    cell_lat = float(lat_grid[i, j])
                    cell_lon = float(lon_grid[i, j])
                else:
                    cell_lat = float(lat_grid[i])
                    cell_lon = float(lon_grid[j])

                # Check area match
                if area_var in ds and obs_area_km2 is not None:
                    if ds[area_var].ndim == 2:
                        model_area_m2 = float(ds[area_var].values[i, j])
                    else:
                        model_area_m2 = float(ds[area_var].values.flat[i * ni + j])

                    model_area_km2 = model_area_m2 / 1e6
                    area_error = abs(model_area_km2 - obs_area_km2) / max(obs_area_km2, 1)

                    if area_error < best_area_error:
                        best_area_error = area_error
                        best_match = {
                            'lat_model': cell_lat,
                            'lon_model': cell_lon,
                            'lat_idx': int(i),
                            'lon_idx': int(j),
                            'area_model_km2': model_area_km2,
                            'area_error_pct': area_error * 100,
                        }
                else:
                    # No area matching, use nearest
                    if di == 0 and dj == 0:
                        best_match = {
                            'lat_model': cell_lat,
                            'lon_model': cell_lon,
                            'lat_idx': int(i),
                            'lon_idx': int(j),
                            'area_model_km2': np.nan,
                            'area_error_pct': np.nan,
                        }

        if best_match is None:
            return None

        if best_area_error > area_tolerance:
            return None

        best_match['match_quality'] = 'good' if best_area_error < 0.1 else 'fair'  # <10% good, 10-20% fair
        return best_match


def get_drainage_area(
    ds: xr.Dataset,
    lat_idx: int,
    lon_idx: int,
    area_var: str = 'areatotal2'
) -> float:
    """
    Get drainage area at a grid cell.

    Args:
        ds: Dataset with drainage area
        lat_idx: Latitude index
        lon_idx: Longitude index
        area_var: Variable name for drainage area

    Returns:
        Drainage area in km²
    """
    if area_var not in ds:
        return np.nan

    area = ds[area_var]

    if area.ndim == 2:
        return float(area.values[lat_idx, lon_idx]) / 1e6
    elif area.ndim == 1:
        return float(area.values[lat_idx]) / 1e6
    else:
        return np.nan


def trace_upstream(
    ds: xr.Dataset,
    start_lat: float,
    start_lon: float,
    max_cells: int = 5000,
    min_area_km2: float = 100,
    area_var: str = 'areatotal2'
) -> List[Dict]:
    """
    Trace river network upstream from a starting point.

    Uses drainage area to follow the main channel upstream,
    always choosing the neighbor with largest upstream area.

    Args:
        ds: Dataset with drainage area
        start_lat: Starting latitude (outlet)
        start_lon: Starting longitude (outlet)
        max_cells: Maximum cells to trace
        min_area_km2: Stop when area drops below this
        area_var: Variable name for drainage area

    Returns:
        List of dicts with cell info, ordered from outlet to headwater
    """
    mesh_type = ds.attrs.get('mesh_type', 'latlon')

    if mesh_type != 'latlon':
        raise NotImplementedError("Upstream tracing only supported for lat-lon grids")

    if area_var not in ds:
        raise ValueError(f"Drainage area variable '{area_var}' not found in dataset")

    # Get grids
    lat = ds['lat'].values
    lon = ds['lon'].values
    area = ds[area_var].values

    if lat.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon, lat)
    else:
        lat_2d, lon_2d = lat, lon

    nj, ni = lat_2d.shape

    # Find starting cell
    start_idx = find_nearest_cell(start_lat, start_lon, lat_2d, lon_2d, 'latlon')
    current_i, current_j = start_idx

    # Calculate grid spacing for distance
    dlat = abs(lat_2d[1, 0] - lat_2d[0, 0]) if nj > 1 else 0.5
    dlon = abs(lon_2d[0, 1] - lon_2d[0, 0]) if ni > 1 else 0.5
    km_per_deg = 111.0  # Approximate

    # Trace upstream
    path = []
    visited = set()
    total_distance = 0

    while len(path) < max_cells:
        if (current_i, current_j) in visited:
            break

        visited.add((current_i, current_j))

        cell_lat = float(lat_2d[current_i, current_j])
        cell_lon = float(lon_2d[current_i, current_j])
        cell_area = float(area[current_i, current_j]) / 1e6  # km²

        if cell_area < min_area_km2 and len(path) > 0:
            break

        path.append({
            'cell_id': len(path),
            'lat': cell_lat,
            'lon': cell_lon,
            'distance_km': total_distance,
            'drainage_area_km2': cell_area,
            'lat_idx': int(current_i),
            'lon_idx': int(current_j),
        })

        # Find upstream neighbor (largest area that's smaller than current)
        best_neighbor = None
        best_area = 0

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni_idx = current_i + di
                nj_idx = current_j + dj

                if ni_idx < 0 or ni_idx >= nj or nj_idx < 0 or nj_idx >= ni:
                    continue

                if (ni_idx, nj_idx) in visited:
                    continue

                neighbor_area = float(area[ni_idx, nj_idx]) / 1e6

                # Upstream cell should have smaller area
                if neighbor_area < cell_area and neighbor_area > best_area:
                    best_area = neighbor_area
                    best_neighbor = (ni_idx, nj_idx)

        if best_neighbor is None:
            break

        # Calculate distance to next cell
        next_lat = float(lat_2d[best_neighbor[0], best_neighbor[1]])
        next_lon = float(lon_2d[best_neighbor[0], best_neighbor[1]])
        step_dist = np.sqrt(
            ((next_lat - cell_lat) * km_per_deg) ** 2 +
            ((next_lon - cell_lon) * km_per_deg * np.cos(np.radians(cell_lat))) ** 2
        )
        total_distance += step_dist

        current_i, current_j = best_neighbor

    return path


def trace_downstream(
    ds: xr.Dataset,
    start_lat: float,
    start_lon: float,
    max_cells: int = 5000,
    area_var: str = 'areatotal2'
) -> List[Dict]:
    """
    Trace river network downstream from a starting point.

    Uses drainage area to follow flow downstream,
    always choosing the neighbor with largest downstream area.

    Args:
        ds: Dataset with drainage area
        start_lat: Starting latitude
        start_lon: Starting longitude
        max_cells: Maximum cells to trace
        area_var: Variable name for drainage area

    Returns:
        List of dicts with cell info, ordered from start to outlet
    """
    mesh_type = ds.attrs.get('mesh_type', 'latlon')

    if mesh_type != 'latlon':
        raise NotImplementedError("Downstream tracing only supported for lat-lon grids")

    if area_var not in ds:
        raise ValueError(f"Drainage area variable '{area_var}' not found")

    # Get grids
    lat = ds['lat'].values
    lon = ds['lon'].values
    area = ds[area_var].values

    if lat.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon, lat)
    else:
        lat_2d, lon_2d = lat, lon

    nj, ni = lat_2d.shape

    # Find starting cell
    start_idx = find_nearest_cell(start_lat, start_lon, lat_2d, lon_2d, 'latlon')
    current_i, current_j = start_idx

    # Grid spacing
    km_per_deg = 111.0

    # Trace downstream
    path = []
    visited = set()
    total_distance = 0

    while len(path) < max_cells:
        if (current_i, current_j) in visited:
            break

        visited.add((current_i, current_j))

        cell_lat = float(lat_2d[current_i, current_j])
        cell_lon = float(lon_2d[current_i, current_j])
        cell_area = float(area[current_i, current_j]) / 1e6

        path.append({
            'cell_id': len(path),
            'lat': cell_lat,
            'lon': cell_lon,
            'distance_km': total_distance,
            'drainage_area_km2': cell_area,
            'lat_idx': int(current_i),
            'lon_idx': int(current_j),
        })

        # Find downstream neighbor (largest area that's bigger than current)
        best_neighbor = None
        best_area = cell_area

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni_idx = current_i + di
                nj_idx = current_j + dj

                if ni_idx < 0 or ni_idx >= nj or nj_idx < 0 or nj_idx >= ni:
                    continue

                if (ni_idx, nj_idx) in visited:
                    continue

                neighbor_area = float(area[ni_idx, nj_idx]) / 1e6

                if neighbor_area > best_area:
                    best_area = neighbor_area
                    best_neighbor = (ni_idx, nj_idx)

        if best_neighbor is None:
            break

        # Distance
        next_lat = float(lat_2d[best_neighbor[0], best_neighbor[1]])
        next_lon = float(lon_2d[best_neighbor[0], best_neighbor[1]])
        step_dist = np.sqrt(
            ((next_lat - cell_lat) * km_per_deg) ** 2 +
            ((next_lon - cell_lon) * km_per_deg * np.cos(np.radians(cell_lat))) ** 2
        )
        total_distance += step_dist

        current_i, current_j = best_neighbor

    return path


def extract_region(
    ds: xr.Dataset,
    extent: Optional[List[float]] = None,
    mask: Optional[xr.DataArray] = None
) -> xr.Dataset:
    """
    Extract data for a geographic region.

    Args:
        ds: Input dataset
        extent: [lon_min, lon_max, lat_min, lat_max] or None for global
        mask: Optional mask DataArray

    Returns:
        Subset dataset
    """
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent

        # Handle different coordinate names
        if 'lat' in ds.coords and 'lon' in ds.coords:
            ds = ds.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max)
            )
        elif 'latitude' in ds.coords and 'longitude' in ds.coords:
            ds = ds.sel(
                latitude=slice(lat_min, lat_max),
                longitude=slice(lon_min, lon_max)
            )

    if mask is not None:
        ds = ds.where(mask)

    return ds
