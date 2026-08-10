import numpy as np

from tools.core.spatial import find_nearest_cell


def test_find_nearest_cell_for_structured_1d_and_2d_grids():
    assert find_nearest_cell(9.0, -179.0, np.array([0.0, 10.0]), np.array([0.0, 180.0])) == (1, 1)

    lat_grid, lon_grid = np.meshgrid(np.array([0.0, 10.0]), np.array([0.0, 20.0]), indexing="ij")
    assert find_nearest_cell(8.0, 18.0, lat_grid, lon_grid) == (1, 1)


def test_find_nearest_cell_for_unstructured_grid_normalizes_longitude():
    assert find_nearest_cell(
        0.0,
        -179.0,
        np.array([0.0, 0.0]),
        np.array([0.0, 181.0]),
        mesh_type="unstructured",
    ) == (1,)
