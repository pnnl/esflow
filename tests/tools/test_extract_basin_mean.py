import numpy as np

from tools.extractors.extract_basin_mean import _make_basin_mask, _point_in_polygon


SQUARE = [[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5], [0.5, 0.5]]


def test_point_in_polygon_identifies_interior_and_exterior_points():
    assert _point_in_polygon(1.0, 1.0, SQUARE)
    assert not _point_in_polygon(3.0, 1.0, SQUARE)


def test_make_basin_mask_marks_only_grid_cell_centers_inside_polygon():
    feature = {"geometry": {"type": "Polygon", "coordinates": [SQUARE]}}

    mask = _make_basin_mask(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 3.0]), feature)

    assert mask.dtype == bool
    assert mask.tolist() == [
        [False, False, False, False],
        [False, True, True, False],
        [False, True, True, False],
        [False, False, False, False],
    ]
