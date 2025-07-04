import numpy as np
import pytest

from uot.data.measure import DiscreteMeasure, GridMeasure


def test_discrete_measure_basic():
    points = np.array(
        [
            [0.0, 0.0],  # 2d points
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    weights = np.array([0.3, 0.3, 0.4])

    dm = DiscreteMeasure(points, weights)
    points_out, weights_out = dm.to_discrete()
    assert np.allclose(points, points_out)
    assert np.allclose(weights, weights_out)


def test_grid_measure_flatten():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    weights = np.array(
        [
            [0.25, 0.25],
            [0.25, 0.25],
        ]
    )

    gm = GridMeasure([x, y], weights)
    points_out, weights_out = gm.to_discrete()
    expected_points = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    expected_weights = np.array(
        [
            0.25,
            0.25,
            0.25,
            0.25,
        ]
    )
    assert set(map(tuple, points_out.tolist())) == set(
        map(tuple, expected_points.tolist())
    )
    assert weights_out.shape == (4,)
    assert np.allclose(expected_weights, weights_out)

    uneven_weights = np.array(
        [
            [1, 2],
            [4, 3],
        ]
    )

    gm_uneven = GridMeasure([x, y], uneven_weights, normalize=True)
    points_out, weights_out = gm_uneven.to_discrete()
    expected_weights = np.array(
        [
            0.1,
            0.2,
            0.4,
            0.3,
        ]
    )

    assert np.allclose(expected_weights, weights_out)


def test_invalid_measure_types():
    points = np.array(
        [
            [0, 0],
            [1, 2],
        ]
    )
    # lengths should not match
    weights = np.array([0.4])

    with pytest.raises(AssertionError):
        DiscreteMeasure(points, weights)

    x = np.array([0, 1, 2])
    y = np.array([1])
    weights = np.ones((2, 2))  # again, invalid shapes
    with pytest.raises(AssertionError):
        GridMeasure([x, y], weights)
