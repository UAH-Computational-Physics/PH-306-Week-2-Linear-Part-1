"""Visible tests for the PH 306 Week 2 assignment.

The autograder includes additional tests. These checks document representative
inputs and outputs without implementing the assignment for students.
"""

import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

import linear


def test_commutator() -> None:
    """Check the matrix commutator $[A, B] = AB - BA$."""
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    expected = np.array([[0.0, -2.0], [2.0, 0.0]])
    assert np.allclose(linear.commutator(pauli_x, pauli_z), expected)


def test_are_commutative() -> None:
    """Check whether two matrices commute."""
    assert linear.are_commutative(np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_is_hermitian() -> None:
    """Check whether a matrix is Hermitian."""
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert linear.is_hermitian(pauli_x)


def test_is_unitary() -> None:
    """Check whether a matrix is unitary."""
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert linear.is_unitary(pauli_x)


def test_is_linear_operator() -> None:
    """Check whether a matrix represents a linear operator."""
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert linear.is_linear_operator(pauli_x)


def test_are_perpendicular() -> None:
    """Check whether two vectors are perpendicular."""
    assert linear.are_perpendicular(np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0]))


def test_are_parallel() -> None:
    """Check whether two nonzero vectors are parallel."""
    assert linear.are_parallel(np.array([1.0, -2.0, 0.0]), np.array([-3.0, 6.0, 0.0]))


def test_projection() -> None:
    """Check the projection of one vector onto another."""
    assert np.allclose(
        linear.projection(np.array([3.0, 4.0]), np.array([1.0, 0.0])),
        [3.0, 0.0],
    )
    assert np.allclose(
        linear.projection(np.array([1, 0]), np.array([0, 1])),
        [0.0, 0.0],
    )


def test_rotate_vector() -> None:
    """Check the rotation of a vector about an axis."""
    assert np.allclose(
        linear.rotate_vector(
            np.array([1.0, 0.0, 0.0]), 2, np.pi / 2
        ),
        [0.0, 1.0, 0.0],
    )


def test_plane_from_points() -> None:
    """Check plane generation from three noncollinear points."""
    normal, offset = linear.plane_from_points(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert np.isclose(np.dot(normal, [1.0, 0.0, 0.0]), offset)


def test_distance_point_to_plane() -> None:
    """Check the minimum distance from a point to a plane."""
    assert np.isclose(
        linear.distance_point_to_plane(
            np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0]), 0.0
        ),
        2.0,
    )


def test_distance_between_lines() -> None:
    """Check the minimum distance between two lines in $\\mathbb{R}^3$."""
    assert np.isclose(
        linear.distance_between_lines(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        ),
        1.0,
    )


def test_solve_cable_tension() -> None:
    """Check that the cable solver preserves physical units and computes tension correctly."""
    segment_count = 10
    length = 10.0 * u.m
    gravity = 9.8 * u.m / u.s**2

    def density(height):
        return 2.0 * u.kg / u.m * (1.0 + height / length)

    positions, tension = linear.solve_cable_tension(
        segment_count, length, density, gravity
    )
    expected_positions = np.linspace(0.0, 10.0, segment_count + 1) * u.m
    midpoint_positions = (expected_positions[:-1] + expected_positions[1:]) / 2
    expected_tension = np.cumsum(
        density(midpoint_positions) * gravity * (length / segment_count)
    )

    assert_quantity_allclose(positions, expected_positions)
    assert_quantity_allclose(tension, expected_tension)


def test_plot_cable_tension() -> None:
    """Check that the cable plotter produces a properly labeled figure with colorbar."""
    import matplotlib.pyplot as plt

    positions = np.linspace(0.0, 10.0, 11)
    tension = np.linspace(19.6, 294.0, 10)
    figure, axes = linear.plot_cable_tension(positions, tension, 10.0)

    assert axes in figure.axes
    assert len(axes.collections) == 1
    assert axes.get_ylabel()
    assert "10" in axes.get_title()
    assert any(axis.get_ylabel() == "Tension (N)" for axis in figure.axes)
    plt.close(figure)
