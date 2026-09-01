"""Visible tests for the PH 306 Week 2 assignment.

The autograder includes additional tests. These checks document representative
inputs and outputs without implementing the assignment for students.
"""

import numpy as np

import linear


def test_commutator_and_matrix_properties() -> None:
    """Check representative matrix operations and properties."""
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    expected = np.array([[0.0, -2.0], [2.0, 0.0]])
    assert np.allclose(linear.commutator(pauli_x, pauli_z), expected)
    assert linear.are_commutative(np.eye(2), pauli_x)
    assert linear.is_hermitian(pauli_x)
    assert linear.is_unitary(pauli_x)
    assert linear.is_linear_operator(pauli_x)


def test_vector_relationships_projection_and_rotation() -> None:
    """Check vector relationships, projection, and a rotation about z."""
    assert linear.are_perpendicular(np.array([1.0, 0.0]), np.array([0.0, 2.0]))
    assert linear.are_parallel(np.array([1.0, -2.0]), np.array([-3.0, 6.0]))
    assert np.allclose(
        linear.projection(np.array([3.0, 4.0]), np.array([1.0, 0.0])),
        [3.0, 0.0],
    )
    assert np.allclose(
        linear.rotate_vector(
            np.array([1.0, 0.0, 0.0]), 2, np.pi / 2
        ),
        [0.0, 1.0, 0.0],
    )


def test_three_dimensional_geometry() -> None:
    """Check representative planes, normal lines, and distances."""

    # Check Plane Generation from Points
    normal, offset = linear.plane_from_points(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert np.isclose(np.dot(normal, [1.0, 0.0, 0.0]), offset)

    # Check Distance from Point to Plane
    assert np.isclose(
        linear.distance_point_to_plane(
            np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0]), 0.0
        ),
        2.0,
    )

    # Check the Minimum Distance Between Two Lines
    assert np.isclose(
        linear.distance_between_lines(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        ),
        1.0,
    )
