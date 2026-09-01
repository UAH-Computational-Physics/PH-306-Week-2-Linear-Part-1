"""PH 306 Week 2: Linear Algebra, Part 1.

Complete each function below. Use NumPy arrays for vector and matrix inputs and
outputs unless the function documentation specifies otherwise.
"""

# --- Imports --- #
# Built-in Libraries
from numpy.typing import NDArray

# Numerical Libraries
import numpy as np

# Local Utilities
from plotutil import colored_line_between_pts


# Type Hints
Array = NDArray[np.float64]


def commutator(first: Array, second: Array) -> Array:
    """Compute the matrix commutator $[A, B] = AB - BA$."""
    raise NotImplementedError("Implement commutator")


def are_perpendicular(first, second, tolerance=1e-10):
    """Determine whether two vectors are perpendicular."""
    raise NotImplementedError("Implement are_perpendicular")


def are_parallel(first, second, tolerance=1e-10):
    """Determine whether two nonzero vectors are parallel."""
    raise NotImplementedError("Implement are_parallel")


def are_commutative(first, second, tolerance=1e-10):
    """Determine whether two matrices commute."""
    raise NotImplementedError("Implement are_commutative")


def is_hermitian(matrix, tolerance=1e-10):
    """Determine whether a matrix is Hermitian."""
    raise NotImplementedError("Implement is_hermitian")


def is_unitary(matrix, tolerance=1e-10):
    """Determine whether a matrix is unitary."""
    raise NotImplementedError("Implement is_unitary")


def is_linear_operator(matrix, tolerance=1e-10):
    """Determine whether a matrix represents a linear operator."""
    raise NotImplementedError("Implement is_linear_operator")


def projection(vector_a, vector_b):
    """Project vector $\\vec{A}$ onto vector $\\vec{B}$."""
    raise NotImplementedError("Implement projection")


def rotate_vector(vector, axis, theta):
    """Rotate a vector through angle theta about an axis in $\\mathbb{R}^3$."""
    raise NotImplementedError("Implement rotate_vector")


def plane_from_points(first, second, third):
    """Find the plane through three noncollinear points."""
    raise NotImplementedError("Implement plane_from_points")


def normal_line(point, normal):
    """Find the line through a point perpendicular to a plane."""
    raise NotImplementedError("Implement normal_line")


def distance_point_to_plane(point, normal, offset):
    """Find the minimum distance from a point to a plane."""
    raise NotImplementedError("Implement distance_point_to_plane")


def distance_between_lines(first_point, first_direction, second_point, second_direction):
    """Find the minimum distance between two lines in $\\mathbb{R}^3$."""
    raise NotImplementedError("Implement distance_between_lines")


def solve_cable_tension(N, L, rho_func, g=9.8):
    """Solve for the tension in a hanging cable discretized into N segments."""
    raise NotImplementedError("Implement solve_cable_tension")


def plot_cable_tension(z, T, L):
    """Plot the tension along a hanging cable, colored by tension magnitude."""
    raise NotImplementedError("Implement plot_cable_tension")
