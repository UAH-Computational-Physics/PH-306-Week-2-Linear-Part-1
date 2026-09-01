import pytest
from numpydoc.validate import validate

import linear


DOCSTRING_TARGETS = [
    linear.commutator,
    linear.are_perpendicular,
    linear.are_parallel,
    linear.are_commutative,
    linear.is_hermitian,
    linear.is_unitary,
    linear.is_linear_operator,
    linear.projection,
    linear.rotate_vector,
    linear.plane_from_points,
    linear.normal_line,
    linear.distance_point_to_plane,
    linear.distance_between_lines,
    linear.solve_cable_tension,
    linear.plot_cable_tension,
]

DOCSTRING_CHECKS = {
    "GL08",
    "PR01",
    "PR02",
    "PR03",
    "PR04",
    "PR05",
    "PR06",
    "PR07",
    "PR08",
    "PR09",
    "PR10",
    "RT01",
    "RT02",
    "RT03",
    "RT04",
    "RT05",
}


@pytest.mark.parametrize("func", DOCSTRING_TARGETS, ids=lambda func: func.__name__)
def test_numpy_style_docstrings_for_inputs_and_outputs(func):
    validation = validate(f"{func.__module__}.{func.__name__}")
    relevant_errors = [
        f"{code}: {message}"
        for code, message in sorted(validation["errors"])
        if code in DOCSTRING_CHECKS
    ]

    assert not relevant_errors, (
        f"{func.__name__} must document its inputs and outputs in NumPy style:\n"
        + "\n".join(relevant_errors)
    )
