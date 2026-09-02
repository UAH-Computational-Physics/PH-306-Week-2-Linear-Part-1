# PH 306 Week 2: Linear Algebra, Part 1

## Homework

### General

1. Calculate a matrix commutator with `commutator`.
1. Determine whether two vectors are perpendicular with `are_perpendicular`.
1. Determine whether two nonzero vectors are parallel with `are_parallel`.
1. Determine whether two matrices commute with `are_commutative`.
1. Determine whether a matrix is Hermitian with `is_hermitian` without `scipy`.
    - Raise a `ValueError` if the matrix is not square.
1. Determine whether a matrix is unitary with `is_unitary` without `scipy`.
    - Raise a `ValueError` if the matrix is not square.
1. Implement `is_linear_operator` for a matrix representing a transformation of a vector space to itself.
    - See Boas Section 3.7.
1. Find $\operatorname{proj}_{\vec{B}}\vec{A}$ with `projection`.
    - Raise a `ValueError` if either vector is the zero vector.
1. Rotate a vector about a coordinate axis by angle $\theta$ in $\mathbb{R}^3$ with `rotate_vector`.
    - Pass `axis=0`, `axis=1`, or `axis=2` for rotations about the $x$, $y$, or $z$ axis, respectively.
    - Raise a `ValueError` if `axis` is not one of `0`, `1`, or `2`.
1. Complete the following exercises from Boas:
    - Example 3.5.1: find a plane through three points with `plane_from_points`.
        - Return values for when the equation is in standard form (everything set to zero)
        - Return the normal to the plane as a unit vector in $\mathbb{R}_3$.
        - Return the offset/constant as a scalar.
    - Example 3.5.3: find the distance from a point to a plane with `distance_point_to_plane`.
    - Example 3.5.5: find the minimum distance between two lines with `distance_between_lines`.

For each function, follow the parameter and return-value contract in its
docstring. Angles are in radians. For numerical comparisons, respect the
provided `tolerance` argument. Raise a suitable exception for invalid input,
such as a zero projection direction or an incorrectly shaped vector.

### Physical: Numerical Solution & Visualization of a Hanging Cable

#### Purpose

In this assignment, you will implement a computational model for the static force equilibrium of a hanging cable with non-uniform mass density. You will write a general matrix solver to solve the discretized system for arbitrary segment counts $N$ and visualize the internal stress state of the physical system using continuous color mapping.

#### Theoretical Background

Consider a cable of length $L$ hanging vertically from $z = L$ down to $z = 0$, with height-dependent mass density $\rho(z) = \rho_0 \left(1 + \frac{z}{L}\right)$. Discretizing the cable into $N$ segments of length $\Delta z = L/N$ yields an linear system where $$T_i - T_{i-1} - \rho(z_i) g \Delta z = 0 $$, $i=0$ represents the lowest segment, $i=N-1$ represents the top segment, and $z_i = \Delta z (i + 1/2)$.

#### Task 1: General Linear System Solver Function

Write a Python function `solve_cable_tension(N, L, rho, g=EARTH_GRAVITY)` that accepts structural parameters and returns the numerical spatial positions $z$ and tension values $T$. The `rho` parameter is a callable: a function that accepts a scalar or NumPy array of heights and returns the corresponding mass density. For the specified density profile, define it with a regular function:

```python
def rho(z):
    return rho_0 * (1.0 + z / L)
```

- Dynamically build the $N \times N$ coefficient matrix $A$ (see [`np.diag`](https://numpy.org/doc/stable/reference/generated/numpy.diag.html)) and construct vector $\vec{b}$ based on the evaluated gravitational forces `rho(z_i) * g * delta_z` at $z_i$.
- Compute tension vector $\vec{T}$ using an efficient linear algebra algorithm (such as `numpy.linalg.solve` or custom back-end substitution).
- Return the array of tensions $\vec{T}$ (length $N$) along with the array of segment boundary positions $\vec{z}$ (length $N+1$, from $z=0$ to $z=L$), so that $T_i$ represents the tension of the segment between $z_i$ and $z_{i+1}$.
- The function must support `astropy.units.Quantity` inputs for `L`, `g`, and the values returned by `rho`. When quantities are provided, return `z` with length units and `T` with force units.
- `numpy.linalg.solve` operates most reliably on unitless arrays. The coefficient matrix $\mathbf{A}$ is dimensionless, while $b$ has force units. Before calling the solver, convert the force vector to SI values with `b.si.value`; after solving, restore the force units with `tension_values * u.N`.

#### Task 2: Visualization Function

Write a Python function `plot_cable_tension(z, T, L)` that generates a visual representation of the cable.

- Plot the 1D cable vertically (with $x=0$ and height $z$). Use the provided `colored_line_between_pts` helper from `plotutil.py` (or a `matplotlib` `LineCollection` / continuous scatter mapping) where line segments are colored according to tension $T_i$.
- Include a colorbar labeled `"Tension (N)"`, clear axis labels ($z$ position in meters), a descriptive plot title showing segment count $N$ (the `inferno` colormap is used by `colored_line_between_pts`).
- The function should return the figure and axes objects.
- The function must accept `z` and `T` as `astropy.units.Quantity` arrays. Convert quantities to values in compatible display units before passing them to Matplotlib.
- Solution will be auto-graded against a plot that will be provided via Canvas.

#### Test Case & Validation

Run your script using $L = 10\,\mathrm{m}$, $\rho_0 = 2\,\mathrm{kg/m}$, $g = 9.8\,\mathrm{m/s}^2$, and compare solutions across $N = 5, 20, 100$. Overlay or plot beside your solution the continuous exact analytical solution:

$$T_{\,\mathrm{exact}}(z) = \int_0^z \rho(z') g \, dz' = \rho_0 g \left(z + \frac{z^2}{2L}\right)$$

## Running checks

From the repository root, run:

```bash
python -m pytest test_public.py test_public_docs.py
python -m mypy --config-file mypy.ini --strict linear.py
```

CodeGrade runs comparable checks after submission.
