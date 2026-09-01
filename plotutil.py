"""Provides plotting capabilities to students."""

# --- Imports --- #
# Standard Libraries
from typing import NDArray, Optional
import warnings

# Numerical Libraries
import numpy as np

# Plotting Libraries
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, Collection


# --- Typing --- #
ArrayLike = NDArray[np.float64] | list[float] | tuple[float, ...]
OptString = Optional[str]


# --- RC Params --- #
# Render all text (labels, titles, colorbars) with LaTeX
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})


# --- Functions --- #
# This is copied from the MatPlotLib documentation (with minor modifications)
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
def colored_line_between_pts(
        x: ArrayLike, y: ArrayLike, force: ArrayLike, ax: Axes,
        clabel: OptString = None, **lc_kwargs
    ) -> Collection:
    """
    Plot a line whose segments are colored by a force magnitude using the "inferno" colormap.

    It does this by creating a collection of line segments between each pair of
    neighboring points. The color of each segment is determined by the
    force values. Each segment is made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    force : array-like
        The force/tension for each segment, which **should have a size one less
        than that of x and y**. These values are mapped to the "inferno" colormap
        from their min to their max.
    ax : Axes
        Axis object on which to plot the colored line.
    clabel : str, optional
        Label to apply to the colorbar. If None, the colorbar is left unlabeled.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array or cmap keyword arguments
        because those are set from the force argument. If provided, they will be
        overridden.

    Returns
    -------
    line : matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')
    if "cmap" in lc_kwargs:
        warnings.warn('The provided "cmap" keyword argument will be overridden')

    # Check color array size (LineCollection still works, but values are unused)
    if len(force) != len(x) - 1:
        warnings.warn(
            "The force argument should have a length one less than the length of x and y. "
            "If it has the same length, use the colored_line function instead."
        )

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc_kwargs["cmap"] = "inferno"
    lc = LineCollection(segments, **lc_kwargs)

    # Set the values used for colormapping (min to max of the force array)
    lc.set_array(force)
    lc.set_clim(np.min(force), np.max(force))

    line = ax.add_collection(lc)
    ax.figure.colorbar(lc, ax=ax, label=clabel)

    return line
