"""Volumetric-positioning plots: circuit width against circuit depth, shaded by fidelity.

One canvas, one meaning: a cell at (depth, width) shaded by how well the device
answered a circuit of that shape. The framework is arXiv:2110.03137's; the
rendering follows QC-App-Oriented-Benchmarks (QED-C, Apache-2.0), from which the
faded-spectral colormap, the logarithmic depth index and the label-spreading
annotation pass are ported.

The depth axis is logarithmic, and that is the load-bearing choice. An ordinal
axis of the depths one algorithm happens to visit cannot hold a second
algorithm: two sweeps share no column basis, so there is nothing to overlay
them onto. ``depth_index`` maps any depth onto one fixed axis, which is what
lets ``plot_volumetric`` render a single algorithm and four algorithms with the
same code path -- the single-algorithm plot is just the one-series case.

Cells are drawn as patches rather than as an image, for the same reason: patches
sit at arbitrary coordinates and may overlap, where ``imshow`` demands a dense
array on a regular lattice.

Nothing here reads module state or touches the current figure, so a caller may
build several plots without them interfering.

Vendored from `eq1bench/volumetric.py` at eq1val rev db660ef -- see this
package's `__init__.py` for what that means for keeping it current.
"""

import math
from dataclasses import dataclass

from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.pyplot import Figure

# Base of the depth axis: one tick per doubling. Ported from QED-C, where the
# comment notes that non-integer bases stretch the axis out but round badly.
DEPTH_BASE = 2

# Below this fidelity the ramp desaturates toward grey, so a cell the device
# effectively failed recedes instead of reading as a measurement in its own
# right. QED-C's defaults, kept because the fade is the point of the ramp.
FADE_LEVEL = 0.16
FADE_RATE = 0.7

LATTICE_FILL = (0.95, 0.95, 0.95)  # reachable (width, depth) shapes
LATTICE_EDGE = (0.75, 0.75, 0.75)
CEILING_FILL = (1.0, 1.0, 1.0)  # first shape past the depth ceiling
CELL_EDGE = (0.5, 0.5, 0.5)
INK_DARK = "#22221f"
INK_LIGHT = "#ffffff"
ANNO_INK = (0.2, 0.2, 0.2)

LATTICE_SIZE = 0.6  # side of a background square, in axis units
CELL_SIZE = 1.0  # side of a data square


def create_faded_spectral_cmap(
    fade_level: float = FADE_LEVEL, fade_rate: float = FADE_RATE
) -> ListedColormap:
    """Spectral, with its low end bleached toward grey. Ported from QED-C.

    Plain Spectral gives a dead cell a saturated red, which draws the eye to
    exactly the cells carrying the least information. Below ``fade_level`` the
    colours are pulled toward their own red channel -- desaturating them without
    shifting hue -- on a curve set by ``fade_rate``, so the bottom of the scale
    fades out rather than shouting.
    """
    steps = 100
    knee = round(fade_level * steps)
    spectral = colormaps["Spectral"]
    colors = [spectral(v / steps) for v in range(steps)]

    faded = []
    for i in range(knee):
        # Normalised position within the faded band: 0 at the floor, 1 at the knee.
        x = i / knee
        r0, g0, b0, a0 = colors[i]

        # Desaturate toward a light red, most strongly at the floor.
        r = r0 + (0.92 - r0) * (1 - x)
        # ...and pull green and blue toward it, which is what removes the hue.
        grey_ratio = 1 - math.pow(x, 1 / fade_rate)
        g = g0 + (r - g0) * grey_ratio
        b = b0 + (r - b0) * grey_ratio
        faded.append((r, g, b, a0))

    return ListedColormap(faded + colors[knee:])


def depth_index(depth: float) -> float:
    """Where a circuit depth sits on the axis: one unit per doubling.

    Depth 1 lands at 1, 2 at 2, 4 at 3, and so on, so the integer ticks are
    exact powers of two and every depth in between has a place. Depth 0 is
    pinned to 0 rather than diverging.
    """
    return 0.0 if depth <= 0 else math.log(depth, DEPTH_BASE) + 1


def format_depth(depth: int) -> str:
    """A depth as an axis label, abbreviated past a thousand. Ported from QED-C.

    A full sweep puts a depth of two million on the same axis as a depth of one,
    and the wide end is what needs the room.
    """
    for suffix, size in (("M", 1_000_000), ("K", 1_000)):
        if depth >= size:
            return f"{depth / size:.1f}".rstrip("0").rstrip(".") + suffix
    return str(depth)


@dataclass(frozen=True)
class Cell:
    """One circuit shape and what it scored.

    ``fidelity is None`` means the shape was attempted and produced no number --
    a different and much weaker claim than a measured zero, and drawn
    differently for it.
    """

    width: int
    depth: int
    fidelity: float | None = None


def _cell_patch(x: float, y: float, color, size: float = CELL_SIZE) -> Rectangle:
    """A filled data cell centred on (x, y)."""
    return Rectangle(
        (x - size / 2, y - size / 2),
        size,
        size,
        edgecolor=CELL_EDGE,
        facecolor=color,
        fill=True,
        lw=0.5,
        zorder=3,
    )


def _lattice_patch(x: float, y: float, facecolor) -> Rectangle:
    """A background square marking a shape the device could in principle reach."""
    return Rectangle(
        (x - LATTICE_SIZE / 2, y - LATTICE_SIZE / 2),
        LATTICE_SIZE,
        LATTICE_SIZE,
        edgecolor=LATTICE_EDGE,
        facecolor=facecolor,
        fill=True,
        lw=0.5,
        zorder=1,
    )


def _ink_for(color) -> str:
    """Readable text colour over a cell, by the luminance of the cell itself.

    The ramp runs red-to-blue through a bright yellow, so neither white nor dark
    ink works across the whole of it and the choice has to be made per cell.
    """
    r, g, b = color[:3]
    return INK_DARK if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else INK_LIGHT


def _draw_lattice(ax, max_width: int, max_depth: int, decades: int) -> None:
    """The reference field of shapes the device could reach.

    Every claim the lattice makes is one the caller can defend: it stops at the
    device's qubit count and at the depth ceiling the sweep actually enforced.
    The hollow square in each row marks the first shape past that ceiling -- the
    boundary is drawn rather than implied by where the field runs out.

    QED-C derives this region from a Quantum Volume model instead. That is
    deliberately not ported: with its default settings the boundary comes from
    an assumed QV of 2048 and a transpile factor of 12.7, which would put an
    invented performance claim next to measured data.
    """
    for width in range(1, max_width + 1):
        beyond = None
        for tick in range(1, decades + 1):
            depth = DEPTH_BASE ** (tick - 1)
            if depth > max_depth:
                beyond = tick
                break
            ax.add_patch(_lattice_patch(tick, width, LATTICE_FILL))
        if beyond is not None:
            ax.add_patch(_lattice_patch(beyond, width, CEILING_FILL))


@dataclass
class _Label:
    """A series label in flight: `y` is nudged upward while collisions resolve."""

    x: float
    y: float
    text: str
    anchor_y: float


def _annotate(
    ax, anchors: list[tuple[float, float, str]], cells: list[tuple[float, float]]
) -> float:
    """Label each series above its widest cell, and return the height used.

    Straight up is the one offset direction with somewhere to go: the plot fills
    from the bottom left, so above a series' widest cell is the nearest reliably
    clear space, and the arrow stays short enough to read as a pointer rather
    than as plot furniture.

    QED-C instead pushes labels radially onto a common circle around the origin.
    That relies on their much taller plots leaving a clear rim; at the widths
    swept here it drops labels straight onto neighbouring cells.

    Clearing the anchor is not enough on a combined grid, because a *different*
    series may occupy that column higher up -- so a label rises above every cell
    near its column, not just its own. Labels that still collide are then walked
    upward one row at a time, which is the case where two algorithms stop at the
    same width: the comparison the grid exists to show.
    """
    if not anchors:
        return 0.0

    placed = []
    for x, y, text in sorted(anchors):
        nearby = [cell_y for cell_x, cell_y in cells if abs(cell_x - x) < 1.0]
        placed.append(
            _Label(x=x, y=max(nearby, default=y) + 1.3, text=text, anchor_y=y)
        )

    for i, entry in enumerate(placed):
        for previous in placed[:i]:
            if abs(entry.x - previous.x) < 2.2 and abs(entry.y - previous.y) < 1.05:
                entry.y = previous.y + 1.05

    for entry in placed:
        ax.annotate(
            entry.text,
            xy=(entry.x, entry.anchor_y + 0.45),
            xytext=(entry.x, entry.y),
            arrowprops={
                "facecolor": ANNO_INK,
                "shrink": 0.05,
                "width": 0.4,
                "headwidth": 4,
                "headlength": 5,
                "edgecolor": (0.8, 0.8, 0.8),
            },
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=9,
            color=ANNO_INK,
            clip_on=True,
            zorder=5,
        )

    return max(entry.y for entry in placed)


def plot_volumetric(
    series: dict[str, list[Cell]],
    *,
    title: str,
    device_qubits: int | None = None,
    depth_ceiling: int | None = None,
    colorbar_label: str = "normalized fidelity vs exact",
    depth_label: str = "Compiled circuit depth",
    show_values: bool | None = None,
    annotate: bool | None = None,
) -> Figure:
    """A volumetric grid for one or more algorithms on one device.

    ``series`` maps a label -- an algorithm name -- to the cells it produced.
    One entry gives the single-algorithm plot, several give the combined one;
    the difference is only in the defaults, which switch the per-cell fidelity
    text off and the per-series arrow labels on once there is more than one
    series to tell apart. Both can be forced either way.

    ``device_qubits`` and ``depth_ceiling`` size the background lattice. Either
    may be None, in which case the lattice takes its extent from the data and
    claims nothing beyond it.

    Cells are drawn widest-first so that a narrower cell landing on the same
    shape stays visible underneath.
    """
    if not series:
        raise ValueError("plot_volumetric needs at least one series")

    if show_values is None:
        show_values = len(series) == 1
    if annotate is None:
        annotate = len(series) > 1

    cmap = create_faded_spectral_cmap()
    cells = [cell for group in series.values() for cell in group]
    if not cells:
        raise ValueError("plot_volumetric needs at least one cell to place")

    # The lattice may extend past the data -- that is its job -- so the axis is
    # sized to whichever reaches further.
    w_data = max(cell.width for cell in cells)
    d_data = max(cell.depth for cell in cells)
    max_width = max(w_data, device_qubits or 0)
    max_depth = max(d_data, depth_ceiling or 0)
    decades = int(depth_index(max_depth)) + 2

    fig, ax = plt.subplots(
        figsize=(0.42 * decades + 3.2, 0.36 * (max_width + 3) + 1.6),
        constrained_layout=True,
    )
    ax.set_xlim(0, decades + 1)

    ticks = list(range(1, decades + 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [format_depth(DEPTH_BASE ** (t - 1)) for t in ticks],
        rotation=45,
        ha="right",
        va="top",
        rotation_mode="anchor",
        fontsize=8,
    )
    ax.set_yticks(range(1, max_width + 1))
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel(depth_label)
    ax.set_ylabel("Register width (qubits)")
    ax.set_title(title)
    for side in ax.spines.values():
        side.set_visible(False)

    _draw_lattice(
        ax,
        max_width=device_qubits or w_data,
        max_depth=depth_ceiling or d_data,
        decades=decades,
    )

    anchors: list[tuple[float, float, str]] = []
    # Every occupied position, kept so labels can be lifted clear of cells
    # belonging to other series as well as their own.
    occupied: list[tuple[float, float]] = []
    for label, group in series.items():
        placed = [cell for cell in group if cell.fidelity is not None]
        # Widest first, so a wider cell never hides a narrower one.
        for cell in sorted(placed, key=lambda c: (-c.width, -c.depth)):
            x = depth_index(cell.depth)
            # Clamped, because a device can land below uniform and the ramp has
            # nowhere to put a negative.
            fidelity = min(max(cell.fidelity or 0.0, 0.0), 1.0)
            # The top of the ramp is reserved: 0.95 rather than 1.0 keeps the
            # darkest end of Spectral out of the data, where it reads as ink.
            color = cmap(fidelity * 0.95)
            ax.add_patch(_cell_patch(x, cell.width, color))
            occupied.append((x, float(cell.width)))
            if show_values:
                ax.text(
                    x,
                    cell.width,
                    f"{fidelity:.2f}".lstrip("0"),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=_ink_for(color),
                    zorder=4,
                )

        # A shape that was attempted and produced no number: marked, not shaded,
        # because "we could not measure this" is not a fidelity of zero.
        for cell in group:
            if cell.fidelity is None:
                ax.plot(
                    depth_index(cell.depth),
                    cell.width,
                    marker="x",
                    markersize=5,
                    markeredgewidth=1.2,
                    color=CELL_EDGE,
                    zorder=3,
                )

        if placed:
            widest = max(placed, key=lambda c: (c.width, c.depth))
            anchors.append((depth_index(widest.depth), float(widest.width), label))

    # Headroom is set last, from what the labels actually needed: reserving it
    # up front would leave a band of empty plot whenever they needed less.
    label_top = _annotate(ax, anchors, occupied) if annotate else 0.0
    ax.set_ylim(0, max(max_width + 1.5, label_top + 1.0))

    bar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0)),
        ax=ax,
        shrink=0.6,
        pad=0.02,
    )
    bar.set_label(colorbar_label)
    bar.outline.set_visible(False)
    return fig
