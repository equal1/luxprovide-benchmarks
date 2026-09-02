#!/usr/bin/env python3
"""Plot hardcoded-circuit results (noisy vs noiseless) from a results directory.

Standalone: stdlib plus matplotlib, nothing else. It reads two JSONL files of
shot counts -- one from a noisy simulation, one from a noiseless one -- pairs
them up by circuit name, and draws each pair the way the LuxProvide algorithm
showcase draws its runs: per outcome, the reference probability as a wide bar
and the measured one as a narrow bar in front of it, so what is left showing of
the wide bar is what the noise cost that outcome.

Two benchmarks live in these files and both are drawn, each the way this
project's tests draw it (``--mode`` picks just one):

**Hardcoded circuits** -- the named algorithm runs -- get one measured-vs-ideal
plot each, after ``tests/test_algorithms.py``, plus a fidelity-vs-width
line chart per device.

**Volumetric grids** come from the same runs read as a width scan, after
``tests/test_width_scan.py``: one grid per algorithm -- GHZ, QFT,
QPE, Grover -- plus one carrying all four at once. Each width is one cell,
placed at its compiled depth, shaded by normalized fidelity on the
faded-spectral ramp, over a lattice of the shapes that were reachable. The
renderer is ported from ``eq1bench/volumetric.py`` (itself following
QC-App-Oriented-Benchmarks, QED-C, Apache-2.0); the metric, the provisional
``score()`` and the depth ceiling are carried over verbatim from the width
scan, so a number here means what a number there means.

The one thing a results file of shot counts does not carry is compiled depth,
which is the x coordinate of every cell and cannot be recovered from counts. It
comes from ``--depths``: either the directory of ``.qasm`` the sweep submitted
(``scripts/dump_width_scan_circuits.py`` writes exactly that, one file per
``<device>-<algorithm>-<width>q``, and `qasm_depth` measures it here without
qiskit) or a JSON/JSONL mapping of circuit name to depth. Nothing is
recomputed and no circuit is rebuilt. A width with no depth has no position on
the grid and is dropped, as it is in the suite, and the script says which.

The ``width-<w>_depth-<d>_<i>`` circuits that make up the bulk of these files
are a different experiment -- a width x depth grid of random circuits, 100 per
shape -- and are counted but not drawn.

Input format, one JSON object per line, in both files::

    {"circuit_name": "Device-1-ghz-3q", "num_qubits": 3, "shots": 1000,
     "counts": {"000": 514, "111": 486}, "backend": "aer_statevector",
     "seed": 42, "transpile_s": 0.01, "simulate_s": 0.005}

``.gz`` inputs are read directly. Counts are normalised to probabilities before
anything is compared or drawn, so the two files need not share a shot budget.

Note on the reference: unlike the showcase this borrows its look from, the
noiseless side here is *sampled*, not computed from the statevector. It carries
shot noise of its own, which puts a ceiling below 1 on the fidelity of even a
noiseless run. The plots say so, and ``--exact-reference`` is not something this
script can offer -- it never sees a circuit, only counts.

Usage::

    python3 plot_results.py --results-dir results/ --out plots/
    python3 plot_results.py --noisy a.jsonl --ideal b.jsonl.gz --out plots/
    python3 plot_results.py --results-dir results/ --list
    python3 plot_results.py --results-dir results/ --out plots/ --only 'ghz'
    python3 plot_results.py --results-dir results/ --out plots/ \
        --mode volumetric --depths data/width-scan-circuits/ --device-qubits 32

Writes ``<out>/hardcoded/<device>/<family>/<circuit>.png``, a fidelity summary
per device, and ``<out>/hardcoded/index.jsonl`` -- one record per circuit, with
the fidelity, the outcome counts and the path of the plot drawn from it; then
``<out>/volumetric/<device>/`` holding one grid per family plus the combined
one, beside ``<out>/volumetric/grids.jsonl`` -- the score card for each grid.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Chosen before pyplot is imported: the script only ever writes files, and the
# default interactive backend would fail outright on a headless machine.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Palette
#
# The two series of a measured-vs-ideal plot, from the validated categorical
# palette (slots 1 and 2): far enough apart to stay distinct under the common
# colour-vision deficiencies, not just to a normal-vision reader. The summary
# chart needs one hue per algorithm family and takes the next slots in order --
# fixed order, never cycled, so a family keeps its colour across every chart
# here even when a chart does not show all four.
# ---------------------------------------------------------------------------

MEASURED_COLOR = "#2a78d6"
IDEAL_COLOR = "#eb6834"

FAMILY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

# The reference is drawn as a tinted pane rather than a solid bar: at full
# strength two saturated hues fight each other for a comparison that is
# supposed to read at a glance. The fill carries the shape, the outline keeps
# the edge crisp where the tint alone would wash out.
IDEAL_FILL_ALPHA = 0.22
IDEAL_EDGE_ALPHA = 0.9
MEASURED_FILL_ALPHA = 0.92

# Surface and inks. The off-white ground is easier to sit next to a report's
# text than paper white, and the two greys keep labels legible without pulling
# attention off the bars.
SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e7e6e1"
AXIS_COLOR = "#d9d8d4"

# The two bars of one outcome share an x position, so it is the widths that
# tell them apart: the reference behind, the measurement in front of it.
IDEAL_BAR_WIDTH = 0.78
MEASURED_BAR_WIDTH = 0.40

DEFAULT_TOP_N = 15

# Volumetric circuits: a width x depth grid of random square circuits, many
# instances per cell. Scored as a sweep and drawn as one grid rather than read
# one plot at a time -- see the volumetric sections below.
VOLUMETRIC_RE = re.compile(
    r"^width-(?P<width>\d+)_depth-(?P<depth>\d+)_(?P<instance>\d+)$"
)

# `<family>-<width>q`, wherever it sits in the circuit name. Everything before
# it and everything after it is device and variant -- see `parse_name`.
NAME_RE = re.compile(r"(?:^|-)(?P<family>[A-Za-z][A-Za-z0-9_]*)-(?P<width>\d+)q(?=$|-)")

# Trailing transpiler-setting tag on the device half of a name, e.g. the
# `-opt2` of `ghz-3q-device-1-opt2`.
VARIANT_RE = re.compile(r"-(opt\d+)$")

# Report labels for the algorithm families. MQT Bench's `qpeexact` and a plain
# `qpe` are the same family as far as a reader is concerned.
FAMILY_LABELS = {
    "ghz": "GHZ",
    "qft": "QFT",
    "qpe": "QPE",
    "qpeexact": "QPE",
    "grover": "Grover",
}

# Display order for families, so a summary chart's legend does not reorder
# itself between runs. Anything unlisted sorts after these, alphabetically.
# Also the order the grids are drawn and the combined grid stacks them.
FAMILY_ORDER = ["GHZ", "QFT", "QPE", "Grover"]

# Report label -> the mqt-bench name the rest of the file speaks, the inverse
# of ALGORITHMS below. QPE is the one that differs: mqt calls it `qpeexact`.
FAMILY_ALGORITHMS = {"GHZ": "ghz", "QFT": "qft", "QPE": "qpeexact", "Grover": "grover"}

Counts = dict[str, float]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield the objects of a JSONL file, transparently un-gzipping a ``.gz``.

    Blank lines are skipped; a line that will not parse fails loudly, since a
    truncated results file is worth knowing about rather than plotting around.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{number}: {error}") from error


def index_runs(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Every run in `path` keyed by circuit name, split by which benchmark it is.

    The two kinds share a file and a record format but nothing else: a
    hardcoded circuit is read one distribution at a time, a volumetric one only
    ever as part of its cell. Splitting them on the way in means neither mode
    has to filter the other out again.
    """
    hardcoded: dict[str, dict] = {}
    volumetric: dict[str, dict] = {}
    for record in read_jsonl(path):
        name = record.get("circuit_name")
        if not name:
            continue
        runs = volumetric if VOLUMETRIC_RE.match(name) else hardcoded
        if name in runs:
            print(f"warning: {path.name}: duplicate {name!r}, keeping the last",
                  file=sys.stderr)
        runs[name] = record
    return hardcoded, volumetric


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    """The noisy and noiseless input paths, from --noisy/--ideal or a directory.

    A results directory is looked at rather than assumed: each side may be
    gzipped or not, and the plain file wins when both are present because it is
    the cheaper read of the same content.
    """
    if args.noisy and args.ideal:
        return Path(args.noisy), Path(args.ideal)
    if args.noisy or args.ideal:
        raise SystemExit("--noisy and --ideal must be given together")

    directory = Path(args.results_dir)
    if not directory.is_dir():
        raise SystemExit(f"not a directory: {directory}")

    def pick(stem: str) -> Path:
        for candidate in (directory / f"{stem}.jsonl", directory / f"{stem}.jsonl.gz"):
            if candidate.is_file():
                return candidate
        raise SystemExit(
            f"no {stem}.jsonl or {stem}.jsonl.gz in {directory} "
            f"-- name the files with --noisy and --ideal instead"
        )

    return pick(args.noisy_stem), pick(args.ideal_stem)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitName:
    """A circuit name split into the axes the report tree is built from."""

    raw: str
    device: str
    family: str
    label: str
    width: int | None
    variant: str | None

    @property
    def title(self) -> str:
        width = f"-{self.width}q" if self.width is not None else ""
        variant = f" ({self.variant})" if self.variant else ""
        return f"{self.device} / {self.label}{width}{variant}"


def parse_name(name: str) -> CircuitName:
    """Split a circuit name into device, family, width and transpiler variant.

    Two shapes appear in practice and both are handled by looking for the
    ``<family>-<width>q`` core and treating whatever surrounds it as the device:
    ``Device-1-ghz-10q`` puts the device first, ``ghz-3q-device-1-opt2``
    puts it last with a transpiler tag on the end. A name that has no such core
    is not an error -- it keeps its whole self as the family and lands under an
    ``unclassified`` device, so an unfamiliar circuit still gets plotted.
    """
    match = NAME_RE.search(name)
    if match is None:
        return CircuitName(name, "unclassified", name, name, None, None)

    family = match.group("family").lower()
    width = int(match.group("width"))
    before = name[: match.start()].strip("-")
    after = name[match.end():].strip("-")

    device = before or after or "unclassified"
    variant = None
    if not before and after:
        # Device sits after the core, and any transpiler tag after that.
        tag = VARIANT_RE.search(after)
        if tag:
            variant = tag.group(1)
            device = after[: tag.start()]

    return CircuitName(
        raw=name,
        device=device,
        family=family,
        label=FAMILY_LABELS.get(family, family.upper()),
        width=width,
        variant=variant,
    )


def family_sort_key(label: str) -> tuple[int, str]:
    if label in FAMILY_ORDER:
        return (FAMILY_ORDER.index(label), "")
    return (len(FAMILY_ORDER), label)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def probabilities(counts: dict[str, float]) -> Counts:
    """`counts` normalised to a distribution.

    The two files are compared as distributions, not as counts: they may have
    been run at different shot budgets, and a fidelity taken over raw counts
    would not be a fidelity at all.
    """
    total = sum(counts.values())
    if not total:
        raise ValueError("empty distribution")
    return {state: value / total for state, value in counts.items()}


def classical_fidelity(ideal: Counts, output: Counts) -> float:
    r"""The fidelity of a measured distribution against a reference one.

    .. math::

        F_s(P_{ideal}, P_{output}) =
            \left( \sum_x \sqrt{P_{output}(x) P_{ideal}(x)} \right)^2

    The squared Bhattacharyya coefficient: 1 when the two distributions agree
    everywhere, 0 when they share no outcome at all. Outcomes missing from
    either side contribute nothing, so only the shared support is summed.

    Both arguments are normalised first: the formula is defined on
    distributions, and a caller holding raw counts would otherwise get a number
    that is not a fidelity.
    """
    ideal_total = sum(ideal.values())
    output_total = sum(output.values())
    if not ideal_total or not output_total:
        raise ValueError("fidelity is undefined against an empty distribution")
    overlap = sum(
        math.sqrt((ideal[x] / ideal_total) * (output[x] / output_total))
        for x in ideal.keys() & output.keys()
    )
    return overlap**2


def total_variation_distance(ideal: Counts, output: Counts) -> float:
    """Half the L1 distance between two distributions, in [0, 1].

    Reported beside the fidelity because the two fail differently: the fidelity
    of a run whose support has drifted off the reference entirely is 0 and says
    nothing more, while the TVD still distinguishes "spread thin over the wrong
    outcomes" from "concentrated on one wrong outcome".
    """
    ideal_p = probabilities(dict(ideal))
    output_p = probabilities(dict(output))
    return 0.5 * sum(
        abs(ideal_p.get(state, 0.0) - output_p.get(state, 0.0))
        for state in ideal_p.keys() | output_p.keys()
    )


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


@dataclass
class Pair:
    """One circuit's noisy run alongside its noiseless reference."""

    name: CircuitName
    measured: Counts
    ideal: Counts
    noisy_record: dict
    ideal_record: dict
    fidelity: float = 0.0
    tvd: float = 0.0
    plot_path: Path | None = None

    def __post_init__(self) -> None:
        self.fidelity = classical_fidelity(self.ideal, self.measured)
        self.tvd = total_variation_distance(self.ideal, self.measured)


def build_pairs(
    noisy: dict[str, dict], ideal: dict[str, dict]
) -> tuple[list[Pair], list[str]]:
    """Pair the two files up by circuit name, in report order.

    Returns the pairs and the names that could not be paired. A noisy run with
    no reference cannot be scored or drawn against anything, so it is reported
    rather than plotted alone.

    Keys are compared as they arrive. Both files come out of the same
    generator, so a circuit that measures fewer clbits than it has qubits
    (``qpeexact`` leaves its eigenstate ancilla unmeasured) is keyed the same
    way on both sides, and a circuit carrying a second unmeasured register
    (the QFT sources) carries it on both sides too. A key-length disagreement
    between the two files would mean they did not come from the same circuits,
    so it fails loudly instead of being silently re-keyed.
    """
    pairs: list[Pair] = []
    unmatched: list[str] = []

    for name, noisy_record in noisy.items():
        if name not in ideal:
            unmatched.append(name)
            continue
        ideal_record = ideal[name]
        measured = probabilities(noisy_record["counts"])
        reference = probabilities(ideal_record["counts"])

        measured_bits = len(next(iter(measured)))
        reference_bits = len(next(iter(reference)))
        if measured_bits != reference_bits:
            raise SystemExit(
                f"{name}: keys are {measured_bits} bits in the noisy file and "
                f"{reference_bits} in the reference -- the two files do not "
                f"describe the same circuit"
            )

        pairs.append(Pair(parse_name(name), measured, reference,
                          noisy_record, ideal_record))

    pairs.sort(
        key=lambda p: (
            p.name.device,
            family_sort_key(p.name.label),
            p.name.width if p.name.width is not None else -1,
            p.name.raw,
        )
    )
    return pairs, sorted(unmatched)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def style_axes(ax) -> None:
    """The recessive frame both chart forms share.

    One set of horizontal rules behind the marks, a baseline under them, and
    nothing else -- the data carries the comparison, the frame should not
    compete with it.
    """
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(length=0, colors=TEXT_SECONDARY, labelsize=8.5)


def plot_measured_vs_ideal(
    measured: Counts,
    ideal: Counts,
    name: str,
    top_n: int = DEFAULT_TOP_N,
    note: str | None = None,
) -> Figure:
    """Bar-plot a measured distribution over its reference, superimposed.

    One x position per outcome, both bars centred on it: the reference
    probability as a wide bar behind, the measured one as a narrow bar in front.
    What is left showing of the wide bar *is* the error on that outcome -- a
    reference shoulder above the measured bar is probability the noise took
    away, and a measured bar standing above the reference one is probability it
    added. Reading that off a pair of adjacent bars means comparing two heights
    by eye; stacking them on one baseline turns it into one length.

    Which outcomes to show is the other half of it. Taking the top of the
    measured distribution alone is wrong here: a noisy enough run pushes the
    correct answers out of its top `top_n` entirely, and the plot would then
    show a gap of exactly zero because every reference bar is off-screen. So the
    candidates are the top of *both*, ranked by whichever distribution weights
    them more heavily, and the ones drawn are ordered by reference probability
    -- the outcomes the circuit is supposed to produce first, in a fixed order,
    however the noise fell.
    """

    def weight(distribution: Counts, state: str) -> float:
        return distribution.get(state, 0.0)

    candidates = set(sorted(ideal, key=lambda s: ideal[s], reverse=True)[:top_n])
    candidates |= set(sorted(measured, key=lambda s: measured[s], reverse=True)[:top_n])
    states = sorted(
        candidates,
        key=lambda s: max(weight(ideal, s), weight(measured, s)),
        reverse=True,
    )[:top_n]
    # Display order, decided separately from which states made the cut.
    states.sort(key=lambda s: (weight(ideal, s), weight(measured, s)), reverse=True)

    positions = list(range(len(states)))

    fig, ax = plt.subplots(figsize=(max(6.4, 0.42 * len(states) + 2.4), 4.4))
    fig.patch.set_facecolor(SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)

    ax.bar(
        positions,
        [weight(ideal, s) for s in states],
        width=IDEAL_BAR_WIDTH,
        facecolor=to_rgba(IDEAL_COLOR, IDEAL_FILL_ALPHA),
        edgecolor=to_rgba(IDEAL_COLOR, IDEAL_EDGE_ALPHA),
        linewidth=1.0,
        label="ideal (noiseless)",
    )
    ax.bar(
        positions,
        [weight(measured, s) for s in states],
        width=MEASURED_BAR_WIDTH,
        facecolor=to_rgba(MEASURED_COLOR, MEASURED_FILL_ALPHA),
        linewidth=0,
        label="measured (noisy)",
    )

    ax.set_xticks(positions)
    # Monospaced: bitstrings are read column by column, and a proportional face
    # puts the same bit at a different offset on every label.
    ax.set_xticklabels(states, rotation=90, family="monospace", fontsize=8.5)
    ax.set_xlim(-0.7, len(states) - 0.3)
    ax.set_ylabel("probability", fontsize=9, color=TEXT_SECONDARY, labelpad=8)
    ax.set_xlabel("measured outcome", fontsize=9, color=TEXT_SECONDARY, labelpad=6)

    # A header row rather than a centred title block: the name on the left, the
    # legend on the right, both clear of the bars.
    ax.set_title(name, loc="left", fontsize=11, color=TEXT_PRIMARY, pad=26)
    if note is not None:
        ax.text(0.0, 1.035, note, transform=ax.transAxes, fontsize=9,
                color=TEXT_SECONDARY)
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=2,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        handlelength=1.1,
        handleheight=1.1,
        borderpad=0.0,
        columnspacing=1.4,
    )

    style_axes(ax)
    fig.tight_layout()
    return fig


# Vertical clearance between two end-of-line labels, in points. Below this the
# two read as one smudge, which is worse than no direct label at all.
LABEL_GAP_PT = 11.0


def label_line_ends(ax, ends: list[tuple[float, float, str]]) -> None:
    """Write each series' name at the right end of its line, nudged apart.

    Lines that finish at the same place -- and on a fidelity sweep several of
    them finish flat on zero -- would otherwise stack their labels into an
    unreadable overprint. So the labels are laid out in display space, where
    "too close" is a fixed number of points rather than a data distance that
    means something different on every chart: within a column, each label is
    pushed up until it clears the one below it.

    Called after the axes limits are set, since the data-to-display mapping is
    what the spacing is computed in.
    """
    if not ends:
        return

    columns: dict[int, list[tuple[float, float, str]]] = {}
    for x, y, text in ends:
        # Rounded: two lines ending one pixel apart are the same column as far
        # as an overlapping label is concerned.
        column = round(ax.transData.transform((x, y))[0] / 4.0)
        columns.setdefault(column, []).append((x, y, text))

    gap_px = LABEL_GAP_PT * ax.figure.dpi / 72.0
    for group in columns.values():
        placed = float("-inf")
        for x, y, text in sorted(group, key=lambda end: end[1]):
            display_y = ax.transData.transform((x, y))[1]
            target = max(display_y, placed + gap_px)
            placed = target
            ax.annotate(
                text,
                xy=(x, y),
                # Back to points, which is what offset_points wants.
                xytext=(6, (target - display_y) * 72.0 / ax.figure.dpi),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=TEXT_SECONDARY,
            )


def plot_fidelity_by_width(device: str, pairs: list[Pair]) -> Figure | None:
    """Fidelity against register width, one line per algorithm family.

    The per-circuit plots say what one run did; this says how the noise scales,
    which is the only thing a sweep of widths is for. Drawn only where there is
    a sweep to draw -- a device whose families are all single points gets None
    and no chart, since a line chart of one point per series is a table.

    Each line is labelled at its right end as well as in the legend, so a reader
    who cannot separate two hues still has the identity in text.
    """
    by_family: dict[str, list[Pair]] = {}
    for pair in pairs:
        if pair.name.width is None:
            continue
        by_family.setdefault(pair.name.label, []).append(pair)

    if not by_family or max(len(group) for group in by_family.values()) < 2:
        return None

    families = sorted(by_family, key=family_sort_key)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fig.patch.set_facecolor(SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)

    ends: list[tuple[float, float, str]] = []
    for index, family in enumerate(families):
        group = sorted(by_family[family], key=lambda p: p.name.width or 0)
        widths = [pair.name.width for pair in group]
        fidelities = [pair.fidelity for pair in group]
        color = FAMILY_COLORS[index % len(FAMILY_COLORS)]
        ax.plot(widths, fidelities, marker="o", markersize=5, linewidth=2,
                color=color, label=family,
                # A surface-coloured ring keeps two markers that land on the
                # same point from merging into one blob.
                markeredgecolor=SURFACE_COLOR, markeredgewidth=1.4)
        ends.append((float(widths[-1]), fidelities[-1], family))

    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("register width (qubits)", fontsize=9, color=TEXT_SECONDARY,
                  labelpad=6)
    ax.set_ylabel("classical fidelity $F_s$", fontsize=9, color=TEXT_SECONDARY,
                  labelpad=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Room on the right for the end-of-line labels.
    all_widths = [p.name.width for p in pairs if p.name.width is not None]
    ax.set_xlim(min(all_widths) - 0.4, max(all_widths) + 1.4)

    ax.set_title(f"{device} -- fidelity against width", loc="left", fontsize=11,
                 color=TEXT_PRIMARY, pad=26)
    ax.text(0.0, 1.035,
            "$F_s$ against a sampled noiseless run of the same circuit",
            transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY)
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=len(families),
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        handlelength=1.4,
        borderpad=0.0,
        columnspacing=1.4,
    )

    style_axes(ax)
    # Limits first, labels second: where a label can go is decided in display
    # space, and display space is not settled until the axes are.
    label_line_ends(ax, ends)
    fig.tight_layout()
    return fig


def save(fig: Figure, base: Path, formats: list[str], dpi: int) -> Path:
    """Write `fig` once per requested format; return the first path written."""
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in formats:
        # Appended, not `with_suffix`: a task id puts a dot in the middle
        # of the name ("B-1.5-ghz-..."), which `with_suffix` would read as an
        # extension and cut everything after it.
        path = base.with_name(f"{base.name}.{suffix}")
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        written.append(path)
    plt.close(fig)
    return written[0]


# ---------------------------------------------------------------------------
# Volumetric rendering
#
# Ported from this project's `eq1bench/volumetric.py`, which itself follows
# QC-App-Oriented-Benchmarks (QED-C, Apache-2.0) for the faded-spectral
# colormap, the logarithmic depth index and the label-spreading annotation
# pass. Kept as close to the original as a standalone file allows, so a grid
# drawn here and a grid drawn by `test_width_scan.py` are the same picture.
# ---------------------------------------------------------------------------

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
    """A depth as an axis label, abbreviated past a thousand. Ported from QED-C."""
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

    Clearing the anchor is not enough on a combined grid, because a *different*
    series may occupy that column higher up -- so a label rises above every cell
    near its column, not just its own. Labels that still collide are then walked
    upward one row at a time.
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
    """A volumetric grid for one or more series on one device.

    ``series`` maps a label to the cells it produced. One entry gives the
    single-series plot, several give the combined one; the difference is only in
    the defaults, which switch the per-cell fidelity text off and the per-series
    arrow labels on once there is more than one series to tell apart. Both can
    be forced either way.

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


# ---------------------------------------------------------------------------
# Width scan
#
# The volumetric benchmark as `tests/test_width_scan.py` runs it:
# each algorithm swept across register widths, one cell per width, scored by
# normalized fidelity and placed on the grid at its compiled depth. The metric,
# the provisional score and the constants below are carried over from that file
# so a number drawn here means what a number drawn there means.
# ---------------------------------------------------------------------------

# mqt-bench name -> (report label, task id). Verbatim from test_width_scan.
ALGORITHMS = {
    "ghz": ("GHZ", "B-1.5"),
    "qft": ("QFT", "B-2.4"),
    "qpeexact": ("QPE", "B-3.4"),
    "grover": ("Grover", "B-4.4"),
}

# Grover's compiled depth triples every 2 qubits, so the sweep enforces a
# ceiling. It is passed to the grid, where it bounds the background lattice --
# the boundary is drawn rather than implied by where the data runs out.
MAX_COMPILED_DEPTH = 200_000

# What the depths handed to the grids are: set once from --depth-kind, and
# recorded beside every score so a grid drawn against logical depth is never
# mistaken later for one drawn against compiled depth.
DEPTH_KIND = "compiled"

# Provisional, and only meaningful through score(). 2/3 is the Quantum Volume
# convention, borrowed by test_width_scan.py for want of a decided one.
SCORE_THRESHOLD = 2 / 3
SCORE_DEFINITION = f"widest width with fidelity >= {SCORE_THRESHOLD:.2f} (PROVISIONAL)"


def normalized_fidelity(ideal: Counts, output: Counts, num_qubits: int) -> float:
    """Hellinger fidelity rescaled so that a dead, uniform device scores 0.

    Raw Hellinger fidelity has a floor: guessing uniformly already scores
    ``Fs_uniform``, which for a spread-out ideal is far from zero, so the useful
    range is squeezed into the top of the scale. Subtracting that floor and
    dividing by what is left puts "no information at all" at 0 and the exact
    distribution at 1. Clamped below, because a device can land under uniform.

    `classical_fidelity` stands in for qiskit's ``hellinger_fidelity`` here:
    the two are the same quantity, by way of :math:`H^2 = 1 - \\sum_x \\sqrt{pq}`.
    Uniform is over the full ``2**num_qubits`` space, and only its shared
    support with `ideal` contributes, so it is summed rather than built.
    """
    fidelity = classical_fidelity(ideal, output)

    outcomes = 2**num_qubits
    total = sum(ideal.values())
    uniform = sum(math.sqrt((p / total) / outcomes) for p in ideal.values()) ** 2

    # A uniform ideal leaves nothing to rescale against -- undefined, not 1.
    if uniform >= 1.0:
        return 0.0

    return max((fidelity - uniform) / (1 - uniform), 0.0)


def score(results: dict[int, tuple[int, float]]) -> float:
    """PLACEHOLDER -- the formula has not been decided yet.

    Carried over verbatim from ``test_width_scan.py``, deliberately: a number
    plotted here should mean what a number plotted there means, including while
    the definition is provisional.

    Provisional: the widest register whose run cleared SCORE_THRESHOLD, or 0.0
    if none did. It is the volumetric framework's own question -- how wide did
    the device get before the answer stopped being the answer -- reduced to one
    number, and it degrades sanely when a sweep is truncated.

    It is also crude: it throws away the fidelities themselves, and it does not
    correct for shot noise.

    Every call site reads through this function, so swapping the body is the
    whole change.
    """
    cleared = [w for w, (_, fidelity) in results.items() if fidelity >= SCORE_THRESHOLD]
    return float(max(cleared)) if cleared else 0.0


# --- compiled depth ---------------------------------------------------------
#
# The x coordinate of every cell, and the one thing a results file of shot
# counts does not carry. It comes from the circuits themselves -- the QASM the
# sweep submitted -- or from a mapping someone else already extracted.

# A `gate name(params) qubits { body }` definition. Removed whole: the body is
# the gate's own decomposition, not instructions in the circuit.
QASM_GATE_DEF_RE = re.compile(r"\bgate\b[^{]*\{[^}]*\}", re.DOTALL)

# `qreg q[16]` / `creg c[16]`.
QASM_REG_RE = re.compile(r"^(qreg|creg)\s+(\w+)\s*\[\s*(\d+)\s*\]$")

# An operation: a name, optional parenthesised parameters, then its operands.
# Parameters are matched as a unit so the commas inside them are not read as
# operand separators.
QASM_OP_RE = re.compile(r"^(\w+)\s*(\([^)]*\))?\s*(.*)$", re.DOTALL)

# One operand: a whole register, or one bit of it.
QASM_BIT_RE = re.compile(r"^(\w+)\s*(?:\[\s*(\d+)\s*\])?$")

# Instructions that occupy no layer. Qiskit's `QuantumCircuit.depth()` filters
# directives and delays out the same way, so a depth taken here matches one
# taken there.
QASM_TRANSPARENT = {"barrier", "delay"}


def qasm_depth(text: str) -> int:
    """The depth of a flat OpenQASM 2 circuit: its longest chain of operations.

    The same number ``QuantumCircuit.depth()`` returns, computed without qiskit.
    Every operation lands one layer past the deepest qubit it touches, which is
    the definition -- so this needs no gate library, only which qubits each
    instruction names.

    `measure` is counted, on the qubit it reads; barriers and delays are not.
    An operand naming a whole register broadcasts, which qiskit expands into one
    independent instruction per bit rather than one instruction across them all,
    so each bit advances on its own here too.
    """
    body = QASM_GATE_DEF_RE.sub("", text)
    sizes: dict[str, int] = {}
    layers: dict[tuple[str, int], int] = {}

    for raw in body.split(";"):
        # Comments run to end of line, so they are stripped per line rather
        # than per statement.
        statement = " ".join(
            line.split("//")[0].strip() for line in raw.splitlines()
        ).strip()
        if not statement or statement.startswith(("OPENQASM", "include")):
            continue

        register = QASM_REG_RE.match(statement)
        if register is not None:
            sizes[register.group(2)] = int(register.group(3))
            continue

        operation = QASM_OP_RE.match(statement)
        if operation is None:
            raise ValueError(f"cannot parse QASM statement: {statement!r}")
        name, _params, operands = operation.groups()
        if name in QASM_TRANSPARENT:
            continue
        if name == "if":
            raise ValueError("conditional QASM is not supported")
        # `measure q[0] -> c[0]`: the classical target is not a qubit.
        operands = operands.split("->")[0]

        qubits: list[tuple[str, int]] = []
        broadcast: list[list[tuple[str, int]]] = []
        for operand in operands.split(","):
            bit = QASM_BIT_RE.match(operand.strip())
            if bit is None:
                raise ValueError(f"cannot parse QASM operand: {operand!r}")
            register_name, index = bit.group(1), bit.group(2)
            if register_name not in sizes:
                raise ValueError(f"unknown QASM register {register_name!r}")
            if index is None:
                broadcast.append(
                    [(register_name, i) for i in range(sizes[register_name])]
                )
            else:
                qubits.append((register_name, int(index)))

        # A whole-register operand stands for one instruction per bit; an
        # explicit bit is shared by all of them.
        groups = (
            [qubits + list(bits) for bits in zip(*broadcast)] if broadcast else [qubits]
        )
        for group in groups:
            if not group:
                continue
            level = max(layers.get(qubit, 0) for qubit in group) + 1
            for qubit in group:
                layers[qubit] = level

    return max(layers.values(), default=0)


def load_depths(source: Path) -> dict[str, int]:
    """Circuit name -> compiled depth, from whatever the caller has.

    A directory is read as the output of ``scripts/dump_width_scan_circuits.py``
    -- one ``<name>.qasm`` per circuit, holding exactly what the sweep submits
    -- and each file's depth is measured. A file is read as JSON or JSONL:
    either ``{name: depth}``, ``{name: {"compiled_depth": ...}}``, or the run
    database the suite writes, whose records carry ``compiled_circuit_depth``.

    Depth is not in the results files themselves, and it cannot be recovered
    from shot counts, so it has to come from one of these.
    """
    if source.is_dir():
        depths = {}
        for path in sorted(source.glob("*.qasm")):
            depths[path.stem] = qasm_depth(path.read_text(encoding="utf-8"))
        if not depths:
            raise SystemExit(f"no .qasm files in {source}")
        return depths

    if not source.is_file():
        raise SystemExit(f"no such depth source: {source}")

    text = source.read_text(encoding="utf-8").strip()
    if source.suffix == ".jsonl" or text.startswith("{\"kind\""):
        depths = {}
        for record in read_jsonl(source):
            data = record.get("raw") or record.get("data") or record
            name = data.get("circuit_name") or data.get("name")
            depth = data.get("compiled_circuit_depth") or data.get("compiled_depth")
            if name and depth:
                depths[name] = int(depth)
        return depths

    loaded = json.loads(text)
    return {
        name: int(value if isinstance(value, int | float) else value["compiled_depth"])
        for name, value in loaded.items()
    }


# --- sweeps -----------------------------------------------------------------


@dataclass
class Outcome:
    """What one width produced. `fidelity is None` means it was not measured."""

    logical_depth: int | None = None
    compiled_depth: int | None = None
    fidelity: float | None = None
    reason: str = ""  # why not, when fidelity is None


def sweep_cells(outcomes: dict[int, Outcome]) -> list[Cell]:
    """One sweep's outcomes as plottable cells, narrowest first.

    Compiled depth is the x coordinate, so a width with no compiled depth has
    no position on the grid and is dropped -- exactly as in the suite, where a
    width that never got as far as being compiled is reported on its own card
    instead.
    """
    return [
        Cell(width=width, depth=outcome.compiled_depth, fidelity=outcome.fidelity)
        for width, outcome in sorted(outcomes.items())
        if outcome.compiled_depth is not None
    ]


def build_sweeps(
    pairs: list[Pair], depths: dict[str, int]
) -> dict[str, dict[str, dict[int, Outcome]]]:
    """The paired runs as device -> family -> width -> Outcome.

    One cell per (device, family, width), which is the shape a width scan
    produces and the shape the grids read. A run whose compiled depth is not in
    `depths` still gets an Outcome, carrying the fidelity and the reason it
    cannot be placed, so the caller can say which widths the grid is missing
    rather than silently drawing a shorter sweep.
    """
    sweeps: dict[str, dict[str, dict[int, Outcome]]] = {}
    for pair in pairs:
        if pair.name.width is None or pair.name.family not in ALGORITHMS:
            continue
        label = ALGORITHMS[pair.name.family][0]
        depth = depths.get(pair.name.raw)
        sweeps.setdefault(pair.name.device, {}).setdefault(label, {})[
            pair.name.width
        ] = Outcome(
            compiled_depth=depth,
            fidelity=normalized_fidelity(
                pair.ideal, pair.measured, pair.noisy_record.get("num_qubits") or pair.name.width
            ),
            reason="" if depth is not None else "no depth for this circuit",
        )
    return sweeps


def grid_data(device: str, family: str, outcomes: dict[int, Outcome]) -> dict:
    """The score card beside one family's grid -- test_width_grid's payload."""
    measured = {
        width: (outcome.compiled_depth, outcome.fidelity)
        for width, outcome in outcomes.items()
        if outcome.fidelity is not None
    }
    widest = max(measured)
    # Scored and plotted are not the same set: a width is scored on its
    # fidelity alone, but it reaches the grid only if it also has a depth to
    # sit at. Both counts are reported, so a score computed over more widths
    # than the picture shows says so instead of being read off the cells.
    plotted = [w for w, o in outcomes.items() if o.compiled_depth is not None
               and o.fidelity is not None]
    return {
        "task": ALGORITHMS[FAMILY_ALGORITHMS[family]][1],
        "algorithm": family,
        "device": device,
        "score": score(measured),
        "score_definition": SCORE_DEFINITION,
        "widths_measured": len(measured),
        "widths_plotted": len(plotted),
        "max_width_measured": widest,
        "best_fidelity": round(max(f for _, f in measured.values()), 4),
        "fidelity_at_max_width": round(measured[widest][1], 4),
        "depth_kind": DEPTH_KIND,
        "widths_without_depth": sorted(
            width for width, outcome in outcomes.items()
            if outcome.compiled_depth is None
        ),
        "fidelity_definition": (
            "Hellinger fidelity rescaled so uniform scores 0 "
            "(see normalized_fidelity)"
        ),
    }


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def slug(text: str) -> str:
    """`text` as a path component: no separators, no surprises on any OS."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "unnamed"


def record(pair: Pair, out: Path, max_entries: int) -> dict:
    """The index record for one circuit.

    The distributions go in the index rather than onto the chart: they are the
    entire result, and the one thing a plot cannot be redrawn without, but as a
    table of up to `shots` rows they would bury everything else. Truncated to
    the `max_entries` most probable outcomes, with a note saying whether that
    truncation bit.
    """

    def top(distribution: Counts) -> tuple[Counts, dict]:
        ordered = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
        return (
            dict(ordered[:max_entries]),
            {"entries": len(ordered), "truncated": len(ordered) > max_entries},
        )

    measured_top, measured_meta = top(pair.measured)
    ideal_top, ideal_meta = top(pair.ideal)
    noisy, reference = pair.noisy_record, pair.ideal_record

    return {
        "circuit_name": pair.name.raw,
        "device": pair.name.device,
        "family": pair.name.label,
        "algorithm": pair.name.family,
        "width": pair.name.width,
        "variant": pair.name.variant,
        "num_qubits": noisy.get("num_qubits"),
        "shots": noisy.get("shots"),
        "ideal_shots": reference.get("shots"),
        "backend": noisy.get("backend"),
        "ideal_backend": reference.get("backend"),
        "seed": noisy.get("seed"),
        "fidelity": round(pair.fidelity, 6),
        "fidelity_definition": (
            "F_s = (sum_x sqrt(P_output(x) P_ideal(x)))^2 against a sampled "
            "noiseless run of the same circuit"
        ),
        "total_variation_distance": round(pair.tvd, 6),
        "transpile_s": noisy.get("transpile_s"),
        "simulate_s": noisy.get("simulate_s"),
        "plot": (
            str(pair.plot_path.relative_to(out)) if pair.plot_path else None
        ),
        "measured": measured_top,
        "measured_meta": measured_meta,
        "ideal": ideal_top,
        "ideal_meta": ideal_meta,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Both benchmarks are drawn by default; --mode picks just one.",
    )
    parser.add_argument("--mode", default="both",
                        choices=["hardcoded", "volumetric", "both"],
                        help="which benchmark to draw (default: both)")
    parser.add_argument("--results-dir", default="results",
                        help="directory holding the two JSONL files (default: results)")
    parser.add_argument("--noisy", help="path to the noisy results JSONL(.gz)")
    parser.add_argument("--ideal", help="path to the noiseless results JSONL(.gz)")
    parser.add_argument("--noisy-stem", default="simulation_results",
                        help="filename stem of the noisy file inside --results-dir")
    parser.add_argument("--ideal-stem", default="ideal_results",
                        help="filename stem of the noiseless file inside --results-dir")
    parser.add_argument("--out", default="plots",
                        help="output directory for the plots (default: plots)")
    parser.add_argument("--only", metavar="REGEX",
                        help="hardcoded mode: plot only circuits matching this regex")
    parser.add_argument("--depths", metavar="PATH",
                        help="volumetric mode: where compiled depths come from -- a "
                             "directory of the .qasm the sweep submitted, or a "
                             "JSON/JSONL mapping. Required: a results file of shot "
                             "counts carries no depth, and a cell has no x "
                             "coordinate without one")
    parser.add_argument("--device-qubits", type=int,
                        help="volumetric mode: qubit count to size the background "
                             "lattice to; without it the lattice stops at the data")
    parser.add_argument("--depth-kind", default="compiled",
                        choices=["compiled", "logical"],
                        help="volumetric mode: what the depths in --depths are, "
                             "which sets the axis label and the lattice bound. "
                             "The suite plots compiled depth; logical depth is the "
                             "circuit before compilation, and understates any "
                             "algorithm built from opaque blocks -- mqt-bench's "
                             "Grover is 4 applications of one `Q` gate, logical "
                             "depth 6 at 6 qubits against 2192 compiled layers "
                             "(default: compiled)")
    parser.add_argument("--depth-ceiling", type=int,
                        help="volumetric mode: depth ceiling bounding the "
                             "background lattice (default: the sweep's "
                             f"{MAX_COMPILED_DEPTH:,} for compiled depth, and the "
                             "data's own extent for logical)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"outcomes per plot (default: {DEFAULT_TOP_N})")
    parser.add_argument("--format", default="png", choices=["png", "svg", "pdf", "both"],
                        help="output format; 'both' means png and svg (default: png)")
    parser.add_argument("--dpi", type=int, default=160, help="raster DPI (default: 160)")
    parser.add_argument("--max-entries", type=int, default=4096,
                        help="most distribution entries kept per index record")
    parser.add_argument("--no-summary", action="store_true",
                        help="skip the per-device fidelity-against-width charts")
    parser.add_argument("--no-index", action="store_true",
                        help="skip writing index.jsonl")
    parser.add_argument("--list", action="store_true",
                        help="list what would be plotted and exit, drawing nothing")
    return parser.parse_args(argv)


def run_hardcoded(args, pairs: list[Pair], out: Path, formats: list[str]) -> None:
    """Draw the algorithm showcase: one measured-vs-ideal plot per circuit."""
    for pair in pairs:
        fig = plot_measured_vs_ideal(
            pair.measured,
            pair.ideal,
            name=pair.name.title,
            top_n=args.top_n,
            note=f"$F_s$ = {pair.fidelity:.3f}   TVD = {pair.tvd:.3f}   "
                 f"{pair.noisy_record.get('shots', '?')} shots",
        )
        base = (out / "hardcoded" / slug(pair.name.device) / slug(pair.name.label)
                / slug(pair.name.raw))
        pair.plot_path = save(fig, base, formats, args.dpi)

    print(f"wrote {len(pairs)} circuit plot(s) under {out}/hardcoded/")

    if not args.no_summary:
        devices: dict[str, list[Pair]] = {}
        for pair in pairs:
            devices.setdefault(pair.name.device, []).append(pair)
        for device, group in sorted(devices.items()):
            fig = plot_fidelity_by_width(device, group)
            if fig is None:
                continue
            path = save(fig, out / "hardcoded" / f"fidelity-vs-width-{slug(device)}",
                        formats, args.dpi)
            print(f"wrote summary {path}")

    if not args.no_index:
        index = out / "hardcoded" / "index.jsonl"
        index.parent.mkdir(parents=True, exist_ok=True)
        with index.open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(record(pair, out, args.max_entries)) + "\n")
        print(f"wrote index {index}")

    worst = min(pairs, key=lambda p: p.fidelity)
    best = max(pairs, key=lambda p: p.fidelity)
    mean = sum(p.fidelity for p in pairs) / len(pairs)
    print(f"F_s: mean {mean:.4f}, best {best.fidelity:.4f} ({best.name.raw}), "
          f"worst {worst.fidelity:.4f} ({worst.name.raw})")


def run_width_scan(args, pairs: list[Pair], out: Path, formats: list[str]) -> None:
    """Draw the volumetric grids: one per algorithm, plus the combined one.

    The same two pictures ``test_width_scan.py`` reports, and the same split of
    labour between them. Each family's grid answers "how far did this algorithm
    get" and carries the per-cell fidelity text; the combined grid answers the
    one they cannot, which is how the four compare, and trades that text for
    arrow labels naming each series -- `plot_volumetric`'s own defaults, left
    to fire on their own rather than forced here.

    Reading right is reading cost: the depth axis is logarithmic, so Grover
    sitting several columns right of GHZ at the same width means orders of
    magnitude more layers for the same register.
    """
    global DEPTH_KIND
    DEPTH_KIND = args.depth_kind

    depths = load_depths(Path(args.depths)) if args.depths else {}
    # The compiled-depth ceiling is a fact about the sweep, not about depth in
    # general: it is the number of layers Grover blew past. It has nothing to
    # say about a logical-depth axis, where the lattice takes its extent from
    # the data instead.
    ceiling = args.depth_ceiling
    if ceiling is None and args.depth_kind == "compiled":
        ceiling = MAX_COMPILED_DEPTH
    depth_label = f"{args.depth_kind.capitalize()} circuit depth"
    if not depths:
        print(
            "warning: no compiled depths given, so no cell has an x coordinate "
            "and no grid can be drawn -- pass --depths with the QASM the sweep "
            "submitted (scripts/dump_width_scan_circuits.py writes it) or a "
            "name-to-depth mapping",
            file=sys.stderr,
        )
        return

    sweeps = build_sweeps(pairs, depths)
    if not sweeps:
        print("warning: no run matched a width-scan algorithm", file=sys.stderr)
        return

    records: list[dict] = []
    for device, families in sorted(sweeps.items()):
        directory = out / "volumetric" / slug(device)

        # Definition order matters in the suite because the grids read what the
        # runs left behind; here it is only display order, but the families are
        # still walked in FAMILY_ORDER so the pages read the same way.
        ordered = sorted(families, key=family_sort_key)
        for family in ordered:
            outcomes = families[family]
            cells = sweep_cells(outcomes)
            if not cells:
                print(f"warning: {device}/{family}: no width has a depth in "
                      f"--depths, so no cell can be placed", file=sys.stderr)
                continue
            missing = sorted(w for w, o in outcomes.items() if o.compiled_depth is None)
            if missing:
                print(f"warning: {device}/{family}: no depth for width(s) "
                      f"{missing}, dropped from the grid", file=sys.stderr)

            task = ALGORITHMS[FAMILY_ALGORITHMS[family]][1]
            fig = plot_volumetric(
                {family: cells},
                title=f"{task} — {family} width scan on {device}",
                device_qubits=args.device_qubits,
                depth_ceiling=ceiling,
                depth_label=depth_label,
            )
            path = save(
                fig,
                directory / f"{task}-{FAMILY_ALGORITHMS[family]}-{slug(device)}-width-scan",
                formats,
                args.dpi,
            )
            data = grid_data(device, family, outcomes)
            records.append(data)
            print(f"wrote {path}  (score {data['score']:g}, "
                  f"{data['widths_plotted']} of {data['widths_measured']} "
                  f"widths on the grid)")

        # The combined grid: every family that placed a cell, on one canvas.
        series = {
            family: sweep_cells(families[family])
            for family in ordered
            if sweep_cells(families[family])
        }
        if not series:
            continue
        fig = plot_volumetric(
            series,
            title=f"Width scan — all algorithms on {device}",
            device_qubits=args.device_qubits,
            depth_ceiling=ceiling,
            depth_label=depth_label,
        )
        path = save(fig, directory / f"{slug(device)}-all-algorithms-width-scan",
                    formats, args.dpi)
        print(f"wrote {path}  ({len(series)} algorithms)")

        combined = {
            "device": device,
            "score_definition": SCORE_DEFINITION,
            "algorithms_compared": len(series),
        }
        for family in ordered:
            measured = {
                width: (outcome.compiled_depth, outcome.fidelity)
                for width, outcome in families[family].items()
                if outcome.fidelity is not None
            }
            combined[f"score_{family}"] = score(measured) if measured else "not measured"
            combined[f"max_width_{family}"] = (
                max(measured) if measured else "not measured"
            )
        records.append(combined)

    if not args.no_index and records:
        index = out / "volumetric" / "grids.jsonl"
        index.parent.mkdir(parents=True, exist_ok=True)
        with index.open("w", encoding="utf-8") as handle:
            for data in records:
                handle.write(json.dumps(data) + "\n")
        print(f"wrote index {index}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    noisy_path, ideal_path = resolve_inputs(args)

    noisy_hardcoded, noisy_volumetric = index_runs(noisy_path)
    ideal_hardcoded, ideal_volumetric = index_runs(ideal_path)

    print(f"noisy:     {noisy_path}  ({len(noisy_hardcoded)} hardcoded, "
          f"{len(noisy_volumetric)} volumetric)")
    print(f"noiseless: {ideal_path}  ({len(ideal_hardcoded)} hardcoded, "
          f"{len(ideal_volumetric)} volumetric)")

    wants_hardcoded = args.mode in ("hardcoded", "both")
    wants_volumetric = args.mode in ("volumetric", "both")

    # Both modes read the same paired runs: the showcase draws one plot per
    # circuit, the width scan reduces each to one cell.
    pairs, unmatched = build_pairs(noisy_hardcoded, ideal_hardcoded)
    if args.only:
        pattern = re.compile(args.only)
        pairs = [pair for pair in pairs if pattern.search(pair.name.raw)]
    if unmatched:
        print(f"warning: {len(unmatched)} run(s) with no reference: "
              f"{', '.join(unmatched[:6])}"
              f"{' ...' if len(unmatched) > 6 else ''}", file=sys.stderr)

    if not pairs:
        print("nothing to plot", file=sys.stderr)
        return 1

    if args.list:
        for pair in pairs:
            print(f"  {pair.name.raw:<34} {pair.name.title:<34} "
                  f"F_s={pair.fidelity:.4f}  TVD={pair.tvd:.4f}")
        print(f"\n{len(pairs)} circuit(s) would be plotted")
        return 0

    formats = ["png", "svg"] if args.format == "both" else [args.format]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if wants_hardcoded and pairs:
        print()
        run_hardcoded(args, pairs, out, formats)
    if wants_volumetric:
        print()
        run_width_scan(args, pairs, out, formats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
