"""The quantum volume sweep: heavy output generation over random model circuits.

Vendored from `eq1bench/device_characterization/quantum_volume_benchmark.py` at
eq1val rev db660ef -- see this package's `__init__.py` for what that means for
keeping it current.
"""

import math
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, override

import numpy as np
from eq1_biskit import Eq1QiskitProvider
from eq1_biskit.backend import Simulator
from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure
from qiskit import QuantumCircuit
from qiskit.circuit.library import quantum_volume
from qiskit.providers.backend import BackendV2
from qiskit.quantum_info import Statevector

from .experiment import Experiment

# Heavy outputs are those above the median of the ideal distribution; HOG passes
# a width/depth point when at least this fraction of shots land on one.
# arXiv:1612.05903 (HOG), arXiv:1811.12926 sec. "Validating quantum computers", eq. 6.
HEAVY_OUTPUT_THRESHOLD = 2 / 3

# arXiv:1811.12926 sec. III.A: the heavy output probability of an ideal random
# circuit converges to this value as circuit size grows, plotted as a reference line.
ASYMPTOTIC_IDEAL_HEAVY_OUTPUT_PROBABILITY = (1 + math.log(2)) / 2

# Minimum number of independently sampled circuits per (width, depth) point.
# TODO: update this back once flow is tested
MIN_REPLICATES = 10


def _bitstring(index: int, width: int) -> str:
    """Zero-padded binary representation of `index` on `width` bits."""
    return format(index, f"0{width}b")


def _parse_size_key(key: str) -> tuple[int, int]:
    """Parse a "WIDTHxDEPTH" heavy-output-fractions key, as stored in the JSONL
    run database, back into the `(width, depth)` tuple key used in-process.
    """
    width, depth = key.split("x")
    return int(width), int(depth)


def plot_quantum_volume(
    heavy_output_fractions: dict[tuple[int, int], float],
    max_num_qubits: int,
    max_depth: int,
    quantum_volume: int,
    filename: Optional[str] = None,
) -> Figure:
    """Plot the square (`width == depth == m`) heavy output probability sweep, in
    the style of arXiv:1811.12926 fig. 2: measured heavy output fraction against
    model circuit size `m`, alongside the HOG threshold and the asymptotic ideal
    probability as reference lines.
    """
    assert heavy_output_fractions, "No heavy output fractions to plot."

    max_size = min(max_num_qubits, max_depth)
    sizes = [m for m in range(2, max_size + 1) if (m, m) in heavy_output_fractions]
    fractions = [heavy_output_fractions[(m, m)] for m in sizes]

    fig, ax = plt.subplots()
    ax.plot(sizes, fractions, "o", color="red", label=r"measured $\hat{h}_d$")
    ax.axhline(
        HEAVY_OUTPUT_THRESHOLD,
        color="black",
        linestyle=":",
        label="HOG threshold (2/3)",
    )
    ax.axhline(
        ASYMPTOTIC_IDEAL_HEAVY_OUTPUT_PROBABILITY,
        color="green",
        linestyle="--",
        label=r"asymptotic ideal $(1+\ln 2)/2$",
    )
    ax.set_xlabel("width/depth of model circuit $m=d$")
    ax.set_ylabel(r"est. heavy output probability $\hat{h}_d$")
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(sizes)
    ax.set_title(f"Quantum Volume = {quantum_volume}")
    ax.legend()
    if filename:
        fig.savefig(filename)
    return fig


def plot_quantum_volume_existing_results(
    database_path: str | os.PathLike[str],
    device: str,
    filename: Optional[str] = None,
) -> Figure:
    """Rebuild the quantum-volume graph for `device` from a run database's
    "test_quantum_volume" result (see `eq1val.database.load_session`), for
    plotting a past run's data without re-running the benchmark.
    """
    from eq1val.database import load_session

    view = load_session(database_path)
    result = next(
        run
        for run in view.runs
        if run.name == "test_quantum_volume" and run.params.get("device") == device
    )
    heavy_output_fractions = {
        _parse_size_key(key): fraction
        for key, fraction in result.data["heavy_output_fractions"].items()
    }
    # Not stored in the database: recovered as the largest circuit size any
    # attempted point touches, which is all `plot_quantum_volume` needs to
    # know which square points to plot.
    max_size = max(max(width, depth) for width, depth in heavy_output_fractions)
    return plot_quantum_volume(
        heavy_output_fractions,
        max_size,
        max_size,
        result.data["quantum_volume"],
        filename=filename,
    )


def _median_of_sorted_values(sorted_values: list[float]) -> float:
    """Median of an already-sorted, even-length sequence."""
    midpoint = len(sorted_values) // 2
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def _ideal_heavy_outputs(active_circuit: QuantumCircuit) -> frozenset[str]:
    """The bitstrings whose ideal probability is above the median of `active_circuit`'s
    output distribution (arXiv:1811.12926 eqs. 2-3).

    Computed exactly from the statevector of the logical (pre-measurement,
    device-idle-qubit-free) `width`-qubit circuit, so the reference carries no shot
    noise of its own and the simulated statevector never scales with device size.
    """
    width = active_circuit.num_qubits
    probabilities = Statevector(active_circuit).probabilities_dict()
    all_probabilities = sorted(
        probabilities.get(_bitstring(i, width), 0.0) for i in range(2**width)
    )
    median = _median_of_sorted_values(all_probabilities)
    return frozenset(
        bitstring
        for bitstring, probability in probabilities.items()
        if probability > median
    )


def _measured_qubits(
    device: BackendV2, width: int, optimization_level: int
) -> list[int]:
    """Qubit indices to measure: the active `width` qubits, plus whichever device
    concurrent-measurement group qubits (arXiv:1811.12926's model circuits aside,
    some devices only allow reading out certain qubits together) might otherwise be
    left partially measured.

    At `optimization_level` <= 1 the layout is trivial (virtual qubit `i` is placed on
    physical qubit `i`), so the active qubits' physical location is known ahead of
    time: only groups the active qubits actually land in -- i.e. intersect -- need
    their other, otherwise-idle members added.

    Above that, the layout is free to remap the active qubits onto any physical
    qubits, so which group, if any, they'll join isn't known at this point; every
    group is measured whole to sidestep that guess. This does not by itself guarantee
    a group is never left partially measured after layout; that would additionally
    need the layout to be pinned.
    """
    active_qubits = set(range(width))
    concurrent_measurements = device.target.concurrent_measurements or []
    if not concurrent_measurements:
        return sorted(active_qubits)

    if optimization_level > 1:
        # Remapping could put an active qubit in any group, so none can be ruled out.
        relevant_groups = concurrent_measurements
    else:
        relevant_groups = [
            group
            for group in concurrent_measurements
            if len(active_qubits & set(group)) > 0
        ]
    relevant_qubits = {qubit for group in relevant_groups for qubit in group}
    return sorted(active_qubits | relevant_qubits)


class QuantumVolumeBenchmark(Experiment):
    """Estimates a device's quantum volume via the heavy output generation (HOG) test.

    Sweeps circuit width `m` from 2 up to `max_num_qubits`. For each width, submits
    increasingly deep random "quantum volume" model circuits (Cross et al.,
    arXiv:1811.12926) until the HOG test (Aaronson & Chen, arXiv:1612.05903) fails,
    giving the achievable depth `d(m)`. The quantum volume is then
    `log2(V_Q) = max_m min(m, d(m))`.

    Each (width, depth) point is measured over `replicates` independently sampled
    circuits, run as one batched job -- a fresh job per point, since whether to try the
    next depth depends on whether this one passed.
    """

    def __init__(
        self,
        provider: Eq1QiskitProvider,
        max_num_qubits: int | None = None,
        max_depth: int | None = None,
        shots: int = 100,
        replicates: int = MIN_REPLICATES,
        device: BackendV2 | None = None,
        simulator: Simulator | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(provider, shots, None, device, simulator)

        if replicates < MIN_REPLICATES:
            raise ValueError(
                f"replicates must be >= {MIN_REPLICATES}, got {replicates}"
            )

        self.max_num_qubits = max_num_qubits
        self.max_depth = max_depth
        self.replicates = replicates
        self.rng = np.random.default_rng(seed)

        self.achievable_depths: dict[int, int] = {}
        self.heavy_output_fractions: dict[tuple[int, int], float] = {}
        self.quantum_volume: Optional[int] = None
        self.log2_quantum_volume: Optional[int] = None
        # Ideal heavy outputs for whichever (width, depth) batch is currently held in
        # `self.circuits` -- both are always written together by `generate_circuits()`.
        self.heavy_sets: Optional[list[frozenset[str]]] = None
        # `generate_circuits()` already transpiles (see below), so `start_job()` must not.
        self.client_side_transpile = False

    @override
    def generate_circuits(
        self, width: Optional[int] = None, depth: Optional[int] = None
    ) -> None:
        """Build `replicates` circuits for (width, depth) and their ideal heavy sets.

        NOTE -- side effect, not a pure builder: stores results directly into
        `self.circuits` (transpiled, submission-ready) and `self.heavy_sets`, instead
        of returning them. Both are rebound to new objects rather than mutated in
        place, so a caller that snapshotted the previous `self.circuits`/
        `self.heavy_sets` into local variables beforehand keeps a stable, unaffected
        view after this runs again -- e.g. from another thread.

        Each circuit is sized to just the active `width` qubits plus any device
        concurrent-measurement group (see `_measured_qubits`), not the whole device.
        A full-device-sized register with only some of it measured still costs almost
        as much: the transpiler's per-qubit scheduling and noise-modelling passes
        (thermal relaxation, SPAM, ...) run over every qubit *declared* in the circuit,
        whether or not it ends up measured. Declaring only the qubits that matter keeps
        that cost -- and the simulated statevector -- proportional to the HOG point
        under test rather than to the device size.

        `width` and `depth` are optional only to match the base signature; both are required.
        """
        assert width is not None and depth is not None, (
            "generate_circuits() requires width and depth"
        )
        assert self._device is not None, "No device set — call select_device() first."
        num_circuit_qubits = len(
            _measured_qubits(self._device, width, self.transpile_optimization_level)
        )
        active_circuits = []
        submission_circuits = []
        for _ in range(self.replicates):
            active_circuit = quantum_volume(width, depth, seed=self.rng)
            active_circuits.append(active_circuit)
            circuit = QuantumCircuit(num_circuit_qubits, num_circuit_qubits)
            circuit.compose(active_circuit, qubits=range(width), inplace=True)
            circuit.measure(range(num_circuit_qubits), range(num_circuit_qubits))
            submission_circuits.append(circuit)

        self.heavy_sets = [_ideal_heavy_outputs(c) for c in active_circuits]
        self.circuits = self._device.transpile(
            submission_circuits, optimization_level=self.transpile_optimization_level
        )

    @override
    def run(
        self,
        on_width_done: Callable[[int, int], None] | None = None,
        **kwargs,
    ) -> int:
        """Sweep width and depth, running the HOG test at each point, and return the
        quantum volume `V_Q`.

        `on_width_done`, if given, is called with `(width, achievable_depth)`
        once a width's own depth sweep is finished and recorded, before the
        next width starts -- so a caller can persist/report progress (e.g. for
        a long sweep that might be interrupted or fail partway through)
        without waiting for the whole sweep to finish.
        """
        self._set_max_num_qubits()
        self.max_depth = self.max_depth or self.max_num_qubits
        if "optimization_level" in kwargs:
            self.transpile_optimization_level = kwargs.pop("optimization_level")
        self.achievable_depths = {}
        self.heavy_output_fractions = {}

        self.log2_quantum_volume = self._sweep_width(on_width_done, **kwargs)
        self.quantum_volume = 2**self.log2_quantum_volume
        return self.quantum_volume

    def _set_max_num_qubits(self) -> None:
        if self.max_num_qubits is None:
            if self._device is None:
                raise ValueError(
                    "Cannot run experiment without maximum number of qubits. "
                    "Please select a device or set max_num_qubits first."
                )
            self.max_num_qubits = self._device.num_qubits
        elif self._device:
            if self.max_num_qubits > self._device.num_qubits:
                self.max_num_qubits = self._device.num_qubits

    def _sweep_width(
        self, on_width_done: Callable[[int, int], None] | None, **kwargs
    ) -> int:
        assert self.max_num_qubits is not None
        log2_volume = 0
        for width in range(2, self.max_num_qubits + 1):
            print(f"[checkpoint] sweeping width={width}")
            depth = self._find_achievable_depth(width, **kwargs)
            self.achievable_depths[width] = depth
            log2_volume = max(log2_volume, min(width, depth))
            print(
                f"[checkpoint] width={width} achievable_depth={depth} log2_volume_so_far={log2_volume}"
            )
            if on_width_done is not None:
                on_width_done(width, depth)

        return log2_volume

    def _find_achievable_depth(self, width: int, **kwargs) -> int:
        """The largest depth at which `width`-qubit model circuits pass the HOG test,
        continuing past intermediate failures below `width` so every (width, depth)
        point up to and including the square `depth == width` one gets measured and
        logged for plotting.

        Below `width`, `min(width, depth)` cannot yet be saturated at `width`, so a
        failure there does not by itself rule out a later depth passing; the sweep
        only stops for good once a failure occurs at `depth >= width`
        (arXiv:1811.12926 eq. 6). Past `width`, `min(width, depth)` is already
        saturated, so continuing to a larger achievable depth cannot raise the
        quantum volume, but is kept for reporting.
        """
        assert self.max_depth is not None
        achievable_depth = 0
        for depth in range(1, self.max_depth + 1):
            print(f"[checkpoint]     depth={depth}")
            if self._passes_heavy_output_test(width, depth, **kwargs):
                achievable_depth = depth
            elif depth >= width:
                break
        return achievable_depth

    def _passes_heavy_output_test(self, width: int, depth: int, **kwargs) -> bool:
        """Run `replicates` circuits of this shape and check the HOG heavy-output fraction.

        While this depth's job is submitted and polled, concurrently prepares the
        next depth's circuits (if the sweep isn't done) into `self.circuits`/
        `self.heavy_sets`, so the next call reuses them instead of generating and
        then waiting serially. This depth's own circuits/heavy sets are snapshotted
        into locals first, so that concurrent preparation never affects the batch
        actually being scored here.
        """
        assert self.shots is not None
        assert self.max_depth is not None

        if not self.circuits or not self.heavy_sets:
            self.generate_circuits(width, depth)
        assert self.circuits and self.heavy_sets
        current_circuits = self.circuits.copy()
        current_heavy_sets = self.heavy_sets.copy()

        next_depth = depth + 1
        if next_depth <= self.max_depth:
            with ThreadPoolExecutor(max_workers=1) as executor:
                current_run_score = executor.submit(
                    self._submit_and_score,
                    current_circuits,
                    width,
                    depth,
                    current_heavy_sets,
                    **kwargs,
                )
                self.generate_circuits(width, next_depth)
                passed = current_run_score.result()
        else:
            passed = self._submit_and_score(
                current_circuits, width, depth, current_heavy_sets, **kwargs
            )

        if not (passed and next_depth <= self.max_depth):
            # Either the test failed (the next batch prepared above is discarded --
            # wasted work, but acceptable) or the sweep is done -- force the next
            # width to generate fresh circuits instead of reusing stale ones.
            self.circuits = None
            self.heavy_sets = None

        return passed

    def _submit_and_score(
        self,
        circuits: list[QuantumCircuit],
        width: int,
        depth: int,
        heavy_sets: list[frozenset[str]],
        **kwargs,
    ) -> bool:
        """Submit `circuits` as one job, wait for it, and score it against `heavy_sets`."""
        assert self.shots is not None
        self.flush_results()
        self.start_job(circuits, **kwargs)
        self.collect_results(as_probabilities=False)

        num_heavy_shots = self._count_heavy_shots(width, heavy_sets)
        heavy_output_fraction = num_heavy_shots / (self.replicates * self.shots)
        self.heavy_output_fractions[(width, depth)] = heavy_output_fraction

        return heavy_output_fraction > HEAVY_OUTPUT_THRESHOLD

    def _count_heavy_shots(self, width: int, heavy_sets: list[frozenset[str]]) -> int:
        """Total shots landing on a heavy output, across all circuits of one (width,
        depth) point. Counts are read in submission order, matching `heavy_sets`.

        The active qubits are always the lowest-indexed of the measured qubits (see
        `generate_circuits`), so they land in the lowest classical bits, i.e. the
        trailing `width` characters of each count key.
        """
        counts_per_circuit = self._flatten_results()
        total_heavy_shots = 0
        for counts, heavy_set in zip(counts_per_circuit, heavy_sets):
            for bitstring, count in counts.items():
                if bitstring[-width:] in heavy_set:
                    total_heavy_shots += int(count)
        return total_heavy_shots

    def plot_graph(self, filename: Optional[str] = None) -> Figure:
        """Plot the square (`width == depth == m`) heavy output probability sweep, in
        the style of arXiv:1811.12926 fig. 2: measured heavy output fraction against
        model circuit size `m`, alongside the HOG threshold and the asymptotic ideal
        probability as reference lines.
        """
        assert self.heavy_output_fractions, "Call run() before plot_graph()."
        assert self.max_num_qubits is not None and self.max_depth is not None
        assert self.quantum_volume is not None

        return plot_quantum_volume(
            self.heavy_output_fractions,
            self.max_num_qubits,
            self.max_depth,
            self.quantum_volume,
            filename=filename,
        )
