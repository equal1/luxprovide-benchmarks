"""Running one pre-built circuit, and scoring it against the exact answer.

Vendored from `eq1bench/algorithms/circuit_benchmark.py` at eq1val rev db660ef
-- see this package's `__init__.py` for what that means for keeping it current.
"""

import math
from typing import Optional, override

from eq1_biskit import Eq1BackendBase, Eq1QiskitProvider
from eq1_biskit.backend import Simulator
from matplotlib.pyplot import Figure
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.result import marginal_distribution

from .experiment import Counts, Experiment


MEASURED_COLOR = "#2a78d6"
IDEAL_COLOR = "#eb6834"

IDEAL_FILL_ALPHA = 0.22
IDEAL_EDGE_ALPHA = 0.9
MEASURED_FILL_ALPHA = 0.92

SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY = "#000000"
TEXT_SECONDARY = "#000000"
GRID_COLOR = "#e7e6e1"
AXIS_COLOR = "#d9d8d4"

IDEAL_BAR_WIDTH = 0.78
MEASURED_BAR_WIDTH = 0.40


def measured_clbits(circuit: QuantumCircuit) -> dict[int, int]:
    """Clbit index -> the qubit measured into it, for each of `circuit`'s measures.

    Read off the circuit's own measure instructions rather than assumed: a
    circuit may measure fewer qubits than it has (``qpeexact`` leaves its ``psi``
    eigenstate ancilla unmeasured) and may measure them into clbits in any order.
    """
    clbit_to_qubit: dict[int, int] = {}
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            clbit = circuit.find_bit(instruction.clbits[0]).index
            clbit_to_qubit[clbit] = circuit.find_bit(instruction.qubits[0]).index
    if not clbit_to_qubit:
        raise ValueError(f"circuit {circuit.name!r} contains no measurements")
    return clbit_to_qubit


def ideal_distribution(circuit: QuantumCircuit) -> Counts:
    """The circuit's noiseless outcome probabilities, keyed as counts are.

    Computed exactly from the statevector rather than sampled, so a reference
    built from it carries no shot noise of its own.

    Keys cover the measured clbits only, low clbit rightmost. A measured
    distribution does not necessarily arrive keyed that way -- see
    `align_to_ideal`, which is how one is brought onto these keys.

    Pass the *logical* circuit, not the compiled one: the two are unitarily
    equivalent, and the logical one is far cheaper to simulate.
    """
    clbit_to_qubit = measured_clbits(circuit)

    bare = circuit.remove_final_measurements(inplace=False)
    probabilities = Statevector(bare).probabilities_dict()
    return marginal_distribution(
        probabilities, indices=[clbit_to_qubit[c] for c in sorted(clbit_to_qubit)]
    )


def classical_fidelity(ideal: Counts, output: Counts) -> float:
    r"""The fidelity of a measured distribution against an exact one.

    .. math::

        F_s(P_{ideal}, P_{output}) =
            \left( \sum_x \sqrt{P_{output}(x) P_{ideal}(x)} \right)^2

    The squared Bhattacharyya coefficient: 1 when the two distributions agree
    everywhere, 0 when they share no outcome at all. Outcomes missing from
    either side contribute nothing, so only the shared support is summed.

    Written out rather than taken from ``qiskit.quantum_info``, whose
    `hellinger_fidelity` computes :math:`(1 - H^2)^2` and is the same quantity
    by way of :math:`H^2 = 1 - \sum_x \sqrt{pq}` -- but only recognisably so
    after that substitution. This is the definition the spec states, in the
    form it states it.

    Both arguments are normalised first, as qiskit does: the formula is defined
    on distributions, and a caller holding raw counts or a clipped tail would
    otherwise get a number that is not a fidelity at all.
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


def align_to_ideal(distribution: Counts, circuit: QuantumCircuit) -> Counts:
    """`distribution` re-keyed to match `ideal_distribution(circuit)`'s keys.

    Two things put a measured key out of step with the exact one, and both come
    from clbits the circuit owns but the reference does not describe. Qiskit
    joins a multi-register key with spaces, last-declared register leftmost, so
    a QFT source that declares an unused ``creg c[4]`` beside the ``meas[4]`` it
    measures into comes back as ``"1110 0000"`` against the reference's
    ``"1110"``. A register that is never measured also occupies clbits the
    reference simply has no bit for.

    One fix for both: drop the spaces, leaving one character per clbit, then
    keep the clbits the circuit actually measures, in the order
    `ideal_distribution` writes them. Probabilities of keys that collapse onto
    the same aligned key are summed, so the result still totals what went in.

    A circuit whose only register is the one it measures passes through
    unchanged.
    """
    wanted = sorted(measured_clbits(circuit))
    aligned: Counts = {}
    for key, value in distribution.items():
        bits = key.replace(" ", "")
        if len(bits) <= wanted[-1]:
            raise ValueError(
                f"key {key!r} has {len(bits)} bits, too few for clbit {wanted[-1]} "
                f"of circuit {circuit.name!r}"
            )
        selected = "".join(bits[-(clbit + 1)] for clbit in reversed(wanted))
        aligned[selected] = aligned.get(selected, 0.0) + value
    return aligned


def plot_measured_vs_ideal(
    measured: Counts,
    ideal: Counts,
    name: str,
    top_n: int = 15,
    note: str | None = None,
) -> Figure:
    """Bar-plot a measured distribution over its exact one, superimposed.

    One x position per outcome, both bars centred on it: the exact probability
    as a wide bar behind, the measured one as a narrow bar in front. What is
    left showing of the wide bar *is* the error on that outcome -- an ideal
    shoulder above the measured bar is probability the noise took away, and a
    measured bar standing above the ideal one is probability it added. Reading
    that off a pair of adjacent bars means comparing two heights by eye;
    stacking them on one baseline turns it into one length.

    Both distributions must be keyed the same way -- run a measured one through
    `align_to_ideal` first.

    Which outcomes to show is the other half of it. Taking the top of the
    measured distribution alone (what `CircuitBenchmark.plot_graph` does) is
    wrong here: a noisy enough run pushes the correct answers out of its top
    `top_n` entirely, and the plot would then show a gap of exactly zero because
    every ideal bar is off-screen. So the candidates are the top of *both*,
    ranked by whichever distribution weights them more heavily, and the ones
    drawn are ordered by ideal probability -- the outcomes the circuit is
    supposed to produce first, in a fixed order, however the noise fell.
    """
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba

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
        label="Ideal",
    )
    ax.bar(
        positions,
        [weight(measured, s) for s in states],
        width=MEASURED_BAR_WIDTH,
        facecolor=to_rgba(MEASURED_COLOR, MEASURED_FILL_ALPHA),
        linewidth=0,
        label="Simulated",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(states, rotation=90, family="monospace", fontsize=8.5)
    ax.set_xlim(-0.7, len(states) - 0.3)
    ax.set_ylabel("Probability", fontsize=9, color=TEXT_SECONDARY, labelpad=8)
    ax.set_xlabel("Outcome", fontsize=9, color=TEXT_SECONDARY, labelpad=6)

    if note is not None:
        ax.text(
            0.0,
            1.035,
            note,
            transform=ax.transAxes,
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=2,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        handlelength=1.1,
        handleheight=1.1,
        # borderpad=1.1,
        borderpad=0.0,
        columnspacing=1.4,
    )

    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(length=0, colors=TEXT_SECONDARY, labelsize=8.5)

    fig.tight_layout()
    return fig


class CircuitBenchmark(Experiment):
    """Runs one pre-built circuit as a single job and reports what came back.

    Unlike the other experiments here, this one does not build its circuit --
    it is handed one (in practice parsed from a ``.qasm`` file) and only owns
    submitting it and reporting the result. That is what lets the same
    byte-identical circuit go to several devices, so any difference in the
    measured distribution is a difference in the noise, not in the input.

    Meant to be instantiated per (circuit, device) pair, e.g. from a
    parametrized test. ``name`` is carried only for labelling plots.
    """

    def __init__(
        self,
        provider: Eq1QiskitProvider,
        circuit: QuantumCircuit,
        name: str,
        shots: int = 10000,
        device: Eq1BackendBase | None = None,
        simulator: Simulator | None = None,
        optimization_level: int = 1,
    ) -> None:
        super().__init__(
            provider=provider,
            shots=shots,
            circuits=circuit,
            device=device,
            simulator=simulator,
        )
        self.name = name
        self.client_side_transpile = True
        self.transpile_optimization_level = optimization_level
        self.distribution: Optional[Counts] = None
        self.metrics: dict[str, float] = {}

    @override
    def generate_circuits(self) -> QuantumCircuit:
        """No-op: the circuit was supplied at construction.

        Defined because `Experiment` declares it abstract. Regenerating is
        precisely what this class exists to avoid.
        """
        assert self.circuits is not None
        return self.circuits

    @override
    def run(self, **kwargs) -> Counts:
        """Run the circuit as a single job and return its measured distribution.

        Also populates `self.metrics` with the server's per-stage timings.
        """
        assert self.circuits is not None
        self.flush_results()
        self.start_job(**kwargs)
        job_id = self._active_jobs[-1].job_id()
        self.collect_results()
        self.metrics = self._fetch_execution_metrics(job_id)
        self.distribution = self._flatten_results()[0]
        return self.distribution

    def _fetch_execution_metrics(self, job_id: str) -> dict[str, float]:
        """The server's per-stage timings for `job_id` (compilation, noise
        building, execution, ...), in seconds.

        Fetched from the raw result payload rather than taken off the job:
        `Eq1Job.result()` builds its qiskit `Result` out of the measurement
        counts alone, so the `execution_metrics` the server sends alongside them
        never reach the caller.

        Timings decorate the report, they are not the point of the run, so a
        server that does not supply them yields {} instead of failing the run.
        """
        try:
            result = self._provider._client.quantum_jobs.get_result(job_id)
            return dict(result.execution_metrics)
        except Exception:
            return {}

    def plot_graph(self, top_n: int = 15) -> Figure:
        """Bar-plot the `top_n` most probable measured outcomes."""
        assert self.distribution is not None, "Call run() before plot_graph()."
        from matplotlib import pyplot as plt

        top = sorted(self.distribution.items(), key=lambda kv: kv[1], reverse=True)[
            :top_n
        ]
        states, probs = zip(*top)

        fig, ax = plt.subplots()
        ax.bar(states, probs)
        ax.set_xlabel("Measured String")
        ax.set_ylabel("Probability")
        ax.set_title(self.name)
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        return fig
