import eq1val
import matplotlib.pyplot as plt
import pytest
from eq1bench.algorithms.circuit_benchmark import (
    CircuitBenchmark,
    align_to_ideal,
    classical_fidelity,
    ideal_distribution,
    plot_measured_vs_ideal,
)
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit

from circuits import CIRCUITS, load
from config import DEVICES

pytestmark = pytest.mark.opt_in

DEFAULT_SHOTS = 10000
MAX_RAW_ENTRIES = 4096
OPTIMIZATION_LEVEL = 1
OPTIMIZATION_LEVEL_OVERRIDES = {"B-3.1": 0}

# (task, algorithm, device, widths) -- the grid the spec names.
GRID = [
    ("B-1.1", "ghz", "Device-Ref", [3]),
    ("B-1.3", "ghz", "Device-1", [3, 10, 12]),
    ("B-1.4", "ghz", "Device-2", [3, 10, 12]),
    ("B-2.1", "qft", "Device-Ref", [4]),
    ("B-2.2", "qft", "Device-1", [4, 10]),
    ("B-2.3", "qft", "Device-2", [4, 10]),
    ("B-3.1", "qpe", "Device-Ref", [4]),
    ("B-3.2", "qpe", "Device-1", [4, 10]),
    ("B-3.3", "qpe", "Device-2", [4, 10]),
    ("B-4.1", "grover", "Device-Ref", [4]),
    ("B-4.2", "grover", "Device-1", [4, 6]),
    ("B-4.3", "grover", "Device-2", [4, 6]),
]

FAMILIES = {"GHZ": "ghz", "QFT": "qft", "QPE": "qpe", "Grover": "grover"}
MQT_NAMES = {"ghz": "ghz", "qft": "qft", "qpe": "qpeexact", "grover": "grover"}

# Grid cells with no stored circuit, generated from MQT Bench instead.
GENERATED = {("ghz", 12)}

TASKS = {
    (device, algorithm, width): task
    for task, algorithm, device, widths in GRID
    for width in widths
}

WIDTHS = sorted({width for _, _, _, widths in GRID for width in widths})


def circuit_for(algorithm: str, width: int) -> QuantumCircuit:
    """The stored circuit, or MQT Bench output where none is stored.

    Same origin either way: the stored sources are themselves verbatim MQT
    Bench output at the target-independent level.
    """
    if (algorithm, width) in CIRCUITS:
        return load(algorithm, width)
    return get_benchmark(
        benchmark=MQT_NAMES[algorithm],
        level=BenchmarkLevel.INDEP,
        circuit_size=width,
    )


@eq1val.category("LuxProvide")
@eq1val.subcategory("Algorithm Circuits")
@eq1val.subcategory.each("device", list(DEVICES))
@eq1val.subcategory.each("family", list(FAMILIES))
@eq1val.params("width", WIDTHS)
def test_algorithm(device, family, width, provider, pytestconfig):
    """Run one (device, family, width) grid cell and report what it measured.

    Nothing is asserted beyond the run having produced a distribution at all:
    the numbers and the plot are the deliverable, not a bar to clear.
    """
    algorithm = FAMILIES[family]
    task = TASKS.get((device, algorithm, width))
    if task is None:
        eq1val.drop()

    stored = (algorithm, width) in CIRCUITS
    if not stored and (algorithm, width) not in GENERATED:
        pytest.skip(f"no circuit for {algorithm} at {width} qubits")

    engine = pytestconfig.getoption("--simulator")
    shots = pytestconfig.getoption("--shots") or DEFAULT_SHOTS
    optimization_level = OPTIMIZATION_LEVEL_OVERRIDES.get(task, OPTIMIZATION_LEVEL)
    name = f"{task}-{algorithm}-{width}q"
    device_id = f"fake-{DEVICES[device]}"

    circuit = circuit_for(algorithm, width)
    backend = provider.get_backend(device_id)
    bench = CircuitBenchmark(
        provider=provider,
        circuit=circuit,
        name=name,
        shots=shots,
        device=backend,
        optimization_level=optimization_level,
    )
    bench.select_simulator(engine)
    bench.select_device(device_id=device_id, is_simulated=True)

    distribution = bench.run()
    assert distribution, f"{name}: the run returned no distribution"

    ideal = ideal_distribution(circuit)
    measured = align_to_ideal(distribution, circuit)
    fidelity = classical_fidelity(ideal, measured)

    fig = plot_measured_vs_ideal(
        measured, ideal, name=name, note=f"$F_s$ = {fidelity:.3f}"
    )
    eq1val.report_plot(fig, name=name)
    plt.close(fig)

    compiled = bench.submitted_circuits
    data = {
        "task": task,
        "algorithm": algorithm,
        "device": device,
        "device_id": DEVICES[device],
        "noise_builder": backend.target.instruction_scheduler,
        "simulator": engine,
        "width": width,
        "shots": shots,
        "circuit_source": "circuits.py" if stored else "mqt-bench (indep)",
        "transpilation": f"client-side (level {optimization_level})",
        "fidelity": round(fidelity, 4),
        "fidelity_definition": (
            "F_s = (sum_x sqrt(P_output(x) P_ideal(x)))^2 against the exact "
            "noiseless distribution"
        ),
        "circuit_depth": circuit.depth(),
        "gate_counts": dict(circuit.count_ops()),
        "compiled_circuit_depth": compiled.depth(),
        "compiled_gate_counts": dict(compiled.count_ops()),
        **{
            f"server_{stage}_s": round(seconds, 4)
            for stage, seconds in bench.metrics.items()
        },
    }
    eq1val.report_data(data)

    measured_top, measured_meta = eq1val.top_entries(measured, MAX_RAW_ENTRIES)
    ideal_top, ideal_meta = eq1val.top_entries(ideal, MAX_RAW_ENTRIES)
    eq1val.report_raw(
        {
            **data,
            "fidelity": fidelity,
            "optimization_level": optimization_level,
            "measured": measured_top,
            "measured_meta": measured_meta,
            "ideal": ideal_top,
            "ideal_meta": ideal_meta,
            "server_metrics": dict(bench.metrics),
        }
    )
