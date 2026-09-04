import eq1val
import pytest
from eq1bench.algorithms.circuit_benchmark import CircuitBenchmark, ideal_distribution
from eq1bench.volumetric import Cell, plot_volumetric
from matplotlib import pyplot as plt
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity

from config import DEVICES

pytestmark = pytest.mark.opt_in

# The reference device is not swept: this suite compares the two real ones.
SCAN_DEVICES = ["Device-1", "Device-2"]

MIN_WIDTH = 2
DEFAULT_MAX_WIDTH = 6
ALGORITHM_MAX_WIDTH = {"grover": 7, "qft": 20, "qpeexact": 20}

# mqt-bench name -> (report label, task id)
ALGORITHMS = {
    "ghz": ("GHZ", "B-1.5"),
    "qft": ("QFT", "B-2.4"),
    "qpeexact": ("QPE", "B-3.4"),
    "grover": ("Grover", "B-4.4"),
}
FAMILIES = {label: algorithm for algorithm, (label, _) in ALGORITHMS.items()}

OPTIMIZATION_LEVEL = 3
DEFAULT_SHOTS = 1000
MAX_RAW_ENTRIES = 4096
MAX_COMPILED_DEPTH = 200_000
CEILING_REASON = "compiled depth ceiling"
COMBINED_PAGE = "All Algorithms"

SCORE_THRESHOLD = 2 / 3
SCORE_DEFINITION = f"widest width with fidelity >= {SCORE_THRESHOLD:.2f} (PROVISIONAL)"

MEASURED = "measured"


def _max_width(config) -> int:
    top = config.getoption("--max-width")
    top = DEFAULT_MAX_WIDTH if top is None else top
    if top < MIN_WIDTH:
        raise pytest.UsageError(
            f"--max-width must be at least {MIN_WIDTH}, got {top}: the scan "
            f"sweeps [{MIN_WIDTH}, --max-width] and cannot be empty"
        )
    return top


def widths(config, algorithm: str) -> list[int]:
    """The widths this run sweeps for one algorithm, narrowest first."""
    top = _max_width(config)
    return list(range(MIN_WIDTH, min(top, ALGORITHM_MAX_WIDTH.get(algorithm, top)) + 1))


def shots(config) -> int:
    return config.getoption("--shots") or DEFAULT_SHOTS


def pytest_generate_tests(metafunc):
    if "width" in metafunc.fixturenames:
        metafunc.parametrize(
            "width", list(range(MIN_WIDTH, _max_width(metafunc.config) + 1))
        )


def _family_runs(device: str, family: str) -> list[eq1val.TestResult]:
    """One (device, family) sweep's per-width runs so far, narrowest first."""
    runs = eq1val.test_runs(
        name="test_width_run", params={"device": device, "family": family}
    )
    return sorted(runs, key=lambda run: run.params["width"])


def _measured(runs: list[eq1val.TestResult]) -> dict[int, tuple[int, float]]:
    """width -> (logical depth, fidelity) for the runs that measured one."""
    return {
        run.params["width"]: (run.data.get("logical_depth"), run.data["fidelity"])
        for run in runs
        if run.data.get("outcome") == MEASURED
    }


def _device_qubits(runs: list[eq1val.TestResult]) -> int | None:
    """The device's qubit count as the runs reported it, None if none knows.

    A run that gave up before asking has no reading, and is passed over rather
    than erasing a real one.
    """
    return next(
        (
            run.data["device_qubits"]
            for run in reversed(runs)
            if run.data.get("device_qubits") is not None
        ),
        None,
    )


def _report_raw(
    data: dict, fidelity: float, measured: dict, ideal: dict, metrics: dict
) -> None:
    """Record one width's distributions in the run database."""
    measured_top, measured_meta = eq1val.top_entries(measured, MAX_RAW_ENTRIES)
    ideal_top, ideal_meta = eq1val.top_entries(ideal, MAX_RAW_ENTRIES)
    eq1val.report_raw(
        {
            **data,
            "fidelity": fidelity,
            "optimization_level": OPTIMIZATION_LEVEL,
            "fidelity_definition": (
                "Hellinger fidelity vs the exact distribution, rescaled so "
                "uniform scores 0 (see normalized_fidelity)"
            ),
            "measured": measured_top,
            "measured_meta": measured_meta,
            "ideal": ideal_top,
            "ideal_meta": ideal_meta,
            "server_metrics": dict(metrics),
        }
    )


def _cells(runs: list[eq1val.TestResult]) -> list[Cell]:
    """One sweep's runs as plottable cells, narrowest first."""
    return [
        Cell(
            width=run.params["width"],
            depth=run.data["compiled_depth"],
            fidelity=run.data["fidelity"]
            if run.data.get("outcome") == MEASURED
            else None,
        )
        for run in runs
        if run.data.get("compiled_depth") is not None
    ]


def _mirror(qc: QuantumCircuit) -> QuantumCircuit:
    """The circuit followed by its own inverse, measured at the end."""
    qc = qc.remove_final_measurements(inplace=False)

    benchmark = QuantumCircuit(qc.num_qubits)
    benchmark.compose(qc, inplace=True)
    benchmark.compose(qc.inverse(), inplace=True)
    benchmark.measure_all()
    return benchmark


def normalized_fidelity(ideal_counts, observed_counts, num_qubits: int) -> float:
    """Hellinger fidelity rescaled so that a dead, uniform device scores 0."""
    Fs = hellinger_fidelity(ideal_counts, observed_counts)

    uniform = {format(i, f"0{num_qubits}b"): 1 for i in range(2**num_qubits)}
    Fs_uniform = hellinger_fidelity(ideal_counts, uniform)

    if Fs_uniform >= 1.0:
        return 0.0

    return max((Fs - Fs_uniform) / (1 - Fs_uniform), 0.0)


def score(results: dict[int, tuple[int, float]]) -> float:
    cleared = [w for w, (_, fidelity) in results.items() if fidelity >= SCORE_THRESHOLD]
    return float(max(cleared)) if cleared else 0.0


@eq1val.category("LuxProvide")
@eq1val.subcategory("Volumetric Benchmarks")
@eq1val.subcategory.each("device", SCAN_DEVICES)
@eq1val.subcategory.each("family", list(FAMILIES))
@eq1val.heading("Runs")
def test_width_run(device, family, width, provider, backends, pytestconfig):
    """Run one algorithm at one width on one device and report what it did."""
    algorithm = FAMILIES[family]
    if width not in widths(pytestconfig, algorithm):
        eq1val.drop()

    task = ALGORITHMS[algorithm][1]
    engine = pytestconfig.getoption("--simulator")
    n_shots = shots(pytestconfig)
    run_id = f"{device}-{algorithm}-{width}q"

    data = {
        "task": task,
        "algorithm": family,
        "device": device,
        "device_id": DEVICES[device],
        "simulator": engine,
        "width": width,
        "shots": n_shots,
    }

    def give_up(reason: str):
        """Report a width that was never measured, as far as it got, and skip.

        A resumed sweep replays skips as-is, transient ones included: an
        "unreachable device" afternoon is cleared by --eq1-rerun-failed.
        """
        eq1val.report_data({**data, "fidelity": "not measured", "outcome": reason})
        pytest.skip(reason)

    backend = backends[device]
    if isinstance(backend, str):
        give_up(backend)

    capacity = getattr(backend, "num_qubits", None)
    data["device_qubits"] = capacity if isinstance(capacity, int) else None
    if isinstance(capacity, int) and width > capacity:
        give_up(f"width {width} exceeds the device's {capacity} qubits")

    try:
        circuit = get_benchmark(
            benchmark=algorithm, level=BenchmarkLevel.INDEP, circuit_size=width
        )
    except Exception as exc:
        give_up(f"mqt-bench cannot generate {algorithm} at {width}q: {exc}")

    if algorithm == "qft":
        circuit = _mirror(circuit)

    data["logical_depth"] = circuit.depth()

    compiled = backend.default_transpile(circuit, optimization_level=OPTIMIZATION_LEVEL)
    data["compiled_depth"] = compiled.depth()
    data["compiled_gates"] = ", ".join(
        f"{gate}={n}" for gate, n in sorted(compiled.count_ops().items())
    )
    if compiled.depth() > MAX_COMPILED_DEPTH:
        give_up(
            f"{CEILING_REASON}: {compiled.depth():,} exceeds "
            f"{MAX_COMPILED_DEPTH:,} layers"
        )

    bench = CircuitBenchmark(
        provider=provider,
        circuit=compiled,
        name=run_id,
        shots=n_shots,
        device=backend,
        optimization_level=OPTIMIZATION_LEVEL,
    )
    bench.client_side_transpile = False
    bench.select_simulator(engine)
    bench.select_device(device_id=f"fake-{DEVICES[device]}", is_simulated=True)

    try:
        measured = bench.run()
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        eq1val.report_data({**data, "fidelity": "not measured", "outcome": reason})
        eq1val.fail(f"the run raised {reason}")
        return

    ideal = ideal_distribution(circuit)
    fidelity = normalized_fidelity(ideal, measured, circuit.num_qubits)
    data["fidelity"] = round(fidelity, 6)
    data["outcome"] = MEASURED
    data.update({f"server_{k}_s": round(v, 4) for k, v in bench.metrics.items()})
    eq1val.report_data(data)

    _report_raw(data, fidelity, measured, ideal, bench.metrics)

    fig = bench.plot_graph()
    eq1val.report_plot(fig, name=f"{run_id}-distribution")
    plt.close(fig)


@eq1val.category("LuxProvide")
@eq1val.subcategory("Volumetric Benchmarks")
@eq1val.subcategory.each("device", SCAN_DEVICES)
@eq1val.subcategory.each("family", list(FAMILIES))
@eq1val.heading("Grid & score")
@eq1val.needs("test_width_run")
def test_width_grid(device, family, pytestconfig):
    """The volumetric grid and the score for one (device, family) sweep.

    Runs no circuits: it plots what the per-width runs reported, read back via
    eq1val.test_runs().
    """
    algorithm = FAMILIES[family]
    task = ALGORITHMS[algorithm][1]
    scan = widths(pytestconfig, algorithm)
    runs = _family_runs(device, family)
    measured = _measured(runs)

    if not measured:
        pytest.skip(f"no width of {family} completed on {device}: nothing to grid")
    if len(runs) < len(scan):
        eq1val.report_warning(
            f"partial sweep: {len(runs)} of {len(scan)} widths reached this "
            f"grid (missing {sorted(set(scan) - {r.params['width'] for r in runs})})"
        )

    fig = plot_volumetric(
        {family: _cells(runs)},
        title=f"{task} — {family} width scan on {device}",
        device_qubits=_device_qubits(runs),
        depth_ceiling=MAX_COMPILED_DEPTH,
    )
    eq1val.report_plot(fig, name=f"{task}-{algorithm}-{device}-width-scan")
    plt.close(fig)

    widest = max(measured)
    by_width = {run.params["width"]: run for run in runs}
    gaps = [w for w in scan if w not in measured]
    eq1val.report_data(
        {
            "task": task,
            "algorithm": family,
            "device": device,
            "device_id": DEVICES[device],
            "simulator": pytestconfig.getoption("--simulator"),
            "shots": shots(pytestconfig),
            "score": score(measured),
            "score_definition": SCORE_DEFINITION,
            "widths_measured": f"{len(measured)} of {len(scan)}",
            "max_width_measured": widest,
            "best_fidelity": round(max(f for _, f in measured.values()), 4),
            "fidelity_at_max_width": round(measured[widest][1], 4),
            "first_gap": gaps[0] if gaps else "none",
            "first_gap_reason": by_width[gaps[0]].data.get("outcome", "")
            if gaps and gaps[0] in by_width
            else "not attempted",
        }
    )


@eq1val.category("LuxProvide")
@eq1val.subcategory("Volumetric Benchmarks")
@eq1val.subcategory.each("device", SCAN_DEVICES)
@eq1val.subcategory(COMBINED_PAGE)
@eq1val.heading("Combined grid")
@eq1val.needs("test_width_run")
def test_width_combined(device, pytestconfig):
    """Every algorithm's sweep on one device, on one grid.

    Runs no circuits: same per-width runs as test_width_grid, read back via
    eq1val.test_runs(). The depth axis is logarithmic, so Grover sitting
    several columns right of GHZ at the same width means orders of magnitude
    more layers for the same register.
    """
    by_family = {family: _family_runs(device, family) for family in FAMILIES}
    series = {
        family: cells for family, runs in by_family.items() if (cells := _cells(runs))
    }

    if not series:
        pytest.skip(f"no algorithm produced a cell on {device}: nothing to grid")
    if missing := [family for family in FAMILIES if family not in series]:
        eq1val.report_warning(
            f"partial comparison: {', '.join(missing)} placed no cell on this "
            "grid, so the algorithms shown are not the full set"
        )

    fig = plot_volumetric(
        series,
        title=f"Width scan — all algorithms on {device}",
        device_qubits=_device_qubits([r for runs in by_family.values() for r in runs]),
        depth_ceiling=MAX_COMPILED_DEPTH,
    )
    eq1val.report_plot(fig, name=f"{device}-all-algorithms-width-scan")
    plt.close(fig)

    data = {
        "device": device,
        "device_id": DEVICES[device],
        "simulator": pytestconfig.getoption("--simulator"),
        "shots": shots(pytestconfig),
        "score_definition": SCORE_DEFINITION,
        "algorithms_compared": len(series),
    }
    for family, runs in by_family.items():
        measured = _measured(runs)
        data[f"score_{family}"] = score(measured) if measured else "not measured"
        data[f"max_width_{family}"] = max(measured) if measured else "not measured"
    eq1val.report_data(data)
