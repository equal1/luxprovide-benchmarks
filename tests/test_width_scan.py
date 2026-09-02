import os
from dataclasses import asdict, dataclass, fields

import pytest
from eq1_biskit import Eq1QiskitProvider
from eq1client import Eq1Client
from matplotlib import pyplot as plt
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity

import eq1val
from eq1bench.algorithms.circuit_benchmark import CircuitBenchmark, ideal_distribution
from eq1bench.volumetric import Cell, plot_volumetric

pytestmark = pytest.mark.opt_in

SERVER_URL = "http://0.0.0.0:62123"
CLIENT_TIMEOUT_S = 60 * 60
DEVICES = {"Device-1": os.environ.get("EQ1_DEVICE_1"), "Device-2": os.environ.get("EQ1_DEVICE_2")}

MIN_WIDTH = 2
DEFAULT_MAX_WIDTH = 6
ALGORITHM_MAX_WIDTH = {"grover": 7, "qft": 20, "qpeexact": 20}

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

NO_RUNS_HINT = (
    "this grid plots what the per-width runs left behind, so the file has to "
    "be run whole -- a -k selection, -x or -n that drops the runs drops this too"
)


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
    """Parametrize widths from the command line.

    Widths cannot come from a decorator the way the device and family axes do:
    --max-width can only be read once a config exists. Every family gets the
    full range; a width past a family's own cap drops itself in test_width_run.
    """
    if "width" in metafunc.fixturenames:
        metafunc.parametrize(
            "width", list(range(MIN_WIDTH, _max_width(metafunc.config) + 1))
        )


@dataclass(frozen=True)
class Outcome:
    """What one width produced. `fidelity is None` means it was not measured."""

    logical_depth: int | None = None
    compiled_depth: int | None = None
    fidelity: float | None = None
    reason: str = ""
    over_ceiling: bool = False


def _sweeps(device_node) -> dict[str, dict[int, Outcome]]:
    """One device's outcomes so far: family label -> width -> Outcome.

    Written by test_width_run, read by the grids. Lives on the device's report
    node so each device's subtree reads its own accumulation, and the All
    Algorithms page -- a different leaf under the same device -- reads the same
    one. The coupling to execution order is real, so the grids guard it.
    """
    return device_node.data.setdefault("sweeps", {})


def _fingerprint(engine: str, n_shots: int) -> str:
    """What a stored width was measured under, for --eq1-resume.

    Every term changes the number a run produces, or whether it runs at all, so
    a change to any of them has to rerun rather than resume. Device, algorithm
    and width are absent because they are the record's identity, which a resume
    matches on before it ever looks at the fingerprint.
    """
    return (
        f"{engine}/shots={n_shots}/opt={OPTIMIZATION_LEVEL}"
        f"/ceiling={MAX_COMPILED_DEPTH}"
    )


def _remember(device_node, family: str, width: int, outcome: Outcome) -> None:
    """Record one width's outcome for this run's grids and for a later resume.

    Node data dies with the session, so without the reported state a resumed
    sweep would show a full set of restored run cards above four empty grids.
    """
    _sweeps(device_node).setdefault(family, {})[width] = outcome
    eq1val.report_state(
        {**asdict(outcome), "device_qubits": device_node.data.get("qubits")}
    )


def _rehydrate(device_node, family: str, width: int, state: dict) -> None:
    """Put a restored width's outcome back where the grids look for it."""
    if (qubits := state.pop("device_qubits", None)) is not None:
        device_node.data["qubits"] = qubits
    known = {f.name for f in fields(Outcome)}
    _sweeps(device_node).setdefault(family, {})[width] = Outcome(
        **{k: v for k, v in state.items() if k in known}
    )


def _report_raw(
    data: dict, fidelity: float, measured: dict, ideal: dict, metrics: dict
) -> None:
    """Record one width's distributions in the run database.

    `data` is what the card shows and is repeated here, so one record describes
    its own run without having to be joined back to anything.
    """
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


def _cells(outcomes: dict[int, Outcome]) -> list[Cell]:
    """One sweep's outcomes as plottable cells, narrowest first.

    Compiled depth is the x coordinate, so a width that never got as far as
    being compiled has no position on the grid and is dropped; it is reported
    on its own card under "Runs", where the reason it stopped is legible.
    """
    return [
        Cell(width=width, depth=outcome.compiled_depth, fidelity=outcome.fidelity)
        for width, outcome in sorted(outcomes.items())
        if outcome.compiled_depth is not None
    ]


def _narrower_over_ceiling(device_node, family: str, width: int) -> int | None:
    """The narrowest width that already blew the depth ceiling, if any did.

    Compiled depth grows monotonically with width, so there is no point
    transpiling a 20-qubit Grover into twelve million layers only to throw it
    away. Only an optimisation: the depth check below catches the case anyway.
    """
    outcomes = _sweeps(device_node).get(family, {})
    hit = [w for w, o in outcomes.items() if w < width and o.over_ceiling]
    return min(hit) if hit else None


def _mirror(qc: QuantumCircuit) -> QuantumCircuit:
    """The circuit followed by its own inverse, measured at the end.

    The composition is the identity, so the ideal distribution is all-zeros
    whatever the width. mqt-bench hands back an already-measured circuit and
    `inverse()` refuses non-unitary instructions, so the measurements come off
    first and go back on at the end.
    """
    qc = qc.remove_final_measurements(inplace=False)

    benchmark = QuantumCircuit(qc.num_qubits)
    benchmark.compose(qc, inplace=True)
    benchmark.compose(qc.inverse(), inplace=True)
    benchmark.measure_all()
    return benchmark


def normalized_fidelity(ideal_counts, observed_counts, num_qubits: int) -> float:
    """Hellinger fidelity rescaled so that a dead, uniform device scores 0.

    Raw Hellinger fidelity has a floor: guessing uniformly already scores
    `Fs_uniform`, which for a spread-out ideal is far from zero. Subtracting
    that floor puts "no information at all" at 0 and the exact distribution
    at 1. Clamped below, because a device can land under uniform.
    """
    Fs = hellinger_fidelity(ideal_counts, observed_counts)

    uniform = {format(i, f"0{num_qubits}b"): 1 for i in range(2**num_qubits)}
    Fs_uniform = hellinger_fidelity(ideal_counts, uniform)

    if Fs_uniform >= 1.0:
        return 0.0

    return max((Fs - Fs_uniform) / (1 - Fs_uniform), 0.0)


def score(results: dict[int, tuple[int, float]]) -> float:
    """PLACEHOLDER -- the formula has not been decided yet.

    Provisionally the widest register whose run cleared SCORE_THRESHOLD, or 0.0
    if none did. It is crude: it throws away the fidelities themselves and does
    not correct for shot noise, which alone drops a flawless 12-qubit QFT to
    0.21. Every call site reads through this function, so swapping the body is
    the whole change.
    """
    cleared = [w for w, (_, fidelity) in results.items() if fidelity >= SCORE_THRESHOLD]
    return float(max(cleared)) if cleared else 0.0


@pytest.fixture(scope="module")
def provider():
    return Eq1QiskitProvider(client=Eq1Client(SERVER_URL, timeout=CLIENT_TIMEOUT_S))


@pytest.fixture(scope="module")
def backends(provider) -> dict[str, object]:
    """One backend per DEVICES entry, keyed by label, or the string that stopped it.

    Hoisted out of the tests because get_backend() rebuilds every backend the
    server knows about. Failures are returned rather than raised so one absent
    device skips its own cards instead of erroring the whole module.
    """
    built: dict[str, object] = {}
    for label, device_id in DEVICES.items():
        try:
            built[label] = provider.get_backend(f"fake-{device_id}")
        except Exception as exc:
            built[label] = f"device unavailable: {type(exc).__name__}: {exc}"
    return built


@eq1val.category("LuxProvide")
@eq1val.subcategory("Volumetric Benchmarks")
@eq1val.subcategory.each("device", list(DEVICES))
@eq1val.subcategory.each("family", list(FAMILIES))
@eq1val.heading("Runs")
def test_width_run(device, family, width, provider, backends, pytestconfig):
    """Run one algorithm at one width on one device and report what it did.

    Nothing is asserted about the fidelity -- where the diagonal stops is the
    result, not a failure. A width that could not be attempted at all skips, so
    the report says "not run" rather than showing a green badge over a
    simulation that never happened; a width that was attempted and threw fails,
    because a broken job is a finding rather than a gap.
    """
    algorithm = FAMILIES[family]
    if width not in widths(pytestconfig, algorithm):
        eq1val.drop()

    task = ALGORITHMS[algorithm][1]
    _root, _suite, device_node, _family_node = eq1val.nesting()
    engine = pytestconfig.getoption("--simulator")
    n_shots = shots(pytestconfig)
    run_id = f"{device}-{algorithm}-{width}q"

    if (state := eq1val.restore(_fingerprint(engine, n_shots))) is not None:
        _rehydrate(device_node, family, width, state)
        return

    data = {
        "task": task,
        "algorithm": family,
        "device": device,
        "device_id": DEVICES[device],
        "simulator": engine,
        "width": width,
        "shots": n_shots,
    }

    def give_up(reason: str, over_ceiling: bool = False, resumable: bool = False):
        """Record a width that was never measured, report it, and skip.

        `resumable` says whether the reason can change between runs: a depth
        ceiling refuses this width forever, while an unreachable device is a
        fact about one afternoon and storing it would carry the outage forward.
        """
        _remember(
            device_node,
            family,
            width,
            Outcome(
                logical_depth=data.get("logical_depth"),
                compiled_depth=data.get("compiled_depth"),
                reason=reason,
                over_ceiling=over_ceiling,
            ),
        )
        eq1val.mark_resumable(resumable)
        eq1val.report_data({**data, "fidelity": "not measured", "outcome": reason})
        pytest.skip(reason)

    backend = backends[device]
    if isinstance(backend, str):
        give_up(backend)

    capacity = getattr(backend, "num_qubits", None)
    device_node.data["qubits"] = capacity if isinstance(capacity, int) else None
    if isinstance(capacity, int) and width > capacity:
        give_up(f"width {width} exceeds the device's {capacity} qubits", resumable=True)

    narrower = _narrower_over_ceiling(device_node, family, width)
    if narrower is not None:
        give_up(
            f"{CEILING_REASON}: {narrower}q already exceeded "
            f"{MAX_COMPILED_DEPTH:,} layers, and depth only grows with width",
            over_ceiling=True,
            resumable=True,
        )

    try:
        circuit = get_benchmark(
            benchmark=algorithm, level=BenchmarkLevel.INDEP, circuit_size=width
        )
    except Exception as exc:
        give_up(
            f"mqt-bench cannot generate {algorithm} at {width}q: {exc}", resumable=True
        )

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
            f"{MAX_COMPILED_DEPTH:,} layers",
            over_ceiling=True,
            resumable=True,
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
        _remember(
            device_node,
            family,
            width,
            Outcome(
                logical_depth=circuit.depth(),
                compiled_depth=compiled.depth(),
                reason=reason,
            ),
        )
        eq1val.report_data({**data, "fidelity": "not measured", "outcome": reason})
        eq1val.fail(f"the run raised {reason}")
        return

    ideal = ideal_distribution(circuit)
    fidelity = normalized_fidelity(ideal, measured, circuit.num_qubits)
    _remember(
        device_node,
        family,
        width,
        Outcome(
            logical_depth=circuit.depth(),
            compiled_depth=compiled.depth(),
            fidelity=fidelity,
        ),
    )
    data["fidelity"] = round(fidelity, 4)
    data["outcome"] = "measured"
    data.update({f"server_{k}_s": round(v, 4) for k, v in bench.metrics.items()})
    eq1val.report_data(data)

    _report_raw(data, fidelity, measured, ideal, bench.metrics)

    fig = bench.plot_graph()
    eq1val.report_plot(fig, name=f"{run_id}-distribution")
    plt.close(fig)


@eq1val.category("LuxProvide")
@eq1val.subcategory("Volumetric Benchmarks")
@eq1val.subcategory.each("device", list(DEVICES))
@eq1val.subcategory.each("family", list(FAMILIES))
@eq1val.heading("Grid & score")
def test_width_grid(device, family, pytestconfig):
    """The volumetric grid and the score for one (device, family) sweep.

    Runs no circuits: it plots what the per-width runs above it left behind on
    the device's report node.
    """
    algorithm = FAMILIES[family]
    task = ALGORITHMS[algorithm][1]
    _root, _suite, device_node, _family_node = eq1val.nesting()
    scan = widths(pytestconfig, algorithm)
    outcomes = _sweeps(device_node).get(family, {})
    measured = {
        width: (outcome.logical_depth, outcome.fidelity)
        for width, outcome in outcomes.items()
        if outcome.fidelity is not None
    }

    if not measured:
        pytest.skip(f"no width of {family} completed on {device}: {NO_RUNS_HINT}")
    if len(outcomes) < len(scan):
        eq1val.report_warning(
            f"partial sweep: {len(outcomes)} of {len(scan)} widths reached this "
            f"grid (missing {sorted(set(scan) - set(outcomes))})"
        )

    fig = plot_volumetric(
        {family: _cells(outcomes)},
        title=f"{task} — {family} width scan on {device}",
        device_qubits=device_node.data.get("qubits"),
        depth_ceiling=MAX_COMPILED_DEPTH,
    )
    eq1val.report_plot(fig, name=f"{task}-{algorithm}-{device}-width-scan")
    plt.close(fig)

    widest = max(measured)
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
            "first_gap_reason": outcomes[gaps[0]].reason
            if gaps and gaps[0] in outcomes
            else "not attempted",
        }
    )


@eq1val.category("LuxProvide")
@eq1val.subcategory("Volumetric Benchmarks")
@eq1val.subcategory.each("device", list(DEVICES))
@eq1val.subcategory(COMBINED_PAGE)
@eq1val.heading("Combined grid")
def test_width_combined(device, pytestconfig):
    """Every algorithm's sweep on one device, on one grid.

    The per-family grids answer "how far did this algorithm get"; this one
    answers how the four compare. The depth axis is logarithmic, so Grover
    sitting several columns right of GHZ at the same width means orders of
    magnitude more layers for the same register.
    """
    _root, _suite, device_node, _page = eq1val.nesting()
    sweeps = _sweeps(device_node)
    series = {
        family: cells
        for family in FAMILIES
        if (cells := _cells(sweeps.get(family, {})))
    }

    if not series:
        pytest.skip(f"no algorithm produced a cell on {device}: {NO_RUNS_HINT}")
    if missing := [family for family in FAMILIES if family not in series]:
        eq1val.report_warning(
            f"partial comparison: {', '.join(missing)} placed no cell on this "
            "grid, so the algorithms shown are not the full set"
        )

    fig = plot_volumetric(
        series,
        title=f"Width scan — all algorithms on {device}",
        device_qubits=device_node.data.get("qubits"),
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
    for family in FAMILIES:
        measured = {
            width: (outcome.compiled_depth, outcome.fidelity)
            for width, outcome in sweeps.get(family, {}).items()
            if outcome.fidelity is not None
        }
        data[f"score_{family}"] = score(measured) if measured else "not measured"
        data[f"max_width_{family}"] = max(measured) if measured else "not measured"
    eq1val.report_data(data)
