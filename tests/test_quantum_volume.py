import eq1val
import matplotlib.pyplot as plt
import pytest

from benchmarks import QuantumVolumeBenchmark
from config import DEVICES

pytestmark = pytest.mark.opt_in

SEED = 1337
MIN_EXPECTED_QUANTUM_VOLUME = 2
OPTIMIZATION_LEVEL = 3
DEFAULT_SHOTS = 1
DEFAULT_REPLICATES = 10
DEFAULT_SIZE = 3


def _log2_volume_so_far(achievable_depths: dict[int, int]) -> int:
    """The best `min(width, depth)` among widths finished so far, 0 if none."""
    return max(
        (min(width, depth) for width, depth in achievable_depths.items()), default=0
    )


def _progress_data(experiment: QuantumVolumeBenchmark, **context) -> dict:
    """A report_data() payload built from whatever the sweep has measured so far.

    Reads the same whether the sweep is still running, finished, or stopped
    partway through.
    """
    log2_volume = _log2_volume_so_far(experiment.achievable_depths)
    return {
        **context,
        "quantum_volume": 2**log2_volume,
        "log2_quantum_volume": log2_volume,
        "achievable_depths": experiment.achievable_depths,
        "heavy_output_fractions": {
            f"{width}x{depth}": fraction
            for (width, depth), fraction in experiment.heavy_output_fractions.items()
        },
        "optimization_level": OPTIMIZATION_LEVEL,
    }


@eq1val.category("LuxProvide")
@eq1val.subcategory("Quantum Volume")
@eq1val.subcategory.each("device", list(DEVICES))
def test_quantum_volume(device, provider, backends, pytestconfig):
    """Sweep the device's quantum volume, reporting after every width so a
    sweep cut short still leaves behind everything it measured.
    """
    backend = backends[device]
    if isinstance(backend, str):
        pytest.skip(backend)

    engine = pytestconfig.getoption("--simulator")
    shots = pytestconfig.getoption("--shots") or DEFAULT_SHOTS
    replicates = pytestconfig.getoption("--replicates") or DEFAULT_REPLICATES
    size = pytestconfig.getoption("--max-width") or DEFAULT_SIZE
    context = {
        "device": device,
        "device_id": DEVICES[device],
        "simulator": engine,
        "shots": shots,
        "replicates": replicates,
        "max_num_qubits": size,
        "max_depth": size,
        "seed": SEED,
    }

    experiment = QuantumVolumeBenchmark(
        provider=provider,
        shots=shots,
        max_depth=size,
        max_num_qubits=size,
        replicates=replicates,
        seed=SEED,
    )
    experiment.select_simulator(engine)
    experiment.select_device(device_id=backend.name, is_simulated=True)

    def on_width_done(width: int, depth: int) -> None:
        eq1val.report_data(_progress_data(experiment, **context))

    try:
        measured_quantum_volume = experiment.run(
            on_width_done=on_width_done, optimization_level=OPTIMIZATION_LEVEL
        )
    except (Exception, KeyboardInterrupt) as exc:
        eq1val.fail(f"the sweep raised {type(exc).__name__}: {exc}")
        experiment.log2_quantum_volume = _log2_volume_so_far(
            experiment.achievable_depths
        )
        experiment.quantum_volume = 2**experiment.log2_quantum_volume
        measured_quantum_volume = experiment.quantum_volume
    else:
        eq1val.check(
            measured_quantum_volume >= MIN_EXPECTED_QUANTUM_VOLUME,
            f"Quantum volume: measured={measured_quantum_volume}, "
            f"expected >= {MIN_EXPECTED_QUANTUM_VOLUME}",
        )

    if experiment.heavy_output_fractions:
        fig = experiment.plot_graph()
        eq1val.report_plot(fig, name=f"{device}-quantum-volume")
        plt.close(fig)

    eq1val.report_data(_progress_data(experiment, **context))
