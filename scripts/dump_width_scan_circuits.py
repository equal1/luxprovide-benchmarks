"""Dump the transpiled circuits ``test_width_scan`` runs, as OpenQASM 2.

One file per (device, algorithm, width) in ``OUT``, holding the circuit in
exactly the state the sweep hands to the server: mqt-bench's target-independent
circuit, mirrored for QFT, compiled by the device backend's
``default_transpile`` at the sweep's optimization level. The sweep sets
``client_side_transpile = False``, so nothing recompiles after this point --
these are the circuits that run.

Everything deciding *which* circuit a width produces is imported from
``tests/test_width_scan.py`` -- the devices, the algorithms, the
per-algorithm width caps, the optimization level, the depth ceiling and the QFT
mirroring -- so the dump cannot drift from the sweep.

All of a device's widths go into one ``default_transpile`` call, which is what
spreads the work over the machine: qiskit's pass manager maps a list of
circuits across processes itself (``QISKIT_PARALLEL`` / ``QISKIT_NUM_PROCS``,
set below to every core). Nothing here manages processes.

No server needed -- the sweep's devices are eq1-biskit fake backends, built
in-process, and the same objects the sweep gets from the provider. Run it and wait:

    uv run python scripts/dump_width_scan_circuits.py

Widths run from ``MIN_WIDTH`` to ``MAX_WIDTH`` below, per-algorithm caps
applying on top (Grover stops at 7). The widest QFT and QPE transpiles dominate
the runtime.
"""

import os

# Before qiskit is imported anywhere: this is what puts every core to work on
# the batch below. Qiskit defaults to *half* the logical CPUs and, on some
# start methods, to no parallelism at all.
os.environ.setdefault("QISKIT_PARALLEL", "TRUE")
os.environ.setdefault("QISKIT_NUM_PROCS", str(os.cpu_count() or 1))

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# The suite is a pytest module, not an installed package, so its directory has
# to be on the path before it can be imported for its constants.
sys.path.insert(0, str(REPO_ROOT / "tests"))

import test_width_scan as suite  # noqa: E402
from eq1_biskit.fake_backends import get_fake_backend  # noqa: E402
from mqt.bench import BenchmarkLevel, get_benchmark  # noqa: E402
from qiskit import qasm2  # noqa: E402

OUT = REPO_ROOT / "data" / "width-scan-circuits"
MIN_WIDTH = suite.MIN_WIDTH
MAX_WIDTH = 20


def logical_circuit(algorithm: str, width: int):
    """What the sweep builds before compiling: mqt-bench's circuit, mirrored
    for QFT.

    ``suite._mirror`` itself, not a copy: QFT is run as the circuit followed by
    its own inverse, so that composition is the circuit to dump.
    """
    circuit = get_benchmark(
        benchmark=algorithm, level=BenchmarkLevel.INDEP, circuit_size=width
    )
    return suite._mirror(circuit) if algorithm == "qft" else circuit


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    written = 0

    for device, device_id in suite.DEVICES.items():
        backend = get_fake_backend(device_id)
        capacity = getattr(backend, "num_qubits", None)

        names, circuits = [], []
        for algorithm in suite.ALGORITHMS:
            cap = suite.ALGORITHM_MAX_WIDTH.get(algorithm, MAX_WIDTH)
            for width in range(MIN_WIDTH, min(MAX_WIDTH, cap) + 1):
                if isinstance(capacity, int) and width > capacity:
                    continue  # wider than the device; the sweep skips it too
                names.append(f"{device}-{algorithm}-{width}q")
                circuits.append(logical_circuit(algorithm, width))

        print(
            f"{device} ({backend.name}): transpiling {len(circuits)} circuits "
            f"on {os.environ['QISKIT_NUM_PROCS']} processes"
        )
        # One call, one batch: the pass manager parallelises over the list.
        compiled = backend.default_transpile(
            circuits, optimization_level=suite.OPTIMIZATION_LEVEL
        )

        for name, circuit in zip(names, compiled):
            if circuit.depth() > suite.MAX_COMPILED_DEPTH:
                # Past the ceiling the sweep refuses to run, so it is not one
                # of the circuits that run. Never fires at the caps above.
                print(f"  {name}: skipped, {circuit.depth():,} layers")
                continue
            # The native basis carries gates qelib1 does not define (rx90);
            # qiskit emits a `gate` line for each, so the file reparses.
            (OUT / f"{name}.qasm").write_text(qasm2.dumps(circuit) + "\n")
            written += 1
            print(f"  {name}.qasm: depth {circuit.depth()}")

    print(f"\n{written} circuits in {OUT} ({time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
