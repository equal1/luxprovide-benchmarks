# LuxProvide Benchmarks

Three benchmark suites for Equal1 quantum devices, run as pytest sessions and
reported through Equal1's `eq1val` framework as one HTML report with plots.

**Algorithm Circuits** (`tests/test_algorithms.py`) runs GHZ, QFT, QPE and
Grover at the fixed widths the spec names, and draws what each run measured
against the circuit's exact noiseless distribution. Circuits come from
`tests/circuits.py` as stored OpenQASM, so every device sees a byte-identical
circuit and the only thing that varies is its noise. Nothing is scored.

**Volumetric Benchmarks** (`tests/test_width_scan.py`) sweeps every register
width from 2 up for each of the four algorithms, scores each run by Hellinger
fidelity against the exact answer, and places it on a volumetric grid of width
against compiled depth (arXiv:2110.03137). A full sweep runs for hours, and is
restartable with `--eq1-resume`.

**Quantum Volume** (`tests/test_quantum_volume.py`) measures each device's
quantum volume with the heavy output generation test: for every width from 2 up
it pushes random model circuits (arXiv:1811.12926) deeper until the HOG test
fails, then reports `V_Q = 2**max_m min(m, d(m))`. Unlike the other two suites
it asserts something — the run fails below `MIN_EXPECTED_QUANTUM_VOLUME`. Each
width is reported as it finishes, so a sweep cut short still leaves behind
everything it measured.

## Requirements

`eq1val` and `eq1bench` ship in one wheel from Equal1's private validation
repository, and pull `eq1-biskit` and `eq1client` transitively. **Resolving this
project therefore needs credentials for that repository**, and running it also
needs access to a simulator server. Without both, the code here can be read but
not run.

```
uv sync
```

## Configuration

No device or server is named in the source; both come from the environment
(`tests/config.py`).

| Variable | Purpose |
| --- | --- |
| `EQ1_DEVICE_REF`, `EQ1_DEVICE_1`, `EQ1_DEVICE_2` | Backend ids the report's `Device-Ref` / `Device-1` / `Device-2` labels resolve to. The committed fallbacks are placeholders and will not resolve against a server. |
| `EQ1_SERVER_URL` | Points at an already-running simulator server. Set this and nothing is cloned, built or started. |
| `EQ1_SERVER_REPO`, `EQ1_SERVER_BRANCH` | Clone URL and branch for the simulator server, when letting the session start its own. Both are required in that case. |
| `EQ1_SERVER_PATH` | An existing server checkout to use, instead of a sibling `simulator-server/` directory. |

## Running

All three suites are opt-in: a bare `pytest` collects them but skips every one.
Name the file to run it.

```
uv run pytest tests/test_algorithms.py --simulator aer_sv_gpu \
    --eq1-db results/algorithms.jsonl
```

```
uv run pytest tests/test_width_scan.py --simulator aer_sv_gpu \
    --max-width 12 --shots 4000 \
    --eq1-db results/scan.jsonl --eq1-resume
```

```
uv run pytest tests/test_quantum_volume.py --simulator aer_sv_gpu \
    --max-width 5 --shots 100 --replicates 100 \
    --eq1-db results/quantum-volume.jsonl
```
