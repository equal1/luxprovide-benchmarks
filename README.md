# LuxProvide Benchmarks

Two benchmark suites for Equal1 quantum devices, run as pytest sessions and
reported through the [`eq1val`](#requirements) framework as a single HTML report
with plots.

**Algorithm Circuits** (`tests/test_algorithms.py`) runs four standard
algorithms — GHZ, QFT, QPE and Grover — at fixed register widths against each
device, and draws what each run measured against the circuit's exact noiseless
distribution. Circuits are held as OpenQASM source in `circuits.py` rather than
generated, so every device sees the byte-identical circuit and the only thing
that varies is the device's noise. It is a showcase: nothing is scored and there
is no bar to clear.

**Volumetric Benchmarks** (`tests/test_width_scan.py`) sweeps every register
width from 2 up for each of the four algorithms, scores each run by Hellinger
fidelity against the exact answer, and places it on a volumetric grid of width
against compiled depth (the framework is arXiv:2110.03137's). A full sweep runs
for hours, and is restartable.


## Requirements

The framework and the device stack come from Equal1's private validation
repository — `eq1val` and `eq1bench` ship in one wheel from there, and pull
`eq1-biskit` and `eq1client` transitively. **Resolving this project therefore
needs credentials for that repository**, and executing a run additionally needs
access to a simulator server. Without both, the code here can be read but not
run.

```
uv sync
```

## Configuration

No device or server is named in the source; both come from the environment.

| Variable | Purpose |
| --- | --- |
| `EQ1_DEVICE_REF`, `EQ1_DEVICE_1`, `EQ1_DEVICE_2` | Backend ids the report's `Device-Ref` / `Device-1` / `Device-2` labels resolve to. The committed fallbacks are placeholders and will not resolve against a server. |
| `EQ1_SERVER_URL` | Points at an already-running simulator server. Set this and nothing is cloned, built or started. |
| `EQ1_SERVER_REPO`, `EQ1_SERVER_BRANCH` | Clone URL and branch for the simulator server, when letting the session start its own. Both are required in that case. |
| `EQ1_SERVER_PATH` | An existing server checkout to use, instead of a sibling `simulator-server/` directory. |

## Running

Both suites are opt-in: a bare `pytest` collects them but skips every one. Name
the file to run it.

```
uv run pytest tests/test_algorithms.py --simulator aer_sv_gpu \
    --eq1-db results/algorithms.jsonl
```

```
uv run pytest tests/test_width_scan.py --simulator aer_sv_gpu \
    --max-width 12 --shots 4000 \
    --eq1-db results/scan.jsonl --eq1-resume
```
