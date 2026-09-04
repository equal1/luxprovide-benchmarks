"""The benchmark machinery the three suites run on, vendored in full.

These modules are copied verbatim from Equal1's `eq1bench`, which ships inside
the private `eq1val` wheel, at the revision this project pins
(`db660ef92fdff9d8b9c45edf1ae55c0decfcea8a`). They are kept here rather than
imported so that anyone holding this repository can read what a benchmark
actually does -- how a job reaches the device, how the exact reference
distribution is computed, how a fidelity or a heavy-output fraction is scored
-- instead of reaching a private import and stopping there.

That is the whole of the benefit, and it is worth being precise about the rest:

- Vendoring does not make this project runnable without Equal1 credentials.
  The transport underneath is still `eq1_biskit` (`Eq1QiskitProvider`,
  `Eq1BackendBase`, `Simulator`), the tests still report through `eq1val`, and
  both arrive from the same private wheel. What is here is the benchmark logic,
  not a standalone client.
- This is a fork as of that revision. An upstream fix does not reach these
  files on its own; re-copying is a deliberate act. `git log` on this directory
  is the record of when that last happened.

Only two changes were made to the copies: the module docstring naming where
each came from, and `eq1bench.experiment` imports rewritten to the sibling
module. Everything else is upstream's, including its comments.
"""

from .circuit_benchmark import (
    CircuitBenchmark,
    align_to_ideal,
    classical_fidelity,
    ideal_distribution,
    measured_clbits,
    plot_measured_vs_ideal,
)
from .experiment import Counts, Experiment
from .quantum_volume_benchmark import (
    QuantumVolumeBenchmark,
    plot_quantum_volume,
    plot_quantum_volume_existing_results,
)
from .volumetric import Cell, depth_index, plot_volumetric

__all__ = [
    "Cell",
    "CircuitBenchmark",
    "Counts",
    "Experiment",
    "QuantumVolumeBenchmark",
    "align_to_ideal",
    "classical_fidelity",
    "depth_index",
    "ideal_distribution",
    "measured_clbits",
    "plot_measured_vs_ideal",
    "plot_quantum_volume",
    "plot_quantum_volume_existing_results",
    "plot_volumetric",
]
