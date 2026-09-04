"""The base class both benchmarks in this package are built on.

Owns everything between a circuit and a distribution: which device and
remote simulator a run goes to, submitting a job, polling it to completion,
and normalising the counts that come back.

Vendored from `eq1bench/experiment.py` at eq1val rev db660ef -- see this
package's `__init__.py` for what that means for keeping it current.
"""

from abc import ABC, abstractmethod
from time import sleep
from typing import Optional

from eq1_biskit import Eq1QiskitProvider
from eq1_biskit.backend import Eq1BackendBase, Simulator
from eq1_biskit.config import FAKE_BACKEND_PREFIX
from eq1_biskit.job import Eq1Job
from qiskit import QuantumCircuit
from qiskit.providers import JobStatus
from qiskit.providers.backend import BackendV2
from qiskit.result import Result

type Counts = dict[str, int] | dict[str, float]


class DeviceNotSetError(Exception):
    pass


class SimulatorNotSetError(Exception):
    pass


class CircuitNotGeneratedError(Exception):
    pass


class Experiment(ABC):
    """Experiments base class - defines properties and methods needed to run an experiment
    on an Equal1 QPU or simulator.
    """

    _provider: Eq1QiskitProvider
    _device: Optional[Eq1BackendBase]
    _simulator: Optional[Simulator]
    shots: int | None
    circuits: list[QuantumCircuit] | QuantumCircuit
    client_side_transpile: bool
    transpile_optimization_level: InterruptedError
    submitted_circuits: Optional[list[QuantumCircuit] | QuantumCircuit]
    _active_jobs: list[Eq1Job]
    _results: list[Counts | list[Counts]]

    def __init__(
        self,
        provider: Eq1QiskitProvider,
        shots: int | None = 1000,
        circuits: list[QuantumCircuit] | QuantumCircuit | None = None,
        device: BackendV2 | None = None,
        simulator: Simulator | None = None,
    ) -> None:
        self._provider = provider
        self.shots = shots
        self.circuits = circuits
        self._device = device
        self._simulator = simulator
        self._transpile_target: str | None = None
        self.client_side_transpile = True
        self.transpile_optimization_level = 3
        self.submitted_circuits = None

        self._active_jobs = []
        self._results = []

    def select_device(self, device_id: str, is_simulated: bool) -> None:
        """Select the backend to run on by id.

        If `is_simulated` is True, a simulator must already be set (see
        `select_simulator`) and it must support the requested device's noise
        model; the resolved backend is an `Eq1FakeBackend`. Otherwise a real
        QPU backend is resolved.
        """
        if is_simulated:
            if self._simulator is None:
                raise SimulatorNotSetError(
                    "No simulator set in `self.simulator`. Run `select_simulator()` first or select a real device instead. "
                )

            noise_model_id = device_id.removeprefix(FAKE_BACKEND_PREFIX)
            if not self._simulator.supports(noise_model_id):
                raise ValueError(
                    f"Selected simulator {self._simulator.id} does not support this device's noise model {noise_model_id}. "
                )

        self._device = self._provider.get_backend(device_id)
        self._transpile_target = None

    def select_simulator(self, simulator_id: str) -> None:
        """Set the remote simulator used when running on a simulated backend."""
        self._simulator = self._provider.get_simulator(simulator_id)

    def select_simulator_backend(
        self, simulator_id: str, transpile_target: str | None = None
    ) -> None:
        """Run circuits directly against a bare simulator engine,
        bypassing any noise-model backend entirely -- e.g. for a noiseless
        ("ideal") comparison run.
        """
        backend = self._provider.get_simulator_backend(simulator_id)
        self._device = backend
        self._transpile_target = transpile_target

    def flush_results(self) -> None:
        """Discard all results collected so far."""
        self._results = []

    @abstractmethod
    def generate_circuits(self):
        """Build the experiment's circuits.

        Implementations should either populate `self.circuits` (the common case,
        used by the default `run()` workflow) or return freshly built circuits to
        be passed directly to `start_job()` (used when each job needs different
        circuits, e.g. an optimizer loop).
        """
        pass

    @abstractmethod
    def run(self, **kwargs):
        """Execute the full experiment workflow and return its analysed result.

        kwargs are forwarded to `start_job()` -> backend `run()` as runtime
        options (e.g. `optimization_level`). See the examples below for the
        common workflow shapes.
        """
        pass

    def start_job(
        self, circuits: Optional[list[QuantumCircuit] | QuantumCircuit] = None, **kwargs
    ) -> None:
        """Submit a job to the selected device and queue it for collection.

        `circuits` defaults to `self.circuits`, but an explicit argument takes
        priority so a custom workflow can intervene and submit its own circuits.
        `self.shots` is forwarded to the backend (unless overridden in kwargs);
        any extra kwargs (e.g. `optimization_level`) are passed straight to the
        backend's `run()` as runtime options.

        The job is appended to `self._active_jobs` in submission order, so the
        results collected later line up with the order circuits were submitted.
        """
        circuits = circuits or self.circuits

        if self._device is None:
            raise DeviceNotSetError(
                "No device set in `self.device`. Run `select_device()` first. "
            )
        if circuits is None:
            raise CircuitNotGeneratedError(
                "Circuits were not generated in the experiment. Make sure `generate_circuits()` is properly implemented. "
            )

        if self.client_side_transpile:
            circuits = self._device.default_transpile(
                circuits,
                optimization_level=self.transpile_optimization_level,
            )
        self.submitted_circuits = circuits

        run_options = dict(kwargs)
        if self.shots is not None:
            run_options.setdefault("shots", self.shots)

        if self._simulator is not None:
            job = self._device.run(
                circuits,
                enable_compilation=False,
                simulator_device_id=self._simulator_device_id(),
                **run_options,
            )
        else:
            job = self._device.run(circuits, enable_compilation=False, **run_options)

        self._active_jobs.append(job)

    def _simulator_device_id(self) -> str | None:
        """The remote simulator engine id to submit the job to."""
        if self._simulator is None:
            return None
        return getattr(self._simulator, "id", None) or self._simulator.name

    def collect_results(self, as_probabilities: bool = True) -> None:
        """Drain all queued jobs into `self._results`, preserving submission order.

        Jobs are popped in the order they were submitted and each job's result is
        read through its own instance, so `self._results` stays aligned with the
        order circuits were submitted across all jobs.
        """
        while self._active_jobs:
            job = self._active_jobs.pop(0)

            # 24h poll
            for _ in range(86400):
                status = job.status()
                if status in (JobStatus.QUEUED, JobStatus.RUNNING):
                    sleep(1)
                else:
                    break

            if job.status() == JobStatus.DONE:
                job_result = job.result()
                self._results.append(self._get_counts(job_result, as_probabilities))
            else:
                error_msg = getattr(job, "error_message", lambda: None)()
                raise RuntimeError(
                    f"Job {job.job_id} failed with status {job.status()}: {error_msg}."
                )

    def _flatten_results(self) -> list[Counts]:
        """Flatten self._results into one Counts per circuit, in submission order."""
        return [
            d
            for entry in self._results
            for d in (entry if isinstance(entry, list) else [entry])
        ]

    def _get_counts(
        self, result: Result, as_probabilities: bool = True
    ) -> Counts | list[Counts]:
        counts = result.get_counts()

        if as_probabilities:
            if isinstance(counts, list):
                counts = [self._convert_to_probabilities(c) for c in counts]
            else:
                counts = self._convert_to_probabilities(counts)

        return counts

    @staticmethod
    def _convert_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
        """Convert counts to probabilities."""
        successful_shots = sum(counts.values())
        return {state: count / successful_shots for state, count in counts.items()}
