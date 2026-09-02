import logging
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

EQ1_SERVER_HOST = "0.0.0.0"
EQ1_SERVER_PORT = 62123
EQ1_SERVER_STARTUP_TIMEOUT = 180

# eq1server declares the qiskit-aer build as a pair of conflicting extras
# (see its pyproject: `eq1rt-cpu` vs `eq1rt-gpu`). Its default `dev` group
# depends on `eq1_server[all]`, which pulls `eq1rt-cpu`, so a bare `uv run`
# resolves to the CPU wheel and *uninstalls* the GPU one if it was there.
# The extra therefore has to be passed explicitly on every run.
EQ1_SERVER_CPU_EXTRA = "sim-cpu"
EQ1_SERVER_GPU_EXTRA = "sim-gpu"

# Tests carrying this marker are skipped by a plain `pytest tests` run and only
# execute when their file is named on the command line (see
# `_explicitly_requested`). For runs too expensive to be part of the default suite.
OPT_IN_MARKER = "opt_in"


def pytest_addoption(parser):
    # Defaults keep a run cheap and CPU-only; override them to scale it up.
    parser.addoption(
        "--simulator",
        default="aer_sv_cpu",
        help=(
            "remote simulator engine to run against (default: aer_sv_cpu; use "
            "aer_sv_gpu or aer_tn_gpu on a CUDA machine — the server is synced "
            "with the matching extra automatically)"
        ),
    )
    parser.addoption(
        "--shots",
        type=int,
        default=None,
        help=(
            "shots per circuit (default: whatever the experiment itself asks "
            "for -- 10000 for the algorithm circuits, 1000 for the width scan). "
            "Noisy runs cost one full trajectory per shot, so lower this well "
            "before scaling --qubits or --max-width up"
        ),
    )
    parser.addoption(
        "--max-width",
        type=int,
        default=None,
        help=(
            "widest register the width scan (tests/test_width_scan.py) "
            "sweeps up to, inclusive (default: 6). Each extra qubit roughly "
            "doubles the work, so the widest few runs dominate the cost"
        ),
    )


def _explicitly_requested(config, item) -> bool:
    """Whether `item` was named on the command line rather than swept up by a directory.

    `pytest tests` collects a directory, `pytest tests/test_width_scan.py`
    (or `...::test_width`) names the file itself -- only the latter counts as
    asking for an opt-in test. Selecting the marker by hand (`-m opt_in`) counts
    too.
    """
    if OPT_IN_MARKER in (config.getoption("-m") or ""):
        return True
    invocation_dir = Path(config.invocation_params.dir)
    for arg in config.args:
        target = (invocation_dir / str(arg).split("::")[0]).resolve()
        if target == item.path:
            return True
    return False


def pytest_collection_modifyitems(config, items):
    skip_opt_in = pytest.mark.skip(
        reason=f"{OPT_IN_MARKER} test: run its file explicitly, or select it with -m {OPT_IN_MARKER}"
    )
    for item in items:
        if OPT_IN_MARKER in item.keywords and not _explicitly_requested(config, item):
            item.add_marker(skip_opt_in)


def _server_path() -> Path:
    """Where the simulator server checkout lives, or should be cloned to.

    `EQ1_SERVER_PATH` names it outright; otherwise it is a sibling of this
    repository, which is where a clone started by this fixture ends up.
    """
    if env_path := os.environ.get("EQ1_SERVER_PATH"):
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "simulator-server"


def _server_branch() -> str:
    """Branch of the simulator server to clone.

    Required rather than defaulted: this file names no repository, so the
    branch has to come from the environment along with the URL.
    """
    branch = os.environ.get("EQ1_SERVER_BRANCH")
    if not branch:
        raise RuntimeError(
            "EQ1_SERVER_BRANCH is not set. Set it alongside EQ1_SERVER_REPO, or "
            "point EQ1_SERVER_URL at an already-running server."
        )
    return branch


def _server_repo() -> str:
    """Clone URL for the simulator server.

    Required rather than defaulted, so the URL lives in the environment and not
    in this file. Either the ssh:// or the https:// form works, depending on how
    the host authenticates to GitHub.
    """
    repo = os.environ.get("EQ1_SERVER_REPO")
    if not repo:
        raise RuntimeError(
            "EQ1_SERVER_REPO is not set. Set it to the simulator server's clone "
            "URL, or point EQ1_SERVER_URL at an already-running server."
        )
    return repo


def _server_extra(simulator: str) -> str:
    """Pick the eq1server extra that matches the simulator engine under test.

    `EQ1_SERVER_EXTRA` overrides the choice, for the case where the server has
    to be built differently from what the engine name suggests.
    """
    if env_extra := os.environ.get("EQ1_SERVER_EXTRA"):
        return env_extra
    if simulator.endswith("_gpu"):
        return EQ1_SERVER_GPU_EXTRA
    return EQ1_SERVER_CPU_EXTRA


def _is_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait_until_ready(host: str, port: int, timeout: float) -> None:
    url = f"http://{host}:{port}/docs"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            logger.info("eq1-server is ready at %s", url)
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    raise RuntimeError(f"eq1-server did not become ready at {url} within {timeout}s")


def _kill_existing_server(host: str, port: int, timeout: float = 10) -> None:
    if not _is_listening(host, port):
        return
    logger.info(
        "eq1-server already listening on %s:%s, killing it before restart", host, port
    )
    subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_listening(host, port):
            logger.info("existing eq1-server stopped")
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"failed to stop existing eq1-server on {host}:{port} within {timeout}s"
    )


@pytest.fixture(scope="session", autouse=True)
def server(pytestconfig):
    # EQ1_SERVER_URL points every test's `SERVER_URL` at an already-running
    # server (local or remote), so skip cloning/building/starting our own.
    if os.environ.get("EQ1_SERVER_URL"):
        logger.info(
            "EQ1_SERVER_URL=%s set, skipping local eq1-server startup",
            os.environ["EQ1_SERVER_URL"],
        )
        yield
        return

    _kill_existing_server(EQ1_SERVER_HOST, EQ1_SERVER_PORT)

    server_path = _server_path()
    branch = _server_branch()
    repo = _server_repo()
    if not server_path.exists():
        logger.info("cloning %s (branch %s) into %s", repo, branch, server_path)
        # eq1server and its eq1client submodule are both private, so surface a
        # credentials problem as a credentials problem rather than as whatever
        # git happens to print several minutes into a clone.
        probe = subprocess.run(
            ["git", "ls-remote", "--exit-code", repo, "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                f"cannot read {repo}: {probe.stderr.strip() or 'unknown error'}\n"
                "This host needs GitHub credentials for the private equal1 "
                "repos. Either add an SSH key it can use (the eq1client "
                "submodule is an ssh:// URL too), or set EQ1_SERVER_REPO to the "
                "https:// form and configure a credential helper."
            )
        subprocess.run(
            [
                "git",
                "clone",
                "--recurse-submodules",
                "--branch",
                branch,
                repo,
                str(server_path),
            ],
            check=True,
        )
        logger.info("clone complete")
    else:
        # An existing checkout is used as-is; it is never fetched or switched,
        # so log which branch it is actually on rather than the one requested.
        current = subprocess.run(
            ["git", "-C", str(server_path), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        logger.info(
            "using existing eq1-server checkout at %s (on branch %s, wanted %s)",
            server_path,
            current or "unknown",
            branch,
        )

    extra = _server_extra(pytestconfig.getoption("--simulator"))
    logger.info("syncing eq1-server dependencies with the %s extra", extra)
    # Done as its own step rather than left to `uv run`: pulling the CUDA
    # wheels can take minutes, which would otherwise eat into the readiness
    # timeout below and surface as a spurious "server did not start".
    subprocess.run(
        ["uv", "sync", "--project", str(server_path), "--extra", extra],
        cwd=server_path,
        check=True,
    )

    log_path = server_path / "server.log"
    logger.info(
        "starting eq1-server on %s:%s with the %s extra (logging to %s)",
        EQ1_SERVER_HOST,
        EQ1_SERVER_PORT,
        extra,
        log_path,
    )
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [
                "uv",
                "run",
                "--project",
                str(server_path),
                "--extra",
                extra,
                "python",
                "-m",
                "eq1_server.simulator_launcher",
                "--port",
                str(EQ1_SERVER_PORT),
            ],
            cwd=server_path,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    logger.info("eq1-server process started (pid %s)", proc.pid)

    try:
        logger.info(
            "waiting up to %ss for eq1-server to become ready",
            EQ1_SERVER_STARTUP_TIMEOUT,
        )
        _wait_until_ready(EQ1_SERVER_HOST, EQ1_SERVER_PORT, EQ1_SERVER_STARTUP_TIMEOUT)
        yield
    finally:
        logger.info("terminating eq1-server process (pid %s)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("eq1-server did not exit in time, killing it")
            proc.kill()
        logger.info("eq1-server process stopped")
