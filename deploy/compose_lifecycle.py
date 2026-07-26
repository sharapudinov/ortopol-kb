"""docker compose subprocess wrapper for smoke_test.py.

Kept separate from the checks themselves so smoke_test.py's failure output
distinguishes "the stack never came up" from "the stack came up wrong" --
the two are diagnosed completely differently.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

# Fallback only: the repository's own copy, used when a caller doesn't pass
# compose_file explicitly. smoke_test.py always passes the extracted
# artifact's own docker-compose.yml (see artifact_data_dir) so a --artifact/
# --artifact-dir run exercises the package's shipped compose file, init
# scripts and entrypoint (bind-mounted relative to it) rather than this
# checkout's -- the two can otherwise drift silently apart.
COMPOSE_FILE = Path(__file__).resolve().parent / "docker-compose.yml"


def compose(project: str, env: dict, *args: str, compose_file: Path = COMPOSE_FILE) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-f", str(compose_file), "-p", project, *args]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def up(project: str, env: dict, *, compose_file: Path = COMPOSE_FILE) -> subprocess.CompletedProcess:
    return compose(project, env, "up", "-d", compose_file=compose_file)


def down(project: str, env: dict, *, compose_file: Path = COMPOSE_FILE) -> subprocess.CompletedProcess:
    return compose(project, env, "down", "-v", compose_file=compose_file)


def container_id(project: str, env: dict, service: str, *, compose_file: Path = COMPOSE_FILE) -> str | None:
    r = compose(project, env, "ps", "-q", service, compose_file=compose_file)
    cid = r.stdout.strip()
    return cid or None


def health_status(cid: str) -> str:
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Health.Status}}", cid],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def wait_healthy(
    project: str,
    env: dict,
    service: str,
    timeout: float = 240,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    is_healthy=None,
    compose_file: Path = COMPOSE_FILE,
) -> bool:
    """Polls until the service's healthcheck reports "healthy", or times out.

    Postgres + ollama's first boot pulls bge-m3 (~1.2 GB) from the network,
    so the timeout is generous rather than tuned to the fast path.

    clock/sleep/is_healthy are injectable so a unit test can simulate "never
    healthy" and "healthy on the Nth poll" without a real Docker daemon or
    real sleeping; production callers get the real docker compose/inspect
    check via the default is_healthy.
    """
    if is_healthy is None:
        def is_healthy():
            cid = container_id(project, env, service, compose_file=compose_file)
            return bool(cid) and health_status(cid) == "healthy"

    deadline = clock() + timeout
    while clock() < deadline:
        if is_healthy():
            return True
        sleep(3)
    return False


def volumes_remaining(project: str) -> list[str]:
    r = subprocess.run(
        ["docker", "volume", "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"],
        capture_output=True, text=True,
    )
    return [v for v in r.stdout.splitlines() if v.strip()]
