"""Conformance tests: the official OpenAI Python client (and, in test_ollama.py,
the Ollama one) against `nanollm serve`.

By default a server is started on the fake backend, built from this checkout
(or taken from $NANOLLM_BIN). Set NANOLLM_BASE_URL to run the same suite
against a running server with a real model; tests that depend on the fake
backend's scripted answers are then skipped.
"""
import os
import socket
import subprocess
import tempfile
import time
import urllib.request

import pytest
from openai import OpenAI

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The key of the second, authenticated server that test_ollama.py starts.
SECURED_KEY = "sk-conformance"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(base, proc, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("nanollm exited early with %s" % proc.returncode)
        try:
            with urllib.request.urlopen(base + "/readyz", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("nanollm did not become ready at " + base)


_built = None


def _binary():
    """$NANOLLM_BIN, or a binary built once from this checkout."""
    global _built
    binary = os.environ.get("NANOLLM_BIN")
    if binary:
        return binary
    if _built is None:
        _built = os.path.join(tempfile.mkdtemp(prefix="nanollm-bin-"), "nanollm")
        subprocess.run(["go", "build", "-o", _built, "./cmd/nanollm"], cwd=ROOT, check=True)
    return _built


def _start(extra=""):
    """Starts `nanollm serve` on the fake backend with every dialect; returns (base URL, process)."""
    tmp = tempfile.mkdtemp(prefix="nanollm-conformance-")
    port = _free_port()
    cfg = os.path.join(tmp, "nanollm.yaml")
    with open(cfg, "w") as f:
        f.write(
            "server:\n  addr: 127.0.0.1:%d\n  data_dir: %s\n"
            "api:\n  dialects: [openai, native, ollama]\n"
            "backends:\n  enabled: [fake]\n"
            "generation:\n  max_tokens_cap: 512\n"
            "log:\n  level: warn\n%s" % (port, os.path.join(tmp, "data"), extra)
        )
    log = open(os.path.join(tmp, "serve.log"), "w")
    proc = subprocess.Popen([_binary(), "serve", "-config", cfg], stdout=log, stderr=log)
    base = "http://127.0.0.1:%d" % port
    try:
        _wait_ready(base, proc)
    except BaseException:
        proc.terminate()
        raise
    return base, proc


def _stop(proc):
    proc.terminate()
    proc.wait(timeout=30)


@pytest.fixture(scope="session")
def server():
    base = os.environ.get("NANOLLM_BASE_URL")
    if base:
        _wait_ready(base.rstrip("/"), None)
        yield {"base": base.rstrip("/"), "fake": False}
        return
    base, proc = _start()
    try:
        yield {"base": base, "fake": True}
    finally:
        _stop(proc)


@pytest.fixture(scope="session")
def secured_server(server):
    """A second fake server that requires an API key: {"base": …, "key": …}."""
    if not server["fake"]:
        pytest.skip("starts a server of its own, on the fake backend")
    base, proc = _start("auth:\n  mode: apikey\n  keys:\n    - {name: tester, secret: %s}\n" % SECURED_KEY)
    try:
        yield {"base": base, "key": SECURED_KEY}
    finally:
        _stop(proc)


@pytest.fixture(scope="session")
def client(server):
    return OpenAI(base_url=server["base"] + "/v1", api_key=os.environ.get("NANOLLM_API_KEY", "none"))


@pytest.fixture
def fake_only(server):
    if not server["fake"]:
        pytest.skip("depends on the fake backend's scripted answers")


@pytest.fixture(scope="session")
def model(client):
    return client.models.list().data[0].id
