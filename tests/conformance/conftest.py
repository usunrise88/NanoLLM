"""Conformance tests: the official OpenAI Python client against `nanollm serve`.

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


@pytest.fixture(scope="session")
def server():
    base = os.environ.get("NANOLLM_BASE_URL")
    if base:
        _wait_ready(base.rstrip("/"), None)
        yield {"base": base.rstrip("/"), "fake": False}
        return
    tmp = tempfile.mkdtemp(prefix="nanollm-conformance-")
    binary = os.environ.get("NANOLLM_BIN")
    if not binary:
        binary = os.path.join(tmp, "nanollm")
        subprocess.run(["go", "build", "-o", binary, "./cmd/nanollm"], cwd=ROOT, check=True)
    port = _free_port()
    cfg = os.path.join(tmp, "nanollm.yaml")
    with open(cfg, "w") as f:
        f.write(
            "server:\n  addr: 127.0.0.1:%d\n  data_dir: %s\n"
            "backends:\n  enabled: [fake]\n"
            "generation:\n  max_tokens_cap: 512\n"
            "log:\n  level: warn\n" % (port, os.path.join(tmp, "data"))
        )
    log = open(os.path.join(tmp, "serve.log"), "w")
    proc = subprocess.Popen([binary, "serve", "-config", cfg], stdout=log, stderr=log)
    base = "http://127.0.0.1:%d" % port
    try:
        _wait_ready(base, proc)
        yield {"base": base, "fake": True}
    finally:
        proc.terminate()
        proc.wait(timeout=30)


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
