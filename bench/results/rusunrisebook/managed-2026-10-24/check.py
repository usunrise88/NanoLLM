"""Manual check of `nanollm managed` on a real machine (plan: Проверка).

Starts the binary as an application would (pipes, CREATE_NO_WINDOW, -parent-pid),
follows its events, and records what it sees. Model answers are never stored or
printed: only status codes and token counts.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BIN, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CREATE_NO_WINDOW = 0x08000000
record = {"runs": []}
t0 = time.time()


def stamp():
    return round(time.time() - t0, 1)


def ps(cmd):
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                       creationflags=CREATE_NO_WINDOW)
    return r.stdout.strip()


class Instance:
    def __init__(self, name):
        self.name = name
        self.events = queue.Queue()
        self.log = []
        self.p = subprocess.Popen([BIN, "managed", "-data-dir", DATA, "-parent-pid", str(os.getpid())],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  creationflags=CREATE_NO_WINDOW)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: self.p.stderr.read(), daemon=True).start()
        self.ready = None
        self.windows = []

    def _read(self):
        for line in self.p.stdout:
            ev = json.loads(line)
            shown = {k: v for k, v in ev.items() if k != "token"}
            print(f"[{stamp():7.1f}] {self.name}: {json.dumps(shown, ensure_ascii=False)}", flush=True)
            self.log.append({"t": stamp(), **shown})
            self.events.put(ev)
        self.events.put(None)

    def expect(self, name, timeout, **fields):
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise SystemExit(f"{self.name}: no {name} within {timeout}s")
            try:
                ev = self.events.get(timeout=min(left, 2))
            except queue.Empty:
                self.check_windows()
                continue
            if ev is None:
                raise SystemExit(f"{self.name}: stdout closed waiting for {name}")
            if ev["event"] == name and all(ev.get(k) == v for k, v in fields.items()):
                return ev
            if ev["event"] == "fatal":
                raise SystemExit(f"{self.name}: fatal {ev}")

    def no_event(self, seconds):
        try:
            ev = self.events.get(timeout=seconds)
            return ev
        except queue.Empty:
            return None

    def call(self, method, path, body=None, timeout=600):
        req = urllib.request.Request(self.ready["url"] + path, method=method,
                                     headers={"Authorization": "Bearer " + self.ready["token"],
                                              "Content-Type": "application/json"},
                                     data=None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, dict(r.headers), json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), json.load(e)

    def chat(self):
        start = time.time()
        code, headers, out = self.call("POST", "/v1/chat/completions",
                                       {"messages": [{"role": "user", "content": "Say hello in one word."}],
                                        "max_tokens": 8})
        res = {"status": code, "retry_after": headers.get("Retry-After"), "seconds": round(time.time() - start, 2)}
        if code == 200:
            res["usage"] = out.get("usage")
        else:
            res["code"] = out.get("code")
        return res

    def priority(self):
        return ps(f"(Get-Process -Id {self.p.pid}).PriorityClass")

    def check_windows(self):
        # Every descendant of the instance, with the window it shows, if any.
        out = ps(
            "$all = Get-CimInstance Win32_Process; $seen = @(" + str(self.p.pid) + "); "
            "do { $n = $seen.Count; $seen = @($seen + ($all | Where-Object { $seen -contains $_.ParentProcessId } "
            "| ForEach-Object ProcessId) | Select-Object -Unique) } while ($seen.Count -gt $n); "
            "$seen | Where-Object { $_ -ne " + str(self.p.pid) + " } | ForEach-Object { $p = Get-Process -Id $_ "
            "-ErrorAction SilentlyContinue; if ($p) { '{0} {1} {2}' -f $p.Id, $p.ProcessName, $p.MainWindowHandle } }")
        for line in filter(None, out.splitlines()):
            pid, name, handle = line.split()
            self.windows.append({"t": stamp(), "pid": int(pid), "name": name, "window": handle != "0"})

    def stop(self):
        self.p.stdin.close()
        ev = self.expect("stopping", 30)
        code = self.p.wait(30)
        return {"stopping": ev.get("reason"), "exit": code}


def summary(inst, extra):
    kids = {}
    for w in inst.windows:
        k = kids.setdefault(w["name"], {"seen": 0, "with_window": 0})
        k["seen"] += 1
        k["with_window"] += int(w["window"])
    record["runs"].append({"name": inst.name, "events": inst.log, "children": kids, **extra})


# --- first start: the quiet mode is measured, then a change to optimal ---
a = Instance("first")
a.ready = a.expect("ready", 60)
first = {"ready": {k: v for k, v in a.ready.items() if k != "token"}}
st = a.call("GET", "/api/v1/managed")[2]
first["status_at_start"] = {"chat": st["features"]["chat"], "autotune": st["features"]["autotune"]["state"]}
started = a.expect("tune_started", 60)
first["priority_silent"] = a.priority()
first["chat_during_measurement"] = a.chat()
a.check_windows()
done = a.expect("tune_done", 900)
first["chat_after_silent"] = a.chat()
st = a.call("GET", "/api/v1/managed")[2]
first["autotune_silent"] = st["features"]["autotune"]

code, _, st = a.call("PATCH", "/api/v1/managed/settings", {"mode": "optimal"})
first["patch_optimal"] = {"status": code, "autotune": st["features"]["autotune"]["state"]}
a.expect("tune_started", 60, reason="settings", mode="optimal")
a.check_windows()
done = a.expect("tune_done", 900, mode="optimal")
first["priority_optimal"] = a.priority()
first["chat_after_optimal"] = a.chat()
st = a.call("GET", "/api/v1/managed")[2]
first["autotune_optimal"] = st["features"]["autotune"]
first["stop"] = a.stop()
summary(a, first)

# --- second start: the profile is on disk, nothing is measured ---
b = Instance("second")
b.ready = b.expect("ready", 60)
second = {"ready": {k: v for k, v in b.ready.items() if k != "token"}}
ev = b.no_event(15)
second["event_within_15s"] = None if ev is None else {k: v for k, v in ev.items() if k != "token"}
deadline = time.time() + 120
while time.time() < deadline:
    st = b.call("GET", "/api/v1/managed")[2]
    if st["features"]["chat"]["state"] == "ready":
        break
    time.sleep(1)
second["status"] = {"chat": st["features"]["chat"], "autotune": st["features"]["autotune"]}
second["priority"] = b.priority()
second["chat"] = b.chat()
second["stop"] = b.stop()
summary(b, second)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
print("written", OUT)
