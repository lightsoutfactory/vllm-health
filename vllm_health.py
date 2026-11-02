#!/usr/bin/env python3
"""vllm-health: check every model backend and the proxy without adding to their load.

    python3 vllm_health.py http://host1:8000/v1 http://host2:8000/v1 --proxy http://proxy:8010/v1 [--model NAME]
                           [--gap 10] [--timeout 60]

A backend that serves vLLM's /metrics is judged by them first. Requests running and the engine stepping through
tokens: healthy and busy, and no test request is sent, so a busy server never makes a check wait in its queue (a check
that times out behind real work is how health checks restart healthy servers). Requests running and no engine step
for two --gap periods in a row: stuck. Nothing running: idle, and only then is a short answer asked for. A backend
without these metrics (another server, an older vLLM) gets the short answer.

A proxy (--proxy) is checked through /models only: its /metrics would come from a different backend on each read,
and the backends behind it are checked directly.

Exit code 0 when all are healthy; 1 otherwise (use it from a systemd timer or cron, and restart or alert on 1).
No dependencies beyond Python 3.9. MIT licence. Part of Factory Floor (https://lightsoutfactory.ai/kits/factory-floor/)."""
import json
import re
import sys
import time
import urllib.request

METRIC = re.compile(r"^(vllm:num_requests_running|vllm:iteration_tokens_total_count|vllm:generation_tokens_total"
                    r"|vllm:prompt_tokens_total|process_start_time_seconds)(?:\{[^}]*\})?\s+(\S+)\s*$", re.M)


def get(url, body=None, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metrics(root):
    """vLLM's counters summed over its engines, or None when the server does not publish them."""
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=10) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    m = {}
    for name, value in METRIC.findall(body):
        m[name] = m.get(name, 0.0) + float(value)
    if "vllm:num_requests_running" not in m:
        return None
    return m


def steps(m):
    """A number that grows whenever the engine works: its steps (prefill chunks count), else tokens in and out."""
    if "vllm:iteration_tokens_total_count" in m:
        return m["vllm:iteration_tokens_total_count"]
    return m.get("vllm:generation_tokens_total", 0.0) + m.get("vllm:prompt_tokens_total", 0.0)


def busy(root, gap):
    """'busy ...' (healthy), 'stuck ...' (not), or None: idle, no metrics, or restarted, so ask for an answer."""
    a = metrics(root)
    if not a or a["vllm:num_requests_running"] <= 0:
        return None
    for _ in range(2):
        time.sleep(gap)
        b = metrics(root)
        if not b or b.get("process_start_time_seconds") != a.get("process_start_time_seconds"):
            return None
        running = int(b["vllm:num_requests_running"])
        if steps(b) > steps(a):
            made = b.get("vllm:generation_tokens_total", 0.0) - a.get("vllm:generation_tokens_total", 0.0)
            return f"busy: {running} running, {made:.0f} tokens in {gap:g} s"
        if running <= 0:
            return None
    return f"stuck: {running} running, the engine made no step in {2 * gap:g} s"


def answer(base, name, timeout):
    t0 = time.time()
    r = get(base + "/chat/completions", {"model": name, "max_tokens": 16, "temperature": 0,
                                         "chat_template_kwargs": {"enable_thinking": False},  # reasoning models: answer
                                         "messages": [{"role": "user", "content": "Reply with the word: ready"}]},
            timeout)
    text = (r["choices"][0]["message"]["content"] or "").strip()[:40]
    return f"{int((time.time() - t0) * 1000)} ms  -> {text!r}"


def check(base, model, gap, timeout, proxy=False):
    """One line for the report, and whether it is healthy."""
    base = base.rstrip("/")
    try:
        ids = [m["id"] for m in get(base + "/models", timeout=10).get("data", [])]
        name = model or (ids[0] if ids else "")
        if proxy:
            return True, f"OK   {base}  proxy, models: {', '.join(ids) or 'none'}"
        state = busy(base[:-3] if base.endswith("/v1") else base, gap)
        if state and state.startswith("stuck"):
            return False, f"FAIL {base}  model={name}  {state}"
        return True, f"OK   {base}  model={name}  {state or answer(base, name, timeout)}"
    except Exception as e:  # noqa: BLE001
        return False, f"FAIL {base}  {type(e).__name__}: {str(e)[:120]}"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):     # nothing to check is not "all healthy"
        print(__doc__)
        return 0 if argv else 2
    opts, urls, proxies = {"--model": None, "--gap": "10", "--timeout": "60"}, [], []
    while argv:
        a = argv.pop(0)
        if a == "--proxy" and argv:
            proxies.append(argv.pop(0))
        elif a in opts and argv:
            opts[a] = argv.pop(0)
        elif a.startswith("-"):
            print(f"unknown option or missing value: {a} (see --help)", file=sys.stderr)
            return 2
        else:
            urls.append(a)
    results = [check(u, opts["--model"], float(opts["--gap"]), float(opts["--timeout"])) for u in urls]
    results += [check(u, opts["--model"], 0, float(opts["--timeout"]), proxy=True) for u in proxies]
    for _, line in results:
        print(line)
    return 0 if all(ok for ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
