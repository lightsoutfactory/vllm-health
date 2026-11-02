# vllm-health

A health check for vLLM servers that never waits in their queue. It reads the server's own counters, leaves a busy
server alone, and fails only one that is stuck or down. One Python file, no dependencies.

```sh
python3 vllm_health.py http://gpu1:8000/v1 http://gpu2:8000/v1 --proxy http://gpu2:8010/v1
```

```
OK   http://gpu1:8000/v1  model=main  busy: 11 running, 528 tokens in 5 s
OK   http://gpu2:8000/v1  model=main  3641 ms  -> 'ready'
OK   http://gpu2:8010/v1  proxy, models: main
```

## Install

```sh
pip install git+https://github.com/lightsoutfactory/vllm-health
vllm-health http://127.0.0.1:8000/v1
```

Or copy `vllm_health.py` next to your server: it is one file with no dependencies.

## Why

The usual check sends the model a small chat request and restarts the server when no answer comes in time. On a
busy server that request waits in the queue behind real work, times out, and a healthy server gets restarted. Ours
was restarted eight times in one day that way, and each restart failed the requests in progress, which made the next
restart come sooner. The whole story is in our
[field notes](https://lightsoutfactory.ai/field-notes/health-check-restarted-healthy-model/).

## How it decides

vLLM publishes its counters at `/metrics`. For each backend the check reads them first:

| What the counters show | Verdict | Test request sent? |
|---|---|---|
| requests running, and the engine takes steps between two reads | **busy**, healthy | no |
| requests running, and no engine step for two `--gap` periods in a row | **stuck**: exit 1 | no |
| nothing running | idle: it asks for a 16-token answer, which an idle engine gives at once | yes |
| no vLLM counters (another server, an old vLLM), or the server restarted between reads | asks for the answer | yes |

- Progress is counted in engine steps (`vllm:iteration_tokens_total`), not output tokens. While vLLM reads a long
  prompt, it produces no tokens for a while but still takes steps, so a long prompt does not look like a hang. Where
  that counter is missing, prompt and generated tokens count.
- Counters are summed over all engines of a server (data-parallel vLLM).
- A restart between two reads is caught by `process_start_time_seconds`, so counters of two different processes are
  never compared.
- A proxy in front of several backends (`--proxy`) is checked through `/models` only: each read of its `/metrics`
  could come from a different backend. Check the backends behind it directly.
- The short answer works with any OpenAI-compatible server, so backends without vLLM's counters can be checked too.

## Use it

Exit code 0 when every backend and proxy is healthy, 1 otherwise. Run it from a systemd timer or cron, and act on 1:

```sh
python3 vllm_health.py http://127.0.0.1:8000/v1 || systemctl restart vllm
```

| Option | Default | Meaning |
|---|---|---|
| `--gap` | 10 | seconds between counter reads; a busy server is called stuck after two gaps without a step |
| `--timeout` | 60 | seconds to wait for the short answer from an idle server |
| `--model` | the first served model | the model name for the short answer |
| `--proxy URL` | | a proxy to check through `/models` only; may be given more than once |

A busy server is judged in one `--gap` (two when it looks stuck); an idle one in the time of its short answer.

## Tested

`python3 -m unittest test_vllm_health` runs 7 tests against a fake vLLM server: idle, busy, stuck, restarted between
reads, without metrics, a proxy and a dead backend. Python 3.9 to 3.13.

Live, on two NVIDIA GB10 machines running a recent vLLM built from source, each serving one copy of the same model
behind a least-connection proxy: with 10 to 12 requests running, both servers were reported busy without a test
request; an idle one answered in 3.6 s; the proxy was checked through `/models`.

## The rest of the kit

This is the model check from **[Factory Floor](https://lightsoutfactory.ai/kits/factory-floor/)**, the autopilot
under our own servers: local AI, watchdogs and self-healing operations for your own GPU box.

Also free from us: [bot-gate](https://github.com/lightsoutfactory/bot-gate) (keep pages up for people during a
scraper flood), [cloudflare-free-shield](https://github.com/lightsoutfactory/cloudflare-free-shield) (three Cloudflare
Free-plan rules against scraper floods) and [translation-checks](https://github.com/lightsoutfactory/translation-checks)
(catch translations that would break your app).

MIT licence.
