"""vllm_health.py against a fake vLLM server: idle, busy, stuck, restarted, without metrics, a proxy and a dead one.
Run: python3 -m unittest test_vllm_health"""
import io
import json
import socket
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import vllm_health as health


class FakeVLLM:
    """Requests running, engine steps (each /metrics read adds `step`), a start time that changes on the read number
    `restart_on`, metrics on or off, and the number of chat requests it was sent."""

    def __init__(self, running=0, step=0, metrics=True, restart_on=None):
        self.running, self.step, self.metrics, self.restart_on = running, step, metrics, restart_on
        self.steps, self.start, self.reads, self.chats = 1000, 1.7e9, 0, 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, code, body, ctype="application/json"):
                data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self.reply(200, {"data": [{"id": "test-model"}]})
                if self.path == "/metrics" and fake.metrics:
                    fake.reads += 1
                    fake.steps += fake.step
                    if fake.reads == fake.restart_on:
                        fake.start += 60
                    labels = '{engine="0",model_name="test-model"}'
                    return self.reply(200, f"# HELP a comment\nprocess_start_time_seconds {fake.start}\n"
                                           f"vllm:num_requests_running{labels} {fake.running}.0\n"
                                           f"vllm:iteration_tokens_total_count{labels} {fake.steps}.0\n"
                                           f"vllm:generation_tokens_total{labels} {fake.steps * 3}.0\n", "text/plain")
                self.reply(404, {"error": "not found"})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                fake.chats += 1
                self.reply(200, {"choices": [{"message": {"content": "ready"}}]})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class LlmHealthTests(unittest.TestCase):
    def run_check(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = health.main(["--gap", "0.05", *argv])
        return code, out.getvalue()

    def server(self, **kw):
        s = FakeVLLM(**kw)
        self.addCleanup(s.close)
        return s

    def test_idle_server_is_asked_for_an_answer(self):
        s = self.server()
        code, out = self.run_check(s.url)
        self.assertEqual((code, s.chats), (0, 1))
        self.assertIn("-> 'ready'", out)

    def test_busy_server_is_left_alone(self):
        s = self.server(running=3, step=5)
        code, out = self.run_check(s.url)
        self.assertEqual((code, s.chats), (0, 0), "no test request for a server that is working")
        self.assertIn("busy: 3 running, 15 tokens", out)

    def test_stuck_server_fails_without_a_test_request(self):
        s = self.server(running=2, step=0)
        code, out = self.run_check(s.url)
        self.assertEqual((code, s.chats), (1, 0))
        self.assertIn("stuck: 2 running", out)
        self.assertEqual(s.reads, 3, "two waits before calling it stuck")

    def test_restart_between_reads_falls_back_to_an_answer(self):
        s = self.server(running=2, step=0, restart_on=2)
        code, out = self.run_check(s.url)
        self.assertEqual((code, s.chats), (0, 1))

    def test_server_without_metrics_is_asked_for_an_answer(self):
        s = self.server(running=4, metrics=False)
        code, out = self.run_check(s.url)
        self.assertEqual((code, s.chats), (0, 1))

    def test_proxy_is_checked_through_models_only(self):
        s = self.server(running=1, step=0)
        code, out = self.run_check("--proxy", s.url)
        self.assertEqual((code, s.chats, s.reads), (0, 0, 0))
        self.assertIn("proxy, models: test-model", out)

    def test_dead_backend_fails(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        alive = self.server()
        code, out = self.run_check(alive.url, f"http://127.0.0.1:{port}/v1")
        self.assertEqual(code, 1)
        self.assertEqual([line[:4] for line in out.splitlines()], ["OK  ", "FAIL"])


if __name__ == "__main__":
    unittest.main()
