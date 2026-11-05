"""A local server speaking each provider's streaming protocol.

Tests drive the real SDKs against it, so request shapes and event parsing are
exercised end to end without an API key. Queue replies with :meth:`ProviderStub.queue`.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ProviderStub:
    def __init__(self):
        self.requests: list[dict] = []
        self.replies: list[dict] = []
        self.models = ["model-b", "model-a", "text-embedding-3", "whisper-1"]
        self.events_sent = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _record(self, body):
                stub.requests.append({
                    "path": self.path,
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": body,
                })

            def do_GET(self):
                self._record(None)
                reply = stub.replies.pop(0) if stub.replies else {}
                status = reply.get("status", 200)
                payload = {"object": "list", "has_more": False, "data": [
                    {"id": name, "object": "model", "type": "model",
                     "display_name": name, "created_at": "2026-01-01T00:00:00Z"}
                    for name in stub.models
                ]}
                self._json(status, payload if status == 200 else {"error": {"message": "x"}})

            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                self._record(body)
                reply = stub.replies.pop(0) if stub.replies else {"text": "OK"}
                if reply.get("status", 200) != 200:
                    self._json(reply["status"], {"error": {
                        "type": "error", "message": "stub failure", "code": "stub",
                    }})
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                if self.path.endswith("/chat/completions"):
                    events = _chat_events(reply)
                elif self.path.endswith("/responses"):
                    events = _responses_events(reply)
                else:
                    events = _anthropic_events(reply)
                try:
                    for name, data in events:
                        if name:
                            self.wfile.write(f"event: {name}\n".encode())
                        text = data if isinstance(data, str) else json.dumps(data)
                        self.wfile.write(f"data: {text}\n\n".encode())
                        self.wfile.flush()
                        stub.events_sent += 1
                        time.sleep(reply.get("delay", 0))
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _json(self, status, payload):
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def queue(self, *replies: dict):
        self.replies.extend(replies)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _chat_events(reply):
    def chunk(delta, finish=None):
        return None, {
            "id": "chunk", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    for piece in reply.get("pieces") or [reply.get("text", "")]:
        if piece:
            yield chunk({"role": "assistant", "content": piece})
    for position, call in enumerate(reply.get("tools") or []):
        yield chunk({"tool_calls": [{
            "index": call.get("index", position),
            "id": call.get("id", f"call_{position}"),
            "type": "function",
            "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
        }]})
    yield chunk({}, reply.get("finish", "stop"))
    yield None, "[DONE]"


def _responses_events(reply):
    sequence = 0
    for piece in reply.get("pieces") or [reply.get("text", "")]:
        if piece:
            sequence += 1
            yield "response.output_text.delta", {
                "type": "response.output_text.delta", "delta": piece, "item_id": "m",
                "output_index": 0, "content_index": 0, "sequence_number": sequence,
                "logprobs": [],
            }
    if reply.get("refusal"):
        yield "response.refusal.done", {
            "type": "response.refusal.done", "refusal": "no", "item_id": "m",
            "output_index": 0, "content_index": 0, "sequence_number": sequence + 1,
        }
    output = [
        {
            "type": "function_call", "id": f"fc_{position}", "call_id": f"call_{position}",
            "name": call["name"], "arguments": json.dumps(call["arguments"]),
            "status": "completed",
        }
        for position, call in enumerate(reply.get("tools") or [])
    ]
    name = "response.incomplete" if reply.get("incomplete") else "response.completed"
    yield name, {
        "type": name, "sequence_number": sequence + 2,
        "response": {
            "id": "r", "object": "response", "created_at": 1, "model": "m",
            "output": output, "status": "completed", "parallel_tool_calls": True,
            "tool_choice": "auto", "tools": [],
        },
    }


def _anthropic_events(reply):
    yield "message_start", {"type": "message_start", "message": {
        "id": "msg", "type": "message", "role": "assistant", "content": [], "model": "m",
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }}
    index = 0
    blocks = list(reply.get("blocks") or [])
    if not blocks:
        text = reply.get("text", "")
        if text:
            blocks.append({"type": "text", "text": text})
        blocks.extend({"type": "tool_use", **call} for call in reply.get("tools") or [])
    for block in blocks:
        if block["type"] == "text":
            yield "content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "text", "text": ""},
            }
            yield "content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": block["text"]},
            }
        elif block["type"] == "tool_use":
            yield "content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {
                    "type": "tool_use", "id": f"toolu_{index}", "name": block["name"],
                    "input": {},
                },
            }
            yield "content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block["arguments"]),
                },
            }
        else:
            yield "content_block_start", {
                "type": "content_block_start", "index": index, "content_block": block,
            }
        yield "content_block_stop", {"type": "content_block_stop", "index": index}
        index += 1
    stop_details = None
    if reply.get("stop") == "refusal":
        stop_details = {
            "type": "refusal", "category": reply.get("category"), "explanation": "x",
        }
    yield "message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": reply.get("stop", "end_turn"), "stop_sequence": None,
            "stop_details": stop_details,
        },
        "usage": {"output_tokens": 1},
    }
    yield "message_stop", {"type": "message_stop"}
