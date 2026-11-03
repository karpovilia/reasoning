#!/usr/bin/env python3
"""Resume one Web-control branch. No grading or gold answers enter this process.

Each accepted LLM/tool response is committed before updating the state machine.
Exit 75 means recoverable transport failure; exit 2 means a configuration/code
problem requiring intervention. Completed model-format failures are explicit
outputs with an empty answer, rather than silently removed examples.
"""
from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

FINAL_PROMPT = (
    "Do not call any more tools. Using only the evidence already in the conversation, "
    "respond with EXACTLY one JSON object and nothing else in this format: "
    '{"concise_answer":"<short final answer>","comment":"<optional brief explanation>"}.'
)
REPAIR_PROMPT = (
    "Your previous reply did not contain the required final-answer JSON. "
    "Do not use tools. Return one JSON object with a nonempty string concise_answer "
    "and an optional string comment. Give the best answer supported by the evidence."
)


class RetryableTransport(RuntimeError):
    pass


class ConfigurationError(RuntimeError):
    pass


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_messages(messages: list[dict], reasoning_field: str = "reasoning_content") -> list[dict]:
    """Keep reasoning verbatim, but remove archived runner-only metadata."""
    output = []
    for raw in messages:
        role = raw.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ConfigurationError(f"Unsupported message role: {role!r}")
        m = {"role": role, "content": raw.get("content") or ""}
        if role == "assistant":
            reasoning = raw.get("reasoning_content", raw.get("reasoning"))
            if reasoning is not None:
                m[reasoning_field] = reasoning
            if raw.get("tool_calls"):
                m["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {
                        "name": c["function"]["name"],
                        "arguments": c["function"]["arguments"] if isinstance(c["function"].get("arguments"), str)
                        else json.dumps(c["function"].get("arguments", {}), ensure_ascii=False),
                    }} for c in raw["tool_calls"]
                ]
        elif role == "tool":
            m["tool_call_id"] = raw["tool_call_id"]
            if raw.get("name"):
                m["name"] = raw["name"]
        output.append(m)
    return output


def validate_closed_prefix(messages: list[dict]) -> None:
    pending = set()
    for m in messages:
        if m.get("role") == "tool":
            key = m.get("tool_call_id")
            if key not in pending:
                raise ConfigurationError(f"Unmatched tool response {key!r}")
            pending.remove(key)
        else:
            if pending:
                raise ConfigurationError("Message precedes outstanding tool responses")
            calls = m.get("tool_calls") or []
            ids = [c.get("id") for c in calls]
            if len(ids) != len(set(ids)) or any(not i for i in ids):
                raise ConfigurationError("Missing/duplicate tool IDs within one assistant turn")
            pending.update(ids)
    if pending:
        raise ConfigurationError("Prefix ends with unfinished tool calls")


class Journal:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)

    def perform(self, key: str, request: dict, operation):
        done = self.path / (key + ".complete.json")
        pin = digest(request)
        if done.exists():
            old = load(done)
            if old["request_sha256"] != pin:
                raise ConfigurationError(f"Changed request on resume: {key}")
            # A crash can occur between the response receipt and the attempt receipt.
            for p in reversed(sorted(self.path.glob(key + ".attempt*.json"))):
                attempt = load(p)
                if attempt.get("status") == "started" and attempt.get("request_sha256") == pin:
                    attempt.update(status="accepted_recovered", cost_unknown=False,
                                   finished_at=old["accepted_at"], wall_s=old["wall_s"])
                    atomic_json(p, attempt)
                    break
            return old["result"]
        attempts = sorted(self.path.glob(key + ".attempt*.json"))
        # Interrupted calls may have reached the server. Preserve that uncertainty.
        for previous in attempts:
            a = load(previous)
            if a.get("status") == "started":
                a.update(status="interrupted_unconfirmed", cost_unknown=True, reconciled_at=stamp())
                atomic_json(previous, a)
        attempt = self.path / (key + f".attempt{len(attempts):04d}.json")
        entry = {"key": key, "request_sha256": pin, "request": request,
                 "started_at": stamp(), "status": "started"}
        atomic_json(attempt, entry)
        start = time.monotonic()
        try:
            result = operation()
        except BaseException as exc:
            entry.update(status="transport_failed" if isinstance(exc, RetryableTransport) else "failed",
                         error_type=type(exc).__name__, error=str(exc), finished_at=stamp(),
                         wall_s=time.monotonic() - start, cost_unknown=key.startswith("llm_"))
            atomic_json(attempt, entry)
            raise
        entry.update(status="accepted", finished_at=stamp(), wall_s=time.monotonic() - start)
        atomic_json(done, {"key": key, "request_sha256": pin, "request": request,
                           "result": result, "wall_s": entry["wall_s"], "accepted_at": stamp()})
        atomic_json(attempt, entry)
        return result


class HTTPModel:
    def __init__(self, config: dict):
        import httpx
        self.config = config
        self.httpx = httpx
        self.client = httpx.Client(timeout=float(config.get("request_timeout", 900)), trust_env=False)

    def call(self, request: dict, request_id: str) -> dict:
        base = self.config["base_url"].rstrip("/")
        url = base if base.endswith("/chat/completions") else base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        headers = {"Content-Type": "application/json", "X-Request-ID": request_id,
                   "Authorization": "Bearer " + self.config.get("api_key", "not-needed")}
        try:
            response = self.client.post(url, json=request, headers=headers)
        except self.httpx.HTTPError as exc:
            raise RetryableTransport(f"LLM transport {type(exc).__name__}: {exc}") from exc
        if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
            raise RetryableTransport(f"LLM HTTP {response.status_code}: {response.text[:500]}")
        if response.status_code != 200:
            raise ConfigurationError(f"LLM HTTP {response.status_code}: {response.text[:1000]}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RetryableTransport("LLM returned non-JSON HTTP200") from exc
        if not payload.get("choices") or not isinstance(payload["choices"][0].get("message"), dict):
            raise ConfigurationError("Malformed engine response: missing choice/message")
        return payload


def make_wikipedia(config: dict):
    """Reuse original search/page behavior; lift infrastructure failures out of it."""
    import httpx
    from qwen_runner.tools import WikipediaToolExecutor

    class StrictClient:
        def __init__(self):
            self.inner = httpx.Client(base_url=config["tool_base_url"].rstrip("/"),
                                      timeout=float(config.get("tool_timeout", 60)), trust_env=False,
                                      headers={"User-Agent": WikipediaToolExecutor.DEFAULT_USER_AGENT})
            self.history = []

        def fetch(self, request):
            try:
                r = self.inner.send(request)
            except httpx.HTTPError as exc:
                raise RetryableTransport(f"Wikipedia transport: {exc}") from exc
            if r.status_code in {408, 425, 429, 500, 502, 503, 504}:
                raise RetryableTransport(f"Wikipedia HTTP {r.status_code}")
            if r.status_code in {401, 403}:
                raise ConfigurationError(f"Wikipedia access denied: HTTP {r.status_code}")
            return r

        def get(self, *args, **kwargs):
            request = self.inner.build_request("GET", *args, **kwargs)
            cache_dir = config.get("tool_http_cache_dir")
            if not cache_dir:
                return self.fetch(request)
            cache = Path(cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            url = str(request.url)
            key = hashlib.sha256(url.encode()).hexdigest()
            saved = cache / (key + ".response.json")
            with (cache / (key + ".lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                hit = saved.exists()
                if hit:
                    entry = load(saved)
                    body = base64.b64decode(entry["body_base64"])
                    if entry["url"] != url or hashlib.sha256(body).hexdigest() != entry["body_sha256"]:
                        raise ConfigurationError(f"Tool HTTP cache integrity failed: {saved}")
                    # httpx has already decoded content-encoding before recording.
                    headers = {k: v for k, v in entry["headers"].items()
                               if k.lower() not in {"content-encoding", "content-length", "transfer-encoding"}}
                    r = httpx.Response(entry["status_code"], headers=headers, content=body, request=request)
                else:
                    if config.get("tool_http_cache_read_only") or (cache / "FROZEN.json").exists():
                        raise ConfigurationError(f"New tool URL requested after cache freeze: {url}")
                    r = self.fetch(request)
                    if not 200 <= r.status_code < 300:
                        self.history.append({"url": url, "status_code": r.status_code,
                                             "cached": False, "recorded_at": stamp()})
                        return r
                    entry = {"url": url, "status_code": r.status_code, "headers": dict(r.headers),
                             "recorded_at": stamp(), "body_sha256": hashlib.sha256(r.content).hexdigest(),
                             "body_base64": base64.b64encode(r.content).decode("ascii")}
                    atomic_json(saved, entry)
                self.history.append({"url": url, "status_code": entry["status_code"], "cached": hit,
                                     "recorded_at": entry["recorded_at"], "body_sha256": entry["body_sha256"],
                                     "receipt": str(saved)})
                return r

    executor = WikipediaToolExecutor(client=StrictClient())
    executor.base_url = config["tool_base_url"].rstrip("/")
    executor.backend = config.get("wiki_backend", "kiwix")
    executor._kiwix_book = config.get("kiwix_book") or None
    return executor


class Branch:
    def __init__(self, task: dict, config: dict, output: Path, model=None, executor=None):
        from qwen_runner.runner import BenchmarkRunner
        self.task, self.config, self.output = task, config, output
        if any(key in task for key in ("correct_answer", "gold_answer", "answer_aliases", "gold")):
            raise ConfigurationError("Gold-bearing task rejected by inference worker")
        if task["action"] not in {"answer", "continue"}:
            raise ConfigurationError("action must be answer or continue")
        if digest(task["messages"]) != task["prefix_sha256"]:
            raise ConfigurationError("Raw prefix SHA mismatch")
        validate_closed_prefix(task["messages"])
        self.pins = {"task_sha256": digest(task), "config_sha256": digest(config),
                     "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.state_path = output.with_name(output.name + ".state.json")
        self.journal = Journal(output.with_name(output.name + ".journal"))
        self.model = model or HTTPModel(config)
        self.tools = executor or make_wikipedia(config)
        self.parser = BenchmarkRunner(model_client=None, tool_executor=self.tools, tool_choice_mode="auto")
        if self.state_path.exists():
            self.state = load(self.state_path)
            if self.state["pins"] != self.pins:
                raise ConfigurationError("Task/config changed across resume")
        else:
            self.state = {"pins": self.pins, "started_at": stamp(), "started_unix": time.time(),
                "messages": copy.deepcopy(task["messages"]), "responses": [], "tool_results": [],
                "tool_queue": [], "tool_rounds": 0, "final_attempts": 0, "errors": [],
                "phase": "final" if task["action"] == "answer" else "continue",
                "final_prompt_added": False, "wiki_path_cache": {}, "status": "running"}
            self.save()
        if hasattr(self.tools, "_path_cache"):
            self.tools._path_cache = self.state["wiki_path_cache"]

    def save(self):
        self.state["heartbeat"] = stamp()
        atomic_json(self.state_path, self.state)

    def generate(self, final: bool):
        s, c = self.state, self.config
        index = len(s["responses"])
        request = {"model": c["model"], "messages": sanitize_messages(s["messages"], c.get("reasoning_field", "reasoning_content")),
                   "max_tokens": int(c.get("max_new_tokens", 2048)), "temperature": c.get("temperature", 1),
                   "seed": (int(self.task["seed"]) + index) % 2147483647}
        if final:
            if c.get("json_response_format", False):
                request["response_format"] = {"type": "json_object"}
        else:
            request.update(tools=self.tools.openai_tools, tool_choice="auto")
        for key, value in c.get("extra_body", {}).items():
            if key in request:
                raise ConfigurationError(f"extra_body may not override protocol key {key}")
            request[key] = value
        key = f"llm_{index:04d}"
        raw = self.journal.perform(key, request, lambda: self.model.call(request, self.task["task_id"] + ":" + key))
        if c.get("capture_required", True):
            validation = raw.get("capture_validation") or {}
            capture = raw.get("capture") or {}
            count = (raw.get("usage") or {}).get("completion_tokens")
            if not (validation.get("status") == "pass"
                    and validation.get("sampled_tokens") == count == len(raw.get("token_ids") or [])
                    and validation.get("layers") == int(c.get("capture_layers", 40))
                    and validation.get("hidden_size") == int(c.get("capture_hidden_size", 2048))
                    and capture.get("manifest_path") and capture.get("manifest_sha256")):
                raise ConfigurationError(f"Required token/state capture not confirmed for {key}; raw response preserved")
        s["responses"].append(raw)
        m = dict(raw["choices"][0]["message"])
        m.setdefault("role", "assistant")
        m.setdefault("content", "")
        return m, index

    def one_tool(self):
        s = self.state
        pending = s["tool_queue"][0]
        call = pending["call"]
        fn = call["function"]
        arguments = fn.get("arguments") or "{}"
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        req = {"name": fn["name"], "arguments": arguments, "tool_call_id": call["id"]}

        def execute():
            client = getattr(self.tools, "client", None)
            before = len(getattr(client, "history", []))
            try:
                result = self.tools.execute_tool_call(fn["name"], arguments)
            except (ValueError, KeyError, TypeError) as exc:
                result = {"error": "invalid_tool_call", "detail": str(exc)}
            return {"content": result, "wiki_path_cache": getattr(self.tools, "_path_cache", {}),
                    "http_receipts": getattr(client, "history", [])[before:]}

        result = self.journal.perform(pending["journal_key"], req, execute)
        s["wiki_path_cache"] = result["wiki_path_cache"]
        if hasattr(self.tools, "_path_cache"):
            self.tools._path_cache = s["wiki_path_cache"]
        s["messages"].append({"role": "tool", "tool_call_id": call["id"], "name": fn["name"],
                              "content": json.dumps(result["content"], ensure_ascii=False)})
        s["tool_results"].append({**req, "result": result["content"], "journal_key": pending["journal_key"],
                                 "http_receipts": result.get("http_receipts", [])})
        s["tool_queue"].pop(0)
        if not s["tool_queue"]:
            s["tool_rounds"] += 1
            if s["tool_rounds"] >= int(self.config.get("additional_tool_rounds", 4)):
                s["phase"] = "final"
        self.save()

    def finish(self, answer: str | None, status: str, reason: str | None = None):
        s = self.state
        s.update(status=status, answer=answer or "", completed_at=stamp(), phase="done")
        if reason:
            s["errors"].append({"type": "semantic", "reason": reason, "at": stamp()})
        self.save()
        totals = {}
        for r in s["responses"]:
            for k, v in (r.get("usage") or {}).items():
                if isinstance(v, (float, int)) and not isinstance(v, bool):
                    totals[k] = totals.get(k, 0) + v
        attempts = [load(p) for p in self.journal.path.glob("*.attempt*.json")]
        result = {"task_id": self.task["task_id"], "question_id": self.task["question_id"],
            "benchmark": self.task["benchmark"], "action": self.task["action"], "seed": self.task["seed"],
            "status": status, "answer": answer or "", "pins": self.pins,
            "prefix_sha256": self.task["prefix_sha256"], "messages": s["messages"],
            "responses": s["responses"], "usage_totals": totals, "tool_calls": s["tool_results"],
            "additional_tool_rounds": s["tool_rounds"], "wall_s": time.time() - s["started_unix"],
            "active_call_wall_s": sum(x.get("wall_s", 0) for x in attempts),
            "unconfirmed_llm_attempts": sum(bool(x.get("cost_unknown")) for x in attempts),
            "errors": s["errors"], "started_at": s["started_at"], "completed_at": s["completed_at"],
            "journal_directory": str(self.journal.path)}
        atomic_json(self.output, result)
        return result

    def run(self):
        s = self.state
        if s["phase"] == "done":
            if self.output.exists():
                return load(self.output)
            return self.finish(s["answer"], s["status"])
        s["status"] = "running"
        self.save()
        try:
            while True:
                if s["tool_queue"]:
                    self.one_tool()
                    continue
                final = s["phase"] == "final"
                if final and not s["final_prompt_added"]:
                    s["messages"].append({"role": "user", "content": FINAL_PROMPT})
                    s["final_prompt_added"] = True
                    self.save()
                message, index = self.generate(final)
                calls = self.parser._extract_tool_calls(message)
                if calls and not final:
                    ids = set()
                    for j, call in enumerate(calls):
                        if not call.get("id") or call["id"] in ids or call["id"].startswith("textual_tool_call"):
                            call["id"] = f"branch_{index}_{j}"
                        ids.add(call["id"])
                    message["tool_calls"] = calls
                    s["messages"].append(message)
                    s["tool_queue"] = [{"call": c, "journal_key": f"tool_{index:04d}_{j:04d}"} for j, c in enumerate(calls)]
                    self.save()
                    continue
                if calls:
                    message["rejected_tool_calls"] = message.pop("tool_calls", calls)
                    message["content"] = message.get("content") or json.dumps({"rejected_tool_calls": calls}, ensure_ascii=False)
                s["messages"].append(message)
                parsed = self.parser._parse_structured_final_answer(message.get("content") or "")
                if parsed is not None and not calls:
                    return self.finish(parsed["concise_answer"], "complete")
                if final:
                    s["final_attempts"] += 1
                    if s["final_attempts"] >= 1 + int(self.config.get("final_repairs", 2)):
                        return self.finish(None, "semantic_failure", "invalid_final_json_after_bounded_repairs")
                    s["messages"].append({"role": "user", "content": REPAIR_PROMPT})
                else:
                    s["phase"] = "final"
                self.save()
        except RetryableTransport as exc:
            s["status"] = "waiting_transport"
            s["errors"].append({"type": "transport", "error": str(exc), "at": stamp()})
            self.save()
            raise
        except Exception as exc:
            s["status"] = "intervention_required"
            s["errors"].append({"type": type(exc).__name__, "error": str(exc), "at": stamp()})
            self.save()
            raise


def configure_imports(config: dict):
    import importlib.util
    if importlib.util.find_spec('qwen_runner') is None:
        raise ConfigurationError('Install the reasoning-control package with retrieval dependencies')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_name(args.output.name + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Branch already active", file=sys.stderr)
            return 73
        try:
            config = load(args.config)
            configure_imports(config)
            result = Branch(load(args.task), config, args.output).run()
            print(json.dumps({"task_id": result["task_id"], "status": result["status"],
                              "llm_calls": len(result["responses"]), "tool_calls": len(result["tool_calls"])}))
            return 0
        except RetryableTransport as exc:
            print(str(exc), file=sys.stderr)
            return 75
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
