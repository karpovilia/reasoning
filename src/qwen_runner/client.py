from __future__ import annotations

import time
from typing import Any

import httpx


class OpenRouterChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        backup_api_key: str | None = None,
        model: str = "qwen/qwen3.6-35b-a3b:nitro",
        reasoning_max_tokens: int = 100,
        max_tokens: int = 7000,
        send_max_tokens: bool = False,
        endpoint_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 300.0,
        client: httpx.Client | None = None,
        max_rate_limit_retries: int = 4,
        sleep_fn=time.sleep,
        send_reasoning: bool = True,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.backup_api_key = backup_api_key
        self.model = model
        self.reasoning_max_tokens = reasoning_max_tokens
        # OpenRouter-specific field; a local OpenAI-compatible vLLM endpoint has no
        # thinking-budget knob, so we drop it there and let the model reason in full.
        self.send_reasoning = send_reasoning
        self.max_tokens = max_tokens
        # Per-turn generation cap. Opt-in on purpose: on OpenRouter the field caps the
        # whole answer and provider runs never sent it, so switching it on globally would
        # silently change corpora already collected there. A self-hosted vLLM has no cap
        # of its own — without this field it hands one turn the entire free remainder of
        # the context window, which is how tool-loop turns ran to 28-43k tokens and the
        # traces died on the request timeout (jobs 4250420/4252704/4253370).
        self.send_max_tokens = send_max_tokens
        self.endpoint_url = endpoint_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self.max_rate_limit_retries = max_rate_limit_retries
        self.sleep_fn = sleep_fn
        # Optional sampling controls (paired T/B designs need temperature=0 + seed).
        self.temperature = temperature
        self.seed = seed

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.send_reasoning:
            payload["reasoning"] = {"max_tokens": self.reasoning_max_tokens}
        if self.send_max_tokens and self.max_tokens is not None:
            # OpenAI-compatible field understood by vLLM: caps tokens generated per turn.
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.seed is not None:
            payload["seed"] = self.seed
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            return self._create_chat_completion_with_api_key(api_key=self.api_key, payload=payload)
        except Exception:
            if not self.backup_api_key or self.backup_api_key == self.api_key:
                raise
            return self._create_chat_completion_with_api_key(
                api_key=self.backup_api_key,
                payload=payload,
            )

    def _create_chat_completion_with_api_key(
        self,
        *,
        api_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        for attempt in range(self.max_rate_limit_retries + 1):
            response = self.client.post(
                self._chat_completions_url(),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()
            if attempt >= self.max_rate_limit_retries:
                response.raise_for_status()
            self.sleep_fn(self._retry_delay_seconds(response, attempt))

        raise RuntimeError("unreachable")

    def _chat_completions_url(self) -> str:
        if self.endpoint_url.endswith("/chat/completions"):
            return self.endpoint_url
        if self.endpoint_url.endswith("/v1"):
            return f"{self.endpoint_url}/chat/completions"
        return f"{self.endpoint_url}/v1/chat/completions"

    def _retry_delay_seconds(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return min(30.0, float(2 ** attempt))
