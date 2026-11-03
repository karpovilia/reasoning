from __future__ import annotations

import json
import re
from typing import Any

from qwen_runner.datasets import BenchmarkRecord


class BenchmarkRunner:
    TEXT_TOOL_PATTERN = re.compile(
        r"TOOL:(wikipedia_search|wikipedia_page)\s+(\{.*?\})",
        re.DOTALL,
    )
    REASONING_JSON_PATTERN = re.compile(r"(\{[^{}]*?\})", re.DOTALL)
    MAX_CONSECUTIVE_REPAIRS = 2
    MAX_CONSECUTIVE_EXTRACTIONS = 2
    REPAIR_PROMPT_PREFIXES = (
        "You have not finished the question yet.",
        "You did not emit a tool call.",
        "Your previous attempt to call",
    )
    TOOL_INTENT_PATTERN = re.compile(
        r"\b(search|wikipedia_search|wikipedia_page|wikipedia|tool\s+call|"
        r"we\s+(?:need|must|should)\s+to|let'?s\s+(?:search|call|use)|"
        r"i'?ll\s+(?:search|call|use))\b",
        re.IGNORECASE,
    )
    # Narrower intent pattern for non-wiki executors (e.g. python): math answers
    # legitimately contain words like "compute"/"calculate", so only explicit
    # tool-call phrasing counts as intent.
    GENERIC_TOOL_INTENT_PATTERN = re.compile(
        r"\b(tool\s+call|let'?s\s+(?:call|use|run)\s+the\s+tool|"
        r"i'?ll\s+(?:call|use|run)\s+the\s+tool)\b",
        re.IGNORECASE,
    )
    FINAL_ANSWER_EXAMPLE = (
        '{"concise_answer": "B", "comment": "The answer to this question is B) '
        'First mass is bigger than second mass"}'
    )

    DEFAULT_TOOL_DATASETS = frozenset({"hotpotqa", "musique"})

    def __init__(
        self,
        *,
        model_client: Any,
        tool_executor: Any,
        max_tool_rounds: int = 8,
        multi_hop_tool_mode: str = "structured",
        tool_datasets: frozenset[str] | set[str] | None = None,
        tool_choice_mode: str = "required_first",
    ) -> None:
        self.model_client = model_client
        self.tool_executor = tool_executor
        self.max_tool_rounds = max_tool_rounds
        self.multi_hop_tool_mode = multi_hop_tool_mode
        self.tool_datasets = (
            frozenset(tool_datasets) if tool_datasets is not None else self.DEFAULT_TOOL_DATASETS
        )
        # "required_first": first call forces a tool (legacy multi-hop behavior);
        # "auto": model decides — used for optional tools (e.g. python on math).
        self.tool_choice_mode = tool_choice_mode
        tool_names: list[str] = []
        if tool_executor is not None:
            try:
                tool_names = [t["function"]["name"] for t in tool_executor.openai_tools]
            except Exception:
                tool_names = []
        self._is_wiki_executor = "wikipedia_search" in tool_names
        if tool_names:
            self.text_tool_pattern = re.compile(
                r"TOOL:(%s)\s+(\{.*?\})" % "|".join(re.escape(n) for n in tool_names),
                re.DOTALL,
            )
        else:
            self.text_tool_pattern = self.TEXT_TOOL_PATTERN
        self.tool_intent_pattern = (
            self.TOOL_INTENT_PATTERN if self._is_wiki_executor else self.GENERIC_TOOL_INTENT_PATTERN
        )

    def run_record(self, dataset_name: str, record: BenchmarkRecord) -> dict[str, Any]:
        messages = self._build_initial_messages(dataset_name, record)
        responses: list[dict[str, Any]] = []
        tool_capable = dataset_name in self.tool_datasets
        use_structured_tools = tool_capable and self.multi_hop_tool_mode == "structured"
        tools = self.tool_executor.openai_tools if use_structured_tools else None
        failure_reason: str | None = None
        completed = False
        consecutive_repairs = 0
        consecutive_extractions = 0

        force_answer_threshold = max(2, self.max_tool_rounds - 2)

        for _ in range(self.max_tool_rounds):
            tool_message_count = sum(1 for m in messages if m.get("role") == "tool")
            forcing_answer = tool_capable and tool_message_count >= force_answer_threshold
            if forcing_answer:
                extraction = self._extract_final_answer_via_followup(messages)
                if extraction is not None:
                    responses.append(extraction["raw_response"])
                    messages.append(extraction["assistant_message"])
                    if extraction["parsed"]:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": extraction["value"],
                                "source": "structured_final_answer",
                            }
                        )
                        completed = True
                        break
                    if (
                        self._is_final_answer(extraction["assistant_message"])
                        and not self._looks_like_tool_intent(
                            extraction["assistant_message"].get("content") or ""
                        )
                    ):
                        completed = True
                        break
                failure_reason = "answer_extraction_failed"
                break
            response = self.model_client.create_chat_completion(
                messages=messages,
                tools=None if forcing_answer else tools,
                tool_choice=None if forcing_answer else self._tool_choice(messages, use_structured_tools),
            )
            responses.append(response)
            assistant_message = self._normalize_assistant_message(response["choices"][0]["message"])
            messages.append(assistant_message)

            tool_calls = self._extract_tool_calls(assistant_message)
            has_tool_message = any(m.get("role") == "tool" for m in messages)

            if not tool_calls:
                # In auto tool mode a direct answer without any tool call is
                # legitimate; handle it exactly like the no-tools branch so the
                # T/B paired design stays symmetric (no wiki repair crutches).
                if not tool_capable or self.tool_choice_mode == "auto":
                    if self._is_final_answer(assistant_message):
                        self._append_normalized_structured_final_answer(
                            messages,
                            assistant_message,
                            source="structured_final_answer",
                        )
                        completed = True
                        break
                    failure_reason = "no_final_answer"
                    break

                if (
                    has_tool_message
                    and (
                        self._parse_structured_final_answer(
                            assistant_message.get("content") or ""
                        )
                        is not None
                        or (
                            self._is_final_answer(assistant_message)
                            and not self._looks_like_tool_intent(
                                assistant_message.get("content") or ""
                            )
                        )
                    )
                ):
                    self._append_normalized_structured_final_answer(
                        messages,
                        assistant_message,
                        source="structured_final_answer",
                    )
                    completed = True
                    break

                if consecutive_extractions < self.MAX_CONSECUTIVE_EXTRACTIONS:
                    consecutive_extractions += 1
                    extraction = self._extract_intent_via_followup(messages, assistant_message)
                    if extraction is not None:
                        responses.append(extraction["raw_response"])
                        if extraction["kind"] == "tool":
                            tool_calls = [extraction["tool_call"]]
                            messages[-1] = {**assistant_message, "tool_calls": tool_calls}
                        elif extraction["kind"] == "answer" and has_tool_message:
                            messages.append({"role": "assistant", "content": extraction["value"]})
                            completed = True
                            break

                if not tool_calls and not completed:
                    if consecutive_repairs < self.MAX_CONSECUTIVE_REPAIRS:
                        consecutive_repairs += 1
                        messages.append(
                            {
                                "role": "user",
                                "content": self._repair_prompt(use_structured_tools),
                            }
                        )
                        continue
                    failure_reason = "no_final_answer"
                    break

            if tool_calls:
                consecutive_repairs = 0
                consecutive_extractions = 0
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    arguments_json = tool_call["function"].get("arguments", "") or "{}"
                    try:
                        result = self.tool_executor.execute_tool_call(tool_name, arguments_json)
                    except ValueError as exc:
                        result = {"error": str(exc)}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

        if not completed:
            completed = self._is_completed(messages, tool_capable=tool_capable)
        if completed:
            failure_reason = None

        return {
            "question_id": record.question_id,
            "question": record.question,
            "correct_answer": record.correct_answer,
            "raw_model_work": {
                "dataset": dataset_name,
                "tool_mode": self.multi_hop_tool_mode if tool_capable else "none",
                "tool_choice_mode": self.tool_choice_mode if tool_capable else None,
                "completed": completed,
                "failure_reason": failure_reason,
                "messages": messages,
                "responses": responses,
            },
        }

    def _build_initial_messages(self, dataset_name: str, record: BenchmarkRecord) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self._system_prompt(dataset_name)},
            {"role": "user", "content": record.question},
        ]

    def _system_prompt(self, dataset_name: str) -> str:
        if dataset_name in {"hotpotqa", "musique"}:
            if self.multi_hop_tool_mode == "structured":
                return (
                    "You are solving a multi-hop question answering benchmark. "
                    "Call wikipedia_search or wikipedia_page through the structured "
                    "tool-calling interface to gather evidence. "
                    "Issue multiple tool calls in sequence when the question has more than one hop. "
                    "Answer as soon as you gathered enough information from the available evidence. "
                    f"{self._final_answer_format_instruction()}"
                )
            return (
                "You are solving a multi-hop question answering benchmark. "
                "You have two tools: wikipedia_search(query) and wikipedia_page(title). "
                "When you want to call a tool, your visible reply MUST be exactly one line "
                "in this form and nothing else:\n"
                "TOOL:wikipedia_search {\"query\":\"<query string>\"}\n"
                "or\n"
                "TOOL:wikipedia_page {\"title\":\"<exact article title>\"}\n"
                "After you receive a tool message back, you may issue another tool "
                "call the same way, or write the final answer as a JSON object without "
                "the TOOL: prefix. "
                f"{self._final_answer_format_instruction()}"
            )
        tool_hint = ""
        if dataset_name in self.tool_datasets:
            tool_hint = getattr(self.tool_executor, "system_hint", "") or ""
            if tool_hint and not tool_hint.endswith(" "):
                tool_hint += " "
        return (
            "You are solving a benchmark question. "
            "Answer the question directly and keep the final answer concise. "
            f"{tool_hint}{self._final_answer_format_instruction()}"
        )

    def _final_answer_format_instruction(self) -> str:
        return (
            "When you are ready to answer, respond with EXACTLY one JSON object in this format: "
            '{"concise_answer": "<short final answer (str)>", "comment": "<optional brief explanation (str) or null>"}'
            f" Example: {self.FINAL_ANSWER_EXAMPLE}"
        )

    def _normalize_assistant_message(self, message: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(message)
        normalized.setdefault("role", "assistant")
        normalized.setdefault("content", "")
        return normalized

    def _tool_choice(self, messages: list[dict[str, Any]], tool_capable: bool) -> str | None:
        if not tool_capable:
            return None
        if self.tool_choice_mode == "auto":
            return "auto"
        has_tool_message = any(message.get("role") == "tool" for message in messages)
        return "auto" if has_tool_message else "required"

    def _extract_tool_calls(self, assistant_message: dict[str, Any]) -> list[dict[str, Any]]:
        structured_calls = assistant_message.get("tool_calls") or []
        if structured_calls:
            return structured_calls

        content = assistant_message.get("content")
        if isinstance(content, str):
            match = self.text_tool_pattern.search(content)
            if match:
                tool_name, arguments_json = match.groups()
                return [self._build_tool_call(tool_name, arguments_json)]

        return []

    def _build_tool_call(self, tool_name: str, arguments_json: str) -> dict[str, Any]:
        return {
            "id": "textual_tool_call_1",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": arguments_json,
            },
        }

    def _extract_intent_via_followup(
        self,
        messages: list[dict[str, Any]],
        failed_assistant_message: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self._is_wiki_executor:
            return None
        prior_text = self._extract_prior_text(failed_assistant_message)
        if not prior_text:
            return None

        sanitized = [
            message
            for message in messages[:-1]
            if not (
                message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message["content"].startswith(self.REPAIR_PROMPT_PREFIXES)
            )
        ]
        sanitized.append(
            {
                "role": "user",
                "content": (
                    "Your previous attempt to call a Wikipedia tool was not formatted correctly. "
                    f'You wrote: "{prior_text}"\n\n'
                    "Now respond with EXACTLY one JSON object and nothing else. "
                    "If you need to search Wikipedia: "
                    "{\"action\":\"search\",\"value\":\"<query string>\"}\n"
                    "If you want a specific Wikipedia article: "
                    "{\"action\":\"page\",\"value\":\"<exact article title>\"}\n"
                    "If you already have enough evidence to answer the original question: "
                    "{\"action\":\"answer\",\"value\":\"<concise final answer>\"}"
                ),
            }
        )

        response = self.model_client.create_chat_completion(
            messages=sanitized,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content") or ""
        parsed = self._parse_first_json_object(content)
        if not isinstance(parsed, dict):
            return None

        action = parsed.get("action")
        value = parsed.get("value")
        if not isinstance(value, str) or not value.strip():
            return None
        value = value.strip()

        if action == "search":
            return {
                "kind": "tool",
                "raw_response": response,
                "tool_call": self._build_tool_call(
                    "wikipedia_search", json.dumps({"query": value}, ensure_ascii=False)
                ),
            }
        if action == "page":
            return {
                "kind": "tool",
                "raw_response": response,
                "tool_call": self._build_tool_call(
                    "wikipedia_page", json.dumps({"title": value}, ensure_ascii=False)
                ),
            }
        if action == "answer":
            return {"kind": "answer", "raw_response": response, "value": value}
        return None

    def _extract_final_answer_via_followup(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        sanitized = [
            message
            for message in messages
            if not (
                message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message["content"].startswith(self.REPAIR_PROMPT_PREFIXES)
            )
        ]
        sanitized.append(
            {
                "role": "user",
                "content": (
                    "You have already gathered enough evidence from the tool results above. "
                    "Do not call any more tools. Using only the evidence already in the conversation, "
                    "respond with EXACTLY one JSON object and nothing else in this format: "
                    '{"concise_answer":"<short final answer>","comment":"<optional brief explanation>"} '
                    f"Example: {self.FINAL_ANSWER_EXAMPLE}"
                ),
            }
        )

        response = self.model_client.create_chat_completion(
            messages=sanitized,
            response_format={"type": "json_object"},
        )
        assistant_message = self._normalize_assistant_message(response["choices"][0]["message"])
        content = assistant_message.get("content") or ""
        parsed = self._parse_structured_final_answer(content)
        if parsed is None:
            return {
                "raw_response": response,
                "assistant_message": assistant_message,
                "parsed": False,
                "value": None,
            }
        return {
            "raw_response": response,
            "assistant_message": assistant_message,
            "parsed": True,
            "value": parsed["concise_answer"],
        }

    def _extract_prior_text(self, assistant_message: dict[str, Any]) -> str:
        for key in ("reasoning", "content"):
            value = assistant_message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _parse_first_json_object(self, content: str) -> Any:
        if not isinstance(content, str):
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _parse_structured_final_answer(self, content: str) -> dict[str, str] | None:
        parsed = self._parse_first_json_object(content)
        if not isinstance(parsed, dict):
            return None
        concise_answer = parsed.get("concise_answer")
        if not isinstance(concise_answer, str) or not concise_answer.strip():
            return None
        result = {"concise_answer": concise_answer.strip()}
        comment = parsed.get("comment")
        if isinstance(comment, str) and comment.strip():
            result["comment"] = comment.strip()
        return result

    def _append_normalized_structured_final_answer(
        self,
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        *,
        source: str,
    ) -> None:
        parsed = self._parse_structured_final_answer(assistant_message.get("content") or "")
        if parsed is None:
            return
        normalized_message: dict[str, Any] = {
            "role": "assistant",
            "content": parsed["concise_answer"],
            "source": source,
        }
        if "comment" in parsed:
            normalized_message["structured_comment"] = parsed["comment"]
        messages.append(normalized_message)

    def _repair_prompt(self, use_structured_tools: bool) -> str:
        if use_structured_tools:
            if not self._is_wiki_executor:
                return (
                    "You did not emit a tool call and did not provide the final answer. "
                    "Either call an available tool via the structured tool-calling "
                    "interface, or respond with the final answer JSON now."
                )
            return (
                "You did not emit a tool call. Call wikipedia_search or wikipedia_page now "
                "via the structured tool-calling interface."
            )
        return (
            "You have not finished the question yet. "
            "Use one of the available Wikipedia tools now. "
            "Reply with exactly one line in this format: "
            "TOOL:wikipedia_search {\"query\":\"...\"} "
            "or TOOL:wikipedia_page {\"title\":\"...\"}."
        )

    def _is_final_answer(self, assistant_message: dict[str, Any]) -> bool:
        content = assistant_message.get("content")
        if not isinstance(content, str):
            return False
        stripped = content.strip()
        return bool(stripped) and not self.text_tool_pattern.search(stripped)

    def _is_completed(self, messages: list[dict[str, Any]], *, tool_capable: bool = False) -> bool:
        has_tool_message = any(m.get("role") == "tool" for m in messages)
        if tool_capable and self.tool_choice_mode != "auto" and not has_tool_message:
            return False
        last_assistant = None
        for message in reversed(messages):
            if message.get("role") == "assistant":
                last_assistant = message
                break
        if last_assistant is None or not self._is_final_answer(last_assistant):
            return False
        if not tool_capable:
            return True
        content = last_assistant.get("content") or ""
        return not self._looks_like_tool_intent(content)

    def _looks_like_tool_intent(self, content: str) -> bool:
        if not isinstance(content, str):
            return False
        if len(content) > 400:
            return False
        return bool(self.tool_intent_pattern.search(content))
