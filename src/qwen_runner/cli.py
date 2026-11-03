from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from pathlib import Path
from time import perf_counter, sleep
from typing import Sequence

import httpx
from tqdm.auto import tqdm

from qwen_runner.client import OpenRouterChatClient
from qwen_runner.datasets import load_dataset_records
from qwen_runner.runner import BenchmarkRunner
from qwen_runner.tools import WikipediaToolExecutor
from qwen_runner.writer import JsonlResultWriter

RETRYABLE_RECORD_EXCEPTIONS = (JSONDecodeError, httpx.RemoteProtocolError)
RETRYABLE_RECORD_ATTEMPTS = 1
RETRYABLE_RECORD_DELAY_SECONDS = 5.0


class _ThreadLocalRuntimeProvider:
    def __init__(self, runtime_factory) -> None:
        self._runtime_factory = runtime_factory
        self._local = threading.local()

    def get(self) -> BenchmarkRunner:
        runtime = getattr(self._local, "runtime", None)
        if runtime is None:
            runtime = self._runtime_factory()
            self._local.runtime = runtime
        return runtime


def env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run normalized benchmarks against Qwen on OpenRouter.")
    parser.add_argument("--dataset", default="all", help="Dataset name from normalize_datasets or 'all'.")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", "qwen/qwen3.6-35b-a3b:nitro"))
    parser.add_argument("--endpoint-url", default=os.getenv("QWEN_ENDPOINT_URL"),
                        help="Override chat endpoint (e.g. http://localhost:PORT/v1 for a local vLLM). "
                             "Defaults to OpenRouter when unset.")
    parser.add_argument("--no-reasoning-field", action="store_true",
                        help="Do not send the OpenRouter 'reasoning' field (use for local vLLM: "
                             "the model reasons in full, no thinking budget).")
    parser.add_argument("--thinking-budget", type=int, default=int(os.getenv("QWEN_THINKING_BUDGET", "100")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("QWEN_MAX_TOKENS", "7000")))
    parser.add_argument("--send-max-tokens", action="store_true",
                        default=env_flag("QWEN_SEND_MAX_TOKENS"),
                        help="Send --max-tokens to the endpoint as the per-turn generation cap. "
                             "Use on a self-hosted vLLM: without it one turn may consume the whole "
                             "free remainder of the context window. Leave off on OpenRouter, where "
                             "the field caps the whole answer and earlier corpora never sent it.")
    parser.add_argument("--workers", type=int, default=int(os.getenv("QWEN_WORKERS", "4")))
    parser.add_argument("--request-timeout", type=float,
                        default=float(os.getenv("QWEN_REQUEST_TIMEOUT", "1200")),
                        help="Per-request HTTP timeout (s); long full-reasoning traces exceed 300.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Canonical results jsonl whose question_ids to skip (leave unset to "
                             "regenerate every row, e.g. when overwriting an inconsistent old corpus).")
    parser.add_argument(
        "--multi-hop-tool-mode",
        choices=["structured", "text"],
        default=os.getenv("QWEN_MULTI_HOP_TOOL_MODE", "structured"),
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=int(os.getenv("QWEN_MAX_TOOL_ROUNDS", "12")),
    )
    parser.add_argument(
        "--tools",
        choices=["wiki", "python"],
        default="wiki",
        help="Tool executor: 'wiki' (legacy multi-hop Wikipedia) or 'python' "
             "(sandboxed code interpreter; tool_choice=auto).",
    )
    parser.add_argument(
        "--tool-datasets",
        default="",
        help="Comma-separated dataset names that get the tool loop "
             "(default: hotpotqa,musique for wiki; REQUIRED for python).",
    )
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (unset = server default).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Sampling seed passed to the endpoint.")
    parser.add_argument(
        "--question-ids-file", type=Path, default=None,
        help="Text file with one question_id per line; only these rows are run "
             "(fixed question set shared by repeated passes and other models).",
    )
    return parser


def load_question_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        candidate = line.strip()
        if candidate and not candidate.startswith("#"):
            ids.append(candidate)
    if not ids:
        raise SystemExit(f"question ids file is empty: {path}")
    return ids


def derive_model_label(model_name: str) -> str:
    label = model_name.rstrip("/").split("/")[-1]
    return label.replace(":", "-")


def find_repository_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "normalize_datasets").exists():
            return candidate
    raise RuntimeError("Could not locate repository root containing normalize_datasets")


def load_dotenv_if_present(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def build_runtime(args: argparse.Namespace) -> BenchmarkRunner:
    api_key = args.api_key
    if not api_key:
        if args.endpoint_url:
            api_key = "EMPTY"  # local vLLM needs no auth; send a non-empty placeholder
        else:
            raise SystemExit("--api-key is required or set OPENROUTER_API_KEY in .env")

    client_kwargs = {}
    if args.endpoint_url:
        client_kwargs["endpoint_url"] = args.endpoint_url
    model_client = OpenRouterChatClient(
        api_key=api_key,
        backup_api_key=os.getenv("OPENROUTER_API_KEY_BACKUP"),
        model=args.model,
        reasoning_max_tokens=args.thinking_budget,
        max_tokens=args.max_tokens,
        send_max_tokens=args.send_max_tokens,
        send_reasoning=not args.no_reasoning_field,
        timeout=args.request_timeout,
        temperature=args.temperature,
        seed=args.seed,
        **client_kwargs,
    )
    tool_datasets = (
        frozenset(name.strip() for name in args.tool_datasets.split(",") if name.strip())
        or None
    )
    if args.tools == "python":
        from qwen_runner.python_tool import PythonToolExecutor

        if tool_datasets is None:
            raise SystemExit("--tools python requires --tool-datasets (e.g. math500)")
        tool_executor = PythonToolExecutor()
        tool_choice_mode = "auto"
    else:
        tool_executor = WikipediaToolExecutor()
        tool_choice_mode = "required_first"
    return BenchmarkRunner(
        model_client=model_client,
        tool_executor=tool_executor,
        max_tool_rounds=args.max_tool_rounds,
        multi_hop_tool_mode=args.multi_hop_tool_mode,
        tool_datasets=tool_datasets,
        tool_choice_mode=tool_choice_mode,
    )


def resolve_dataset_names(dataset_arg: str, datasets_dir: Path) -> list[str]:
    if dataset_arg == "all":
        return sorted(path.stem for path in datasets_dir.glob("*.jsonl"))
    return [dataset_arg]


def default_output_dir(repo_root: Path, model_label: str) -> Path:
    return repo_root / "raw_reasoning_trajectories" / model_label


def run_dataset(
    *,
    dataset_name: str,
    records: list,
    writer_path: Path,
    runtime: BenchmarkRunner | None = None,
    runtime_factory=None,
    workers: int = 1,
    progress_factory=tqdm,
    time_fn=perf_counter,
    sleep_fn=sleep,
) -> dict[str, float | int]:
    if runtime is None and runtime_factory is None:
        raise ValueError("run_dataset requires either runtime or runtime_factory")
    if runtime is not None and runtime_factory is not None:
        raise ValueError("run_dataset accepts runtime or runtime_factory, not both")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    writer = JsonlResultWriter(writer_path)
    pending_records = [record for record in records if record.question_id not in writer.processed_ids]
    progress = progress_factory(pending_records, total=len(pending_records), desc=dataset_name, unit="q")
    progress_writer = _build_progress_writer(progress)
    runtime_factory = runtime_factory or (lambda: runtime)

    processed_count = 0
    cumulative_record_seconds = 0.0
    started_at = time_fn()
    try:
        if workers == 1:
            for record in progress:
                result, elapsed = _execute_record(
                    runtime_factory=runtime_factory,
                    dataset_name=dataset_name,
                    record=record,
                    progress_write=progress_writer,
                    sleep_fn=sleep_fn,
                    time_fn=time_fn,
                )
                writer.write_result(**result)
                processed_count += 1
                cumulative_record_seconds += elapsed
                average_seconds = cumulative_record_seconds / processed_count
                if hasattr(progress, "set_postfix_str"):
                    progress.set_postfix_str(f"last={elapsed:.2f}s avg={average_seconds:.2f}s")
        else:
            runtime_provider = _ThreadLocalRuntimeProvider(runtime_factory)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _execute_record,
                        runtime_factory=runtime_provider.get,
                        dataset_name=dataset_name,
                        record=record,
                        progress_write=progress_writer,
                        sleep_fn=sleep_fn,
                        time_fn=time_fn,
                    ): record.question_id
                    for record in pending_records
                }
                for future in as_completed(futures):
                    result, elapsed = future.result()
                    writer.write_result(**result)
                    processed_count += 1
                    cumulative_record_seconds += elapsed
                    progress.update(1)
                    average_seconds = cumulative_record_seconds / processed_count
                    if hasattr(progress, "set_postfix_str"):
                        progress.set_postfix_str(f"last={elapsed:.2f}s avg={average_seconds:.2f}s")
    finally:
        if hasattr(progress, "close"):
            progress.close()

    total_seconds = time_fn() - started_at
    average_seconds = total_seconds / processed_count if processed_count else 0.0
    return {
        "processed_count": processed_count,
        "total_seconds": round(total_seconds, 2),
        "avg_seconds": round(average_seconds, 2),
        "skipped_count": len(records) - len(pending_records),
    }


def _progress_write(progress: object, message: str) -> None:
    writer = getattr(progress, "write", None)
    if callable(writer):
        writer(message)
        return
    print(message)


def _build_progress_writer(progress: object):
    lock = threading.Lock()

    def progress_writer(message: str) -> None:
        with lock:
            _progress_write(progress, message)

    return progress_writer


def _execute_record(
    *,
    runtime_factory,
    dataset_name: str,
    record: object,
    progress_write,
    sleep_fn,
    time_fn,
) -> tuple[dict, float]:
    started_at = time_fn()
    runtime = runtime_factory()
    try:
        result = _run_record_with_retry(
            runtime=runtime,
            dataset_name=dataset_name,
            record=record,
            progress_write=progress_write,
            sleep_fn=sleep_fn,
        )
    except httpx.ReadTimeout:
        result = _build_failure_result(
            dataset_name=dataset_name,
            record=record,
            failure_reason="endpoint_read_timeout",
        )
    except httpx.HTTPStatusError as exc:
        # 400 = permanent (e.g. context overflow from tool history) — record the
        # failure and move on instead of retrying until walltime (jobs 4182062/3).
        if exc.response.status_code < 500 and exc.response.status_code != 400:
            raise
        result = _build_failure_result(
            dataset_name=dataset_name,
            record=record,
            failure_reason=f"endpoint_http_{exc.response.status_code}",
        )
    elapsed = time_fn() - started_at
    # End-to-end wall time (generation + tool execution) — the one budget metric
    # no alpha-pricing choice can game; consumed by budget_metrics.py.
    result.setdefault("raw_model_work", {})["record_wall_s"] = round(elapsed, 3)
    return result, elapsed


def _build_failure_result(*, dataset_name: str, record: object, failure_reason: str) -> dict:
    return {
        "question_id": record.question_id,
        "question": record.question,
        "correct_answer": record.correct_answer,
        "raw_model_work": {
            "dataset": dataset_name,
            "completed": False,
            "failure_reason": failure_reason,
            "messages": [],
            "responses": [],
        },
    }


def _run_record_with_retry(
    *,
    runtime: BenchmarkRunner,
    dataset_name: str,
    record: object,
    progress_write,
    sleep_fn,
) -> dict:
    for attempt in range(RETRYABLE_RECORD_ATTEMPTS + 1):
        try:
            return runtime.run_record(dataset_name, record)
        except Exception as exc:
            if not _is_retryable_record_exception(exc):
                raise
            if attempt >= RETRYABLE_RECORD_ATTEMPTS:
                raise
            progress_write(
                f"{dataset_name}: transient {type(exc).__name__} on question {record.question_id}. "
                f"Retrying question {record.question_id} in 5 seconds.",
            )
            sleep_fn(RETRYABLE_RECORD_DELAY_SECONDS)

    raise RuntimeError("unreachable")


def _is_retryable_record_exception(exc: Exception) -> bool:
    if isinstance(exc, RETRYABLE_RECORD_EXCEPTIONS):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 400:
        return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = find_repository_root()
    load_dotenv_if_present(repo_root / ".env")

    parser = build_parser()
    args = parser.parse_args(argv)

    datasets_dir = repo_root / "normalize_datasets"
    dataset_names = resolve_dataset_names(args.dataset, datasets_dir)
    model_label = derive_model_label(args.model)
    output_dir = args.output_dir or default_output_dir(repo_root, model_label)

    if args.dry_run:
        for dataset_name in dataset_names:
            dataset_path = datasets_dir / f"{dataset_name}.jsonl"
            record_count = len(load_dataset_records(dataset_path))
            if args.limit is not None:
                record_count = min(record_count, args.limit)
            output_path = output_dir / f"{dataset_name}_{model_label}.jsonl"
            print(f"{dataset_name}: {record_count} records -> {output_path}")
        return 0

    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"invalid shard config: index={args.shard_index} count={args.shard_count}")
    already_done: set[str] = set()
    if args.resume_from is not None and args.resume_from.exists():
        already_done = JsonlResultWriter(args.resume_from).processed_ids

    wanted_ids: set[str] | None = None
    if args.question_ids_file is not None:
        wanted_ids = set(load_question_ids(args.question_ids_file))

    for dataset_name in dataset_names:
        dataset_path = datasets_dir / f"{dataset_name}.jsonl"
        output_path = output_dir / f"{dataset_name}_{model_label}{args.output_suffix}.jsonl"
        records = load_dataset_records(dataset_path)
        if wanted_ids is not None:
            records = [r for r in records if str(r.question_id) in wanted_ids]
            if len(records) != len(wanted_ids):
                raise SystemExit(
                    f"{dataset_name}: matched {len(records)} of {len(wanted_ids)} "
                    f"question ids from {args.question_ids_file}"
                )
        if already_done:
            records = [r for r in records if r.question_id not in already_done]
        if args.shard_count > 1:
            records = records[args.shard_index :: args.shard_count]
        if args.limit is not None:
            records = records[: args.limit]

        stats = run_dataset(
            dataset_name=dataset_name,
            records=records,
            writer_path=output_path,
            runtime_factory=lambda: build_runtime(args),
            workers=args.workers,
        )
        print(
            f"{dataset_name}: wrote results to {output_path} "
            f"(processed={stats['processed_count']}, skipped={stats['skipped_count']}, "
            f"total={stats['total_seconds']:.2f}s, avg={stats['avg_seconds']:.2f}s)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
