import os
import json
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter

from core.shared import env

def make_session(num_hosts: int, max_workers: int) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=num_hosts,      # um pool por host, sem eviction
        pool_maxsize=max(max_workers // num_hosts, 4) + 4,  # conexões simultâneas por host, com folga
        max_retries=0,                   # deixe o retry na sua própria lógica
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def count_chunks(base_path: str, hash_size_bits: int, hard_limit: int) -> int:
    chunk_size = hash_size_bits // 8
    total = 0
    for file in os.listdir(base_path):
        path = os.path.join(base_path, file)
        if not os.path.isfile(path):
            continue
        total += (os.path.getsize(path) + chunk_size - 1) // chunk_size
        if total >= hard_limit:
            return hard_limit
    return total


def store_chunk(session, urls, key: str, value: str, idx: int):
    url = urls[idx % len(urls)]
    request = {
        "method": "POST",
        "url": f"{url}/put",
        "json": {"key": key, "value": value},
        "timeout_seconds": 5,
    }
    try:
        r = session.post(
            request["url"],
            json=request["json"],
            timeout=request["timeout_seconds"],
        )
        return {
            "idx": idx,
            "key": key,
            "value": value,
            "request": request,
            "status_code": r.status_code,
            "response_body": r.text,
            "error": None,
        }
    except requests.exceptions.RequestException as e:
        return {
            "idx": idx,
            "key": key,
            "value": value,
            "request": request,
            "status_code": None,
            "response_body": None,
            "error": f"{type(e).__name__}: {e}",
        }


def retry_request(session, request: dict):
    method = (request.get("method") or "POST").upper()
    timeout = request.get("timeout_seconds", 5)
    url = request.get("url")
    if not url:
        return {
            "status_code": None,
            "response_body": None,
            "error": "Missing request URL",
        }

    try:
        if method == "GET":
            response = session.get(url, timeout=timeout)
        else:
            response = session.post(url, json=request.get("json"), timeout=timeout)
        return {
            "status_code": response.status_code,
            "response_body": response.text,
            "error": None,
        }
    except requests.exceptions.RequestException as e:
        return {
            "status_code": None,
            "response_body": None,
            "error": f"{type(e).__name__}: {e}",
        }


def retry_failed_requests(
    report_path: str = "fill_failed_requests.json",
    max_workers: int = 64,
    timeout_seconds: int = 5,
    dry_run: bool = False,
    output_report_path: str | None = None,
    show_progress: bool = True,
) -> dict:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    failures = report.get("failures", [])
    retry_requests = []

    for failure in failures:
        request = failure.get("request")
        if not request:
            key = failure.get("key")
            value = failure.get("value")
            if key is None or value is None:
                continue
            request = {
                "method": "POST",
                "url": "",
                "json": {"key": key, "value": value},
                "timeout_seconds": timeout_seconds,
            }
        request_copy = {
            "method": request.get("method", "POST"),
            "url": request.get("url", ""),
            "json": request.get("json"),
            "timeout_seconds": request.get("timeout_seconds", timeout_seconds),
        }
        retry_requests.append({
            "idx": failure.get("idx"),
            "key": failure.get("key"),
            "value": failure.get("value"),
            "request": request_copy,
        })

    total = len(retry_requests)
    if dry_run:
        if show_progress:
            print(f"Dry run: {total} failed requests queued for retry")
        return {
            "submitted": total,
            "attempted": total,
            "requests": [item["request"] for item in retry_requests],
            "failures": [],
            "report_path": None,
        }

    success_count = 0
    failed_count = 0
    retry_results = []
    failed_retry_details = []

    if output_report_path is None:
        output_report_path = os.path.splitext(report_path)[0] + "_retry_report.json"


    with make_session(env.Vars.SERVER_COUNT, max_workers) as session, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(retry_request, session, item["request"]) for item in retry_requests]
        progress = tqdm(total=total, unit="request", desc="retry_failed") if show_progress else None
        for index, future in enumerate(futures):
            result = future.result()
            item = retry_requests[index]
            status_code = result.get("status_code")
            is_success = status_code is not None and 200 <= status_code < 300
            if is_success:
                success_count += 1
            else:
                failed_count += 1
                failed_retry_details.append({
                    "idx": item.get("idx"),
                    "key": item.get("key"),
                    "value": item.get("value"),
                    "request": item.get("request"),
                    "status_code": status_code,
                    "response_body": result.get("response_body"),
                    "error": result.get("error"),
                })
            retry_results.append({
                "idx": item.get("idx"),
                "key": item.get("key"),
                "value": item.get("value"),
                "request": item.get("request"),
                "status_code": status_code,
                "response_body": result.get("response_body"),
                "error": result.get("error"),
            })
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    ok=success_count,
                    failed=failed_count,
                    submitted=total,
                    refresh=False,
                )
        if progress is not None:
            progress.close()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": report_path,
        "total_requested": total,
        "successful": success_count,
        "failed": failed_count,
        "failed_retries": failed_retry_details,
        "results": retry_results,
    }
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {
        "submitted": total,
        "attempted": total,
        "successful": success_count,
        "failed": failed_count,
        "results": retry_results,
        "failures": failed_retry_details,
        "report_path": output_report_path,
    }


def chunk_generator(base_path: str, hash_size_bits: int):
    # pega todos os files de base_path, divide em pedaços de hash_size_bits bits
    chunk_size = hash_size_bits // 8
    for file in os.listdir(base_path):
        path = os.path.join(base_path, file)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            data = f.read()
        total_chunks = (len(data) + chunk_size - 1) // chunk_size
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size].hex(), file, total_chunks


async def store_chunks(
    base_path: str,
    urls: list[str],
    hard_limit: int = 10000,
    max_workers: int = 64,
    window_size: int | None = None,
    hash_size_bits: int = 64,
    failure_report_path: str = "fill_failed_requests.json",
) -> tuple[int, float]:
    if window_size is None:
        window_size = max_workers * 4  

    gen = chunk_generator(base_path, hash_size_bits)
    pending = set()

    count = 0
    finished_count = 0
    ok_count = 0
    failed_count = 0
    start_time = time.time()
    file_stats = {}
    failures = []
    total_chunks = count_chunks(base_path, hash_size_bits, hard_limit)

    with (
        make_session(env.Vars.SERVER_COUNT, max_workers) as session,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
        tqdm(total=total_chunks, unit="chunk", desc="fill") as progress,
    ):

        def submit_next() -> bool:
            nonlocal count
            if count >= hard_limit:
                return False
            try:
                key, file, total_chunks = next(gen)
            except StopIteration:
                return False
            if file not in file_stats:
                file_stats[file] = {
                    "total": total_chunks,
                    "finished": 0,
                    "ok": 0,
                    "failed": 0,
                    "start_time": time.time(),
                }
            fut = executor.submit(store_chunk, session, urls, key, file, count)
            pending.add(fut)
            count += 1
            return True

        # preenche a janela inicial de tasks pendentes
        for _ in range(window_size):
            if not submit_next():
                break

        # consome resultados conforme terminam e repõe a janela
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)

            for fut in done:
                result = fut.result()
                file = result["value"]
                finished_count += 1
                stats = file_stats[file]
                stats["finished"] += 1
                if result["status_code"] == 200:
                    stats["ok"] += 1
                    ok_count += 1
                else:
                    stats["failed"] += 1
                    failed_count += 1
                    failures.append({
                        "idx": result["idx"],
                        "file": file,
                        "key": result["key"],
                        "request": result["request"],
                        "response": {
                            "status_code": result["status_code"],
                            "body": result["response_body"],
                        },
                        "error": result["error"],
                    })
                progress.update(1)
                progress.set_postfix(
                    ok=ok_count,
                    failed=failed_count,
                    submitted=count,
                    file=file,
                    refresh=False,
                )

            for _ in range(len(done)):
                submit_next()

    elapsed = time.time() - start_time
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_path": base_path,
        "hash_size_bits": hash_size_bits,
        "urls": urls,
        "total_submitted": count,
        "total_finished": finished_count,
        "failure_count": len(failures),
        "failures": failures,
    }
    with open(failure_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return count, elapsed
