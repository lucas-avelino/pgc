import asyncio
import multiprocessing
import os
import json
import time
import zlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import aiohttp
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

def in_shard(file: str, shard_index: int, shard_count: int) -> bool:
    # divide os arquivos entre processos pelo nome, sem depender da ordem do listdir
    return shard_count <= 1 or zlib.crc32(file.encode("utf-8")) % shard_count == shard_index


def count_chunks(
    base_path: str, hash_size_bits: int, hard_limit: int, shard_index: int = 0, shard_count: int = 1
) -> int:
    chunk_size = hash_size_bits // 8
    total = 0
    for file in os.listdir(base_path):
        if not file.endswith(".bin") or not in_shard(file, shard_index, shard_count):
            continue
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


def chunk_generator(base_path: str, hash_size_bits: int, shard_index: int = 0, shard_count: int = 1):
    # pega os .bin de base_path, divide em pedaços de hash_size_bits bits
    # (a pasta tem index.jsonl e manifest.json junto, que não são embeddings)
    chunk_size = hash_size_bits // 8
    for file in os.listdir(base_path):
        if not file.endswith(".bin") or not in_shard(file, shard_index, shard_count):
            continue
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
    timeout_seconds: float = 5,
    shard_index: int = 0,
    shard_count: int = 1,
    show_progress: bool = True,
    progress_counter=None,
) -> tuple[int, float]:
    """
    Faz o PUT de cada chunk dos .bin de base_path, com max_workers PUTs simultaneos.

    Roda num unico event loop com aiohttp: sem threads disputando o GIL e sem
    esperar sobre uma janela de futures, o que limitava a versao com
    ThreadPoolExecutor + requests a ~1 nucleo de CPU. window_size ficou so por
    compatibilidade -- a quantidade de PUTs em andamento e max_workers.

    shard_index/shard_count fazem o processo cuidar so de uma parte dos arquivos
    (ver store_chunks_parallel); progress_counter e um multiprocessing.Value que
    recebe os chunks terminados, para o processo pai acompanhar o progresso.
    """
    del window_size

    gen = chunk_generator(base_path, hash_size_bits, shard_index, shard_count)
    total_chunks = (
        count_chunks(base_path, hash_size_bits, hard_limit, shard_index, shard_count) if show_progress else None
    )

    count = 0
    finished_count = 0
    ok_count = 0
    failures = []
    last_file = ""

    def next_chunk():
        nonlocal count
        if count >= hard_limit:
            return None
        try:
            key, file, _ = next(gen)
        except StopIteration:
            return None
        idx = count
        count += 1
        return idx, key, file

    async def put(session: aiohttp.ClientSession, idx: int, key: str, file: str) -> None:
        nonlocal finished_count, ok_count, last_file
        request = {
            "method": "POST",
            "url": f"{urls[idx % len(urls)]}/put",
            "json": {"key": key, "value": file},
            "timeout_seconds": timeout_seconds,
        }
        status_code = None
        body = None
        error = None
        try:
            async with session.post(request["url"], json=request["json"]) as response:
                status_code = response.status
                if status_code != 200:
                    body = await response.text()
                else:
                    await response.read()
        except Exception as e:  # timeout, conexao recusada...
            error = f"{type(e).__name__}: {e}"

        finished_count += 1
        last_file = file
        if status_code == 200:
            ok_count += 1
            return
        failures.append({
            "idx": idx,
            "file": file,
            "key": key,
            "request": request,
            "response": {"status_code": status_code, "body": body},
            "error": error,
        })

    async def worker(session: aiohttp.ClientSession) -> None:
        while True:
            item = next_chunk()
            if item is None:
                return
            await put(session, *item)

    shared_shown = 0

    def push_shared() -> None:
        nonlocal shared_shown
        if progress_counter is None or finished_count == shared_shown:
            return
        with progress_counter.get_lock():
            progress_counter.value += finished_count - shared_shown
        shared_shown = finished_count

    async def report_progress(progress: tqdm) -> None:
        # atualizar o tqdm a cada chunk custava CPU do proprio cliente
        shown = 0
        while True:
            await asyncio.sleep(0.5)
            push_shared()
            progress.update(finished_count - shown)
            shown = finished_count
            progress.set_postfix(
                ok=ok_count,
                failed=len(failures),
                submitted=count,
                file=last_file,
                refresh=False,
            )

    connector = aiohttp.TCPConnector(limit=max_workers, limit_per_host=0, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    start_time = time.time()
    with tqdm(total=total_chunks, unit="chunk", desc="fill", disable=not show_progress) as progress:
        reporter = asyncio.create_task(report_progress(progress))
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                await asyncio.gather(*(worker(session) for _ in range(max_workers)))
        finally:
            reporter.cancel()
            try:
                await reporter
            except asyncio.CancelledError:
                pass
        push_shared()
        progress.update(finished_count - progress.n)
        progress.set_postfix(ok=ok_count, failed=len(failures), submitted=count, refresh=True)

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


def _fill_shard(kwargs: dict, progress_counter) -> None:
    # alvo dos processos filhos de store_chunks_parallel: um event loop por processo
    asyncio.run(store_chunks(**kwargs, show_progress=False, progress_counter=progress_counter))


async def store_chunks_parallel(
    base_path: str,
    urls: list[str],
    processes: int = 1,
    hard_limit: int = 10000,
    max_workers: int = 64,
    hash_size_bits: int = 64,
    failure_report_path: str = "fill_failed_requests.json",
    log=print,
    log_interval: float = 30,
) -> tuple[int, float]:
    """
    Divide o fill entre `processes` processos, cada um com max_workers PUTs simultaneos.

    Um event loop Python satura um nucleo; com varios processos o cliente deixa de
    ser o gargalo. Os arquivos sao divididos pelo nome (in_shard) e o hard_limit e
    repartido entre os processos. Os relatorios de falha de cada processo sao
    juntados em failure_report_path, no mesmo formato do store_chunks, e o tempo
    devolvido e o de relogio do fill inteiro.
    """
    if processes <= 1:
        return await store_chunks(
            base_path,
            urls,
            hard_limit=hard_limit,
            max_workers=max_workers,
            hash_size_bits=hash_size_bits,
            failure_report_path=failure_report_path,
        )

    # spawn: o pai tem um event loop rodando, fork copiaria esse estado
    context = multiprocessing.get_context("spawn")
    counter = context.Value("q", 0)
    shard_reports = [f"{failure_report_path}.shard{index}" for index in range(processes)]
    children = []
    for index in range(processes):
        shard_limit = hard_limit // processes + (1 if index < hard_limit % processes else 0)
        kwargs = {
            "base_path": base_path,
            "urls": urls,
            "hard_limit": shard_limit,
            "max_workers": max_workers,
            "hash_size_bits": hash_size_bits,
            "failure_report_path": shard_reports[index],
            "shard_index": index,
            "shard_count": processes,
        }
        children.append(context.Process(target=_fill_shard, args=(kwargs, counter), daemon=True))

    start_time = time.time()
    for child in children:
        child.start()
    last_log = start_time
    try:
        while any(child.is_alive() for child in children):
            await asyncio.sleep(1)
            if time.time() - last_log >= log_interval:
                last_log = time.time()
                done = counter.value
                rate = done / (last_log - start_time)
                log(f"  fill: {done} chunks em {processes} processos ({rate:.0f} chunks/s)")
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join()
    elapsed = time.time() - start_time

    failed = [index for index, child in enumerate(children) if child.exitcode != 0]
    if failed:
        raise RuntimeError(f"processos do fill terminaram com erro: {failed}")

    count = 0
    finished = 0
    failures = []
    for path in shard_reports:
        with open(path, "r", encoding="utf-8") as f:
            shard = json.load(f)
        count += shard["total_submitted"]
        finished += shard["total_finished"]
        failures.extend(shard["failures"])
        os.remove(path)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_path": base_path,
        "hash_size_bits": hash_size_bits,
        "urls": urls,
        "processes": processes,
        "total_submitted": count,
        "total_finished": finished,
        "failure_count": len(failures),
        "failures": failures,
    }
    with open(failure_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return count, elapsed
