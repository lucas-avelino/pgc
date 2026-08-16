import os
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import requests


def store_chunk(session, urls, key: str, value: str, idx: int):
    url = urls[idx % len(urls)]
    try:
        r = session.post(
            f"{url}/put",
            json={"key": key, "value": value},
            timeout=5,
        )
        return key, value, r.status_code, r.text
    except requests.exceptions.RequestException as e:
        return key, value, None, str(e)


def chunk_generator(base_path: str, hash_size_bits: int):
    # pega todos os files de base_path, divide em pedaços de hash_size_bits bits
    chunk_size = hash_size_bits // 8
    for file in os.listdir(base_path):
        path = os.path.join(base_path, file)
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
) -> tuple[int, float]:
    if window_size is None:
        window_size = max_workers * 4  

    gen = chunk_generator(base_path, hash_size_bits)
    pending = set()

    count = 0
    finished_count = 0
    start_time = time.time()
    file_stats = {}

    with requests.Session() as session, ThreadPoolExecutor(max_workers=max_workers) as executor:

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
                key, file, status_code, text = fut.result()
                finished_count += 1
                stats = file_stats[file]
                stats["finished"] += 1
                if status_code == 200:
                    stats["ok"] += 1
                else:
                    stats["failed"] += 1
                if stats["finished"] == stats["total"]:
                    file_elapsed = time.time() - stats["start_time"]
                    print(
                        f"[{finished_count}/{count}] Stored file {file}: "
                        f"{stats['ok']} ok, {stats['failed']} failed, "
                        f"{file_elapsed:.2f} seconds"
                    )

            for _ in range(len(done)):
                submit_next()

    elapsed = time.time() - start_time
    print(f"Stored {count} chunks in {elapsed:.2f} seconds")
    return count, elapsed
