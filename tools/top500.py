"""Calcula os 500 vizinhos mais proximos de cada imagem pela distancia de
Hamming do embedding binario de 1024 bits.

Grava um JSON por embedding em ./data/top500/<nome>.bin.json:

    {
      "name": "000000000009.jpg.bin",
      "k": 500,
      "bits": 1024,
      "neighbors": ["000000093683.jpg.bin", ...],   # ordenado por distancia
      "distances": [212, ...]
    }

Uso:
    python -m tools.top500 ../embeddings-1024 ./data/top500
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

K = 500
BITS = 1024
BYTES = BITS // 8
BLOCK = 1024


def read_signatures(base_path: str):
    names = sorted(
        entry.name
        for entry in os.scandir(base_path)
        if entry.is_file() and entry.name.endswith(".bin") and entry.stat().st_size == BYTES
    )
    signatures = np.empty((len(names), BYTES), dtype=np.uint8)

    def read(i):
        with open(os.path.join(base_path, names[i]), "rb") as f:
            signatures[i] = np.frombuffer(f.read(BYTES), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=64) as executor:
        list(executor.map(read, range(len(names)), chunksize=256))

    return names, signatures


def top_k(signatures):
    # bits como +-1: a distancia de Hamming vira produto interno,
    # d = (BITS - q . x) / 2, o que deixa o BLAS fazer todo o trabalho
    vectors = np.ascontiguousarray(np.unpackbits(signatures, axis=1).astype(np.float32) * 2 - 1)
    transposed = np.ascontiguousarray(vectors.T)

    total = len(vectors)
    neighbors = np.empty((total, K), dtype=np.int32)
    distances = np.empty((total, K), dtype=np.uint16)

    for start in range(0, total, BLOCK):
        end = min(start + BLOCK, total)
        block = ((BITS - vectors[start:end] @ transposed) * 0.5).astype(np.int16)
        block[np.arange(end - start), np.arange(start, end)] = 32767  # ignora a propria imagem

        selected = np.argpartition(block, K, axis=1)[:, :K]
        selected_distances = np.take_along_axis(block, selected, axis=1)
        order = np.argsort(selected_distances, axis=1)

        neighbors[start:end] = np.take_along_axis(selected, order, axis=1)
        distances[start:end] = np.take_along_axis(selected_distances, order, axis=1)
        print(f"  {end}/{total}", flush=True)

    return neighbors, distances


def write_files(output_path: str, names, neighbors, distances):
    os.makedirs(output_path, exist_ok=True)

    def write(i):
        entry = {
            "name": names[i],
            "k": K,
            "bits": BITS,
            "neighbors": [names[j] for j in neighbors[i].tolist()],
            "distances": distances[i].tolist(),
        }
        with open(os.path.join(output_path, f"{names[i]}.json"), "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=64) as executor:
        list(executor.map(write, range(len(names)), chunksize=256))


def main():
    base_path = sys.argv[1] if len(sys.argv) > 1 else "../embeddings-1024"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "./data/top500"

    start = time.time()
    names, signatures = read_signatures(base_path)
    print(f"lidos {len(names)} embeddings em {time.time() - start:.1f}s")

    start = time.time()
    neighbors, distances = top_k(signatures)
    print(f"top-{K} calculado em {time.time() - start:.1f}s")

    start = time.time()
    write_files(output_path, names, neighbors, distances)
    print(f"{len(names)} arquivos escritos em {output_path} em {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
