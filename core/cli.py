
import asyncio
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from core.client.consumer import Client
from core.client.embedding_reader import embedding_file_to_keys
from core.client.utils import write_file, write_json
from core.client.fill import retry_failed_requests, store_chunks
from core.shared import env

async def get_near(client: Client, command: List[str], write_output: bool = True) -> Tuple[Dict[str, any], Dict[str, timedelta]]:
  start_time = datetime.now()
  keys = embedding_file_to_keys(command[1], int(command[2]))
  read_file_time = datetime.now()

  tasks = [client.get_near(k, int(command[3])) for k in keys]
  responses = await asyncio.gather(*tasks)
  request_time = datetime.now()
  if write_output:
    write_json("responses.json", responses)
  
  grouped = {}
  for response in responses:
      key = response['key']
      r = response['r']
      for result in response['results']:
        if result["value"] not in grouped:
          grouped[result["value"]] = {
            "total_distance": 0,
            "amount_of_close_keys": 0,
            "results": []
          }

        grouped[result["value"]]["total_distance"] += result["distance"]
        grouped[result["value"]]["amount_of_close_keys"] += 1

        if len(command) > 4 and command[4] in ["1", "true"]:
          grouped[result["value"]]["results"].append({
              "search_key": key,
              "actual_key": result["key"],
              "distance": result["distance"],
              "r": r
          })

  for key in grouped:
    missed_keys = len(keys) - grouped[key]["amount_of_close_keys"]
    grouped[key]["total_distance"] = grouped[key]["total_distance"] + (missed_keys * int(command[2]))

  #to array
  grouped = [{"value": key, **grouped[key]} for key in grouped]
  grouped.sort(key=lambda x: x["total_distance"])

  if write_output:
    write_json("grouped_responses.json", grouped)
    print(f"Responses written to responses.json")
    print(f"Time taken to read file: {read_file_time - start_time}")
    print(f"Time taken to process requests: {request_time - read_file_time}")
  return grouped, {
    "read_file_time": read_file_time - start_time,
    "request_time": request_time - read_file_time,
  }

DEFAULT_SEED = 42

def get_random_embedding_file_path(path: str = "", amount: int = 1, seed: int = DEFAULT_SEED) -> List[str]:
  # a lista precisa estar ordenada: os.listdir nao garante ordem estavel,
  # e sem isso a mesma seed sortearia imagens diferentes em cada maquina
  files = sorted(f for f in os.listdir(path) if f.endswith(".bin"))
  rng = random.Random(seed)
  return [os.path.join(path, rng.choice(files)) for _ in range(amount)]

RECALL_KS = [1, 10, 50, 100, 500]

_embedding_cache: Dict[str, Dict[str, bytes]] = {}

def load_embeddings(base_path: str) -> Dict[str, bytes]:
  if base_path in _embedding_cache:
    return _embedding_cache[base_path]

  names = sorted(f for f in os.listdir(base_path) if f.endswith(".bin"))
  embeddings: Dict[str, bytes] = {}

  def read(name: str) -> None:
    with open(os.path.join(base_path, name), "rb") as f:
      embeddings[name] = f.read()

  print(f"Carregando {len(names)} embeddings de {base_path} para ranquear por Hamming...")
  start = datetime.now()
  with ThreadPoolExecutor(max_workers=64) as executor:
    list(executor.map(read, names, chunksize=256))
  print(f"Embeddings carregados em {datetime.now() - start}")

  _embedding_cache[base_path] = embeddings
  return embeddings

def hamming(a: bytes, b: bytes) -> int:
  return (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).bit_count()

def rank_by_hamming(grouped: List[Dict[str, any]], query_file: str, base_path: str) -> List[Dict[str, any]]:
  embeddings = load_embeddings(base_path)
  query = embeddings[os.path.basename(query_file)]

  ranked = [
    {**entry, "hamming_distance": hamming(query, embeddings[entry["value"]])}
    for entry in grouped
    if entry["value"] in embeddings
  ]
  ranked.sort(key=lambda entry: entry["hamming_distance"])
  return ranked

def percentile(values: List[float], q: float) -> float:
  ordered = sorted(values)
  return ordered[min(int(len(ordered) * q), len(ordered) - 1)]

def load_top500(ground_truth_path: str, embedding_file: str) -> List[str] | None:
  path = os.path.join(ground_truth_path, f"{os.path.basename(embedding_file)}.json")
  if not os.path.isfile(path):
    return None
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)["neighbors"]

async def recall(client: Client, command: List[str]) -> Dict[str, any]:
  base_path = command[1]
  amount = int(command[2])
  radius = command[3] if len(command) > 3 else "12"
  ground_truth_path = command[4] if len(command) > 4 else "./data/top500"
  seed = int(command[5]) if len(command) > 5 else DEFAULT_SEED

  files = get_random_embedding_file_path(base_path, amount, seed)
  results = []

  for i, file in enumerate(files):
    truth = load_top500(ground_truth_path, file)
    if truth is None:
      continue

    grouped, times = await get_near(client, ["get_near_files", file, "64", radius], write_output=False)
    reranked = rank_by_hamming(grouped, file, base_path)

    # a propria imagem consultada nao esta no ground truth, entao sai da comparacao
    query_name = os.path.basename(file)
    found = [entry["value"] for entry in grouped if entry["value"] != query_name]
    found_by_hamming = [entry["value"] for entry in reranked if entry["value"] != query_name]

    # quantas das 500 imagens verdadeiras a rede devolveu, em qualquer posicao
    # (independe do ranking: os dois ordenam o mesmo conjunto de candidatos)
    top500_found = len(set(found) & set(truth))

    result = {
      "file": query_name,
      "returned": len(found),
      "top500_found": top500_found,
      "coverage": top500_found / len(truth),  # o mesmo numero, normalizado
      "request_time": times["request_time"].total_seconds(),
    }
    for k in RECALL_KS:
      result[f"recall@{k}"] = len(set(found[:k]) & set(truth[:k])) / k
      result[f"hamming_recall@{k}"] = len(set(found_by_hamming[:k]) & set(truth[:k])) / k
    results.append(result)
    print(f"[{i+1}/{len(files)}] {query_name}: {result['top500_found']}/{len(truth)} top500 found, {result['returned']} returned, request_time={result['request_time']:.3f}s")

  if not results:
    print(f"Nenhuma imagem sorteada tem ground truth em {ground_truth_path}")
    return {}

  def stats(field: str) -> Dict[str, float]:
    values = [r[field] for r in results]
    avg = sum(values) / len(values)
    return {
      "avg": avg,
      "std": (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5,
      "p50": percentile(values, 0.50),
      "p90": percentile(values, 0.90),
      "min": min(values),
      "max": max(values),
    }

  fields = ( ["top500_found"] + [f"recall@{k}" for k in RECALL_KS] + [f"hamming_recall@{k}" for k in RECALL_KS] + ["returned", "request_time"])
  summary = {
    "params": {
      "base_path": base_path,
      "amount": len(results),
      "r": int(radius),
      "ground_truth_path": ground_truth_path,
      "seed": seed,
    },
    "coverage": sum(r["coverage"] for r in results) / len(results),  # top500_found normalizado
    **{field: stats(field) for field in fields},
  }

  report_path = f"recall_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
  write_json(report_path, {**summary, "results": results})
  return summary

async def main():
  urls = [f"http://localhost:{i}" for i in range(5000, 5000 + env.Vars.SERVER_COUNT)]
  print(f"Known server count: {len(urls)}")
  client = Client(urls=urls)

  command = input("> ").split(" ")
  while command[0] != "exit":
    match command[0]:
      case "fill":
        base_path = command[1]
        limit = int(command[2]) if len(command) > 2 else 1000000
        hash_size_bits = int(command[3]) if len(command) > 3 else 64
        failure_report_path = command[4] if len(command) > 4 else "fill_failed_requests.json"
        await store_chunks(
          base_path,
          urls,
          hard_limit=limit,
          hash_size_bits=hash_size_bits,
          failure_report_path=failure_report_path,
        )

      case "get_key":
        result = await client.get(command[1])
        print(f"Result for key {command[1]}: {result}")

      case "get_near_key":
        result = await client.get_near(command[1], int(command[2]))
        print(f"Result for key {command[1]} near {command[2]}: {result}")

      case "put":
        await client.put(command[1], command[2])

      case "get_near_files":  # nome do arquivo, tamanho do embedding em bits, raio
        await get_near(client, command)

      case "recall":  # pasta dos embeddings, qtd de imagens, [raio], [pasta do ground truth]
        await recall(client, command)

      case "retry_failed" | "retry_failed_requests":
        report_path = command[1] if len(command) > 1 else "fill_failed_requests.json"
        output_report_path = command[2] if len(command) > 2 else None
        result = retry_failed_requests(report_path, output_report_path=output_report_path)
        print(f"Retry summary: {result['submitted']} submitted, {result['successful']} successful, {result['failed']} failed")
        print(f"Retry failures report: {result['report_path']}")

      case "performance":
        embedding_folder_path = command[1]
        amount_of_searches = int(command[2])
        parallelism = int(command[3]) if len(command) > 3 else 1
        seed = int(command[4]) if len(command) > 4 else DEFAULT_SEED
        results = []

        embedding_file_path = get_random_embedding_file_path(embedding_folder_path, amount_of_searches, seed)
        for i in range(0, amount_of_searches, parallelism):
          print(f"[{i/parallelism}/ {amount_of_searches/parallelism}] Searching for near keys for embedding files: {embedding_file_path[i:i+parallelism]}")
          tasks = [get_near(client, ["get_near_files", k, "64", "12"]) for k in embedding_file_path[i:i+parallelism]]
          r = await asyncio.gather(*tasks)
          results.extend([(res[1], len(res[0])) for res in r])


        write_json(f"performance_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json", {
          "params":
            {
              "amount_of_searches": amount_of_searches,
              "parallelism": parallelism,
              "seed": seed,
            },
          "qtd_results_total": sum([r[1] for r in results]),
          "avg_results_per_search": sum([r[1] for r in results]) / len(results),
          "max_results_per_search": max([r[1] for r in results]),
          "min_results_per_search": min([r[1] for r in results]),
          "avg_request_time": sum([r[0]["request_time"].total_seconds() for r in results]) / len(results),
          "max_request_time": max([r[0]["request_time"].total_seconds() for r in results]),
          "min_request_time": min([r[0]["request_time"].total_seconds() for r in results]),
          "p99_request_time": sorted([r[0]["request_time"].total_seconds() for r in results])[int(len(results) * 0.9)],
          "p90_request_time": sorted([r[0]["request_time"].total_seconds() for r in results])[int(len(results) * 0.9)],
        })
        
      case _:
        print(f"Unknown command: {command[0]}")

    command = input("> ").split(" ")

if __name__ == "__main__":
  asyncio.run(main())
