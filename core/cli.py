
import asyncio
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from core.client.consumer import Client
from core.client.embedding_reader import embedding_file_to_keys
from core.client.recall import get_random_embedding_file_path, rank_by_hamming, recall, recall_rand
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
    write_json(f"{env.Vars.RESULT_FOLDER_PREFIX}responses.json", responses)
  
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
  # ordem final e por distancia de Hamming real contra o embedding da query;
  # total_distance (ordem da rede) continua em cada entrada como desempate
  grouped = rank_by_hamming(grouped, command[1], os.path.dirname(command[1]) or ".")

  if write_output:
    write_json(f"{env.Vars.RESULT_FOLDER_PREFIX}grouped_responses.json", grouped)
    print(f"Responses written to responses.json")
    print(f"Time taken to read file: {read_file_time - start_time}")
    print(f"Time taken to process requests: {request_time - read_file_time}")
  return grouped, {
    "read_file_time": read_file_time - start_time,
    "request_time": request_time - read_file_time,
  }

def ho_get_near(client: Client) -> callable:
  async def _call(command: List[str], write_output: bool = True):
    return await get_near(client, command, write_output)
  return _call

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

      case "recall_rand":  # pasta dos embeddings, qtd de imagens, [raio], [pasta do ground truth]
        await recall_rand(command, ho_get_near(client))

      case "recall": # embedding da imagem, [raio], [pasta do ground truth]
        await recall(command, ho_get_near(client))

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
        seed = int(command[4]) if len(command) > 4 else env.Vars.RANDOM_SEED
        results = []

        embedding_file_path = get_random_embedding_file_path(embedding_folder_path, amount_of_searches, seed)
        for i in range(0, amount_of_searches, parallelism):
          print(f"[{i/parallelism}/ {amount_of_searches/parallelism}] Searching for near keys for embedding files: {embedding_file_path[i:i+parallelism]}")
          tasks = [get_near(client, ["get_near_files", k, "64", "12"]) for k in embedding_file_path[i:i+parallelism]]
          r = await asyncio.gather(*tasks)
          results.extend([(res[1], len(res[0])) for res in r])

        write_json(f"{env.Vars.RESULT_FOLDER_PREFIX}performance_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json", {
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
