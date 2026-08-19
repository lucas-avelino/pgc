
import asyncio
from datetime import datetime
from typing import List

from core.client.consumer import Client
from core.client.embedding_reader import embedding_file_to_keys
from core.client.utils import write_file, write_json
from core.client.fill import retry_failed_requests, store_chunks
from core.shared import env

async def get_near(client: Client, command: List[str]):
  start_time = datetime.now()
  keys = embedding_file_to_keys(command[1], int(command[2]))
  read_file_time = datetime.now()

  tasks = [client.get_near(k, int(command[3])) for k in keys]
  responses = await asyncio.gather(*tasks)
  request_time = datetime.now()
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

  write_json("grouped_responses.json", grouped)

  print(f"Responses written to responses.json")
  print(f"Time taken to read file: {read_file_time - start_time}")
  print(f"Time taken to process requests: {request_time - read_file_time}")

async def main():
  urls = [f"http://localhost:{i}" for i in range(5000, 5000 + env.Vars.SERVER_COUNT)]
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

      case "retry_failed" | "retry_failed_requests":
        report_path = command[1] if len(command) > 1 else "fill_failed_requests.json"
        output_report_path = command[2] if len(command) > 2 else None
        result = retry_failed_requests(report_path, output_report_path=output_report_path)
        print(f"Retry summary: {result['submitted']} submitted, {result['successful']} successful, {result['failed']} failed")
        print(f"Retry failures report: {result['report_path']}")
      case _:
        print(f"Unknown command: {command[0]}")

    command = input("> ").split(" ")

if __name__ == "__main__":
  asyncio.run(main())
