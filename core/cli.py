
import asyncio
from datetime import datetime
import json
import re

from utils.client import Client
from utils.embedding_reader import embedding_file_to_keys
from utils.utils import write_file, write_json

async def get_near(client: Client, command: [str]):
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
  client = Client(urls=["http://localhost:5000","http://localhost:5001","http://localhost:5002","http://localhost:5003","http://localhost:5004"])

  command = input("> ").split(" ")
  while command[0] != "exit":
    match command[0]:
      case "fill":
        from core.tools.utils.fill import store_chunks
        base_path = command[1]
        base_url = command[2]
        amount_of_urls = int(command[3])
        limit = int(command[4]) if len(command) > 4 else 1000000
        hash_size_bits = int(command[5]) if len(command) > 5 else 64
        urls = [f"{base_url}:{port}" for port in range(5000, 5000 + amount_of_urls)]
        print(f"Storing chunks from {base_path} to {urls} with limit {limit} and hash size {hash_size_bits} bits.")
        await store_chunks(base_path, urls, hard_limit=limit, hash_size_bits=hash_size_bits)

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

      case _:
        print(f"Unknown command: {command[0]}")

    command = input("> ").split(" ")

if __name__ == "__main__":
  asyncio.run(main())