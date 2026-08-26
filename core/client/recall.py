from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import random
from typing import Dict, List, Tuple

from core.client.utils import write_json
from core.shared import env

def get_random_embedding_file_path(path: str = "", amount: int = 1, seed: int = env.Vars.RANDOM_SEED) -> List[str]:
  # a lista precisa estar ordenada: os.listdir nao garante ordem estavel,
  # e sem isso a mesma seed sortearia imagens diferentes em cada maquina
  files = sorted(f for f in os.listdir(path) if f.endswith(".bin"))
  rng = random.Random(seed)
  return [os.path.join(path, rng.choice(files)) for _ in range(amount)]

RECALL_KS = [1, 10, 50, 100, 500]

_embedding_cache: Dict[str, Dict[str, bytes]] = {}

def load_embeddings(base_path: str) -> Dict[str, bytes]:
  # normaliza para nao carregar duas vezes quando o mesmo diretorio chega
  # escrito de formas diferentes ("./data/x" vs "data/x/")
  base_path = os.path.normpath(base_path)
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

  # sem embedding local nao da pra medir Hamming: joga o candidato para o fim
  # da lista em vez de descarta-lo
  unknown = len(query) * 8 + 1

  ranked = [
    {
      **entry,
      "hamming_distance": hamming(query, embeddings[entry["value"]]) if entry["value"] in embeddings else unknown,
    }
    for entry in grouped
  ]
  ranked.sort(key=lambda entry: (entry["hamming_distance"], entry["total_distance"]))
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

async def recall_rand(command: List[str], get_near: callable) -> Dict[str, any]:
  base_path = command[1]
  amount = int(command[2])
  radius = command[3] if len(command) > 3 else "12"
  ground_truth_path = command[4] if len(command) > 4 else "./data/top500"
  seed = int(command[5]) if len(command) > 5 else env.Vars.RANDOM_SEED

  files = get_random_embedding_file_path(base_path, amount, seed)
  results = []

  for i, file in enumerate(files):
    # get_near ja devolve ordenado por Hamming; a ordem da rede e reconstruida
    # a partir de total_distance so para manter a comparacao entre os rankings
    grouped, times = await get_near(["get_near_files", file, "64", radius], write_output=False)
    result, query_name, truth = measure_recall(file, grouped, times, ground_truth_path=ground_truth_path)
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

  report_path = f"{env.Vars.RESULT_FOLDER_PREFIX}recall_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
  write_json(report_path, {**summary, "results": results})
  return summary

def measure_recall(file: str, grouped: List[Dict[str, any]], times: Dict[str, datetime] = None, ground_truth_path: str = "./data/top500") -> Tuple[Dict[str, any], str, List[str]]:
  truth = load_top500(ground_truth_path, file)
  if truth is None:
    raise ValueError(f"Ground truth not found for {file} in {ground_truth_path}")
  
  by_network = sorted(grouped, key=lambda entry: entry["total_distance"])
  # a propria imagem consultada nao esta no ground truth, entao sai da comparacao
  query_name = os.path.basename(file)
  found = [entry["value"] for entry in by_network if entry["value"] != query_name]
  found_by_hamming = [entry["value"] for entry in grouped if entry["value"] != query_name]

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
  return (result, query_name, truth)

async def recall(command: List[str], get_near: callable) -> Dict[str, any]:
  file = command[1]
  radius = command[2] if len(command) > 2 else "12"
  ground_truth_path = command[3] if len(command) > 3 else "./data/top500"

  grouped, times = await get_near(["get_near_files", file, "64", radius], write_output=False)
  result, query_name, truth = measure_recall(file, grouped, times, ground_truth_path=ground_truth_path)
  print(f"{query_name}: {result['top500_found']}/{len(truth)} top500 found, {result['returned']} returned, request_time={result['request_time']:.3f}s")

  write_json(f"{env.Vars.RESULT_FOLDER_PREFIX}recall_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json", {
    "params": {
      "file": file,
      "radius": radius,
      "ground_truth_path": ground_truth_path,
    },
    "results": [result]
  })

  return result