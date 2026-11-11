"""
Bateria de testes da rede multi-no (docker-compose.multi.yml).

Para cada combinacao de parametros (quantidade de nos x pasta de embeddings x
REPLICATION_FACTOR x K_SIZE) o script:

  1. derruba a rede que estiver de pe;
  2. limpa a persistencia mantendo os node.txt (mesmos node ids e portas
     em todas as rodadas);
  3. sobe a rede com os parametros da combinacao;
  4. espera todos os nos responderem e a DHT se formar;
  5. faz o fill com a pasta de embeddings da combinacao;
  6. roda N buscas (padrao 1000) medindo recall contra os dois ground truths
     (hamming em data/top500* e clip em data/top500-clip) e a latencia;
  7. roda o mesmo lote em paralelo para medir vazao;
  8. derruba a rede e copia a persistencia para ../networks/<grupo>/.

Nada do codigo da rede e alterado: o script so gera um override de compose com
os parametros do grupo e reaproveita as funcoes que ja existem em core/ e
tools/ (store_chunks, get_near, measure_recall, tools/top500.py).

A quantidade de nos de server e uma variavel da matriz (--nodes). O override
de cada grupo troca o numero de replicas do service server e o NODES_PER_PROC
de cada container (--nodes-per-container, padrao 1: um no por container, ou
seja, --nodes containers de server mais o bootstrap).

Um grupo ja rodado so e reaproveitado quando foi rodado com a mesma quantidade
de nos por container. As rodadas antigas usavam 8 nos por container; os
grupos de 128 nos com 8 por container reaproveitam tambem as pastas de antes
de a quantidade de nos entrar na matriz (sem o sufixo _n128). Quando um grupo
e refeito com outro arranjo, os resultados anteriores vao para
<results-dir>/_substituidos/ e ../networks/_substituidos/ em vez de serem
apagados.

Uso tipico (dataset inteiro, 1000 buscas por grupo):

    python tools/bench_matrix.py

So algumas quantidades de nos:

    python tools/bench_matrix.py --nodes 32,64

Cada eixo da matriz pode ser escolhido pela linha de comando (a matriz e o
produto de todos):

    python tools/bench_matrix.py --nodes 64 --embeddings 1024 --rf 1,2,3 --k 10,20,40

Varios nos por container, como nas rodadas antigas:

    python tools/bench_matrix.py --nodes-per-container 8

Em outra faixa de portas (quando a 4000-4128 / 5000-5128 ja esta em uso na
maquina): o bootstrap fica em --base-port, os servers nas seguintes e o HTTP
de cada no em +1000. Os node ids continuam os mesmos, so as portas andam:

    python tools/bench_matrix.py --base-port 14000

Mais PUTs simultaneos no fill (padrao 64; a vazao do fill depende disso, entao
so compare grupos rodados com o mesmo valor):

    python tools/bench_matrix.py --fill-workers 256

Rodada rapida de validacao (2 grupos, 2000 imagens no fill, 50 buscas):

    python tools/bench_matrix.py --only emb1024_rf2_k10_n64,emb1024_rf1_k10_n64 \
        --fill-images 2000 --queries 50
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

# porta do bootstrap no node_identities.json; a rede pode subir em outra faixa
# com --base-port, deslocando todas as portas do mapa pela mesma diferenca
BOOTSTRAP_PORT = 4000
HTTP_PORT_OFFSET = 1000
KEY_SIZE_BITS = 64            # tamanho da chave da rede, igual ao env.Vars.KEY_SIZE
NODE_ID_BITS = 64             # o node id tem o mesmo tamanho da chave
DEFAULT_COMPOSE_FILE = REPO_ROOT / "docker-compose.multi.yml"
PERSIST_DIR = REPO_ROOT / "persist"
BOOTSTRAP_PERSIST_DIRNAME = "persist-bootstrap"
BOOTSTRAP_PERSIST_DIR = REPO_ROOT / BOOTSTRAP_PERSIST_DIRNAME
NODE_FILE_NAME = "node.txt"


def _early_compose_file() -> Path:
    """
    Le o --compose-file antes do argparse.

    A topologia (quantos nos a rede tem) sai do compose, e o SERVER_COUNT
    precisa estar no ambiente antes de importar core.shared.env, que le as
    variaveis no import.
    """
    argv = sys.argv[1:]
    chosen = None
    for index, argument in enumerate(argv):
        if argument == "--compose-file" and index + 1 < len(argv):
            chosen = argv[index + 1]
        elif argument.startswith("--compose-file="):
            chosen = argument.split("=", 1)[1]
    if not chosen:
        return DEFAULT_COMPOSE_FILE
    path = Path(chosen)
    return path if path.is_absolute() else (REPO_ROOT / path)


def detect_topology(compose_file: Path) -> Tuple[int, int]:
    """
    Descobre quantos nos de server a rede tem, a partir do compose.

    docker-compose.multi.yml roda varios nos por processo (x-nodes-per-proc)
    em x-replicas containers; docker-compose.yml roda um no por container.
    """
    text = compose_file.read_text(encoding="utf-8")

    per_process = re.search(r'^x-nodes-per-proc:\s*&nodes-per-proc\s*"?(\d+)"?', text, re.MULTILINE)
    nodes_per_process = int(per_process.group(1)) if per_process else 1

    for pattern in (
        r"^x-replicas:\s*&replicas\s*(\d+)",
        r"^x-custom-server-count:\s*&custom-server-count\s*(\d+)",
        r"replicas:\s*(\d+)",
    ):
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            replicas = int(match.group(1))
            break
    else:
        raise RuntimeError(f"nao consegui descobrir a quantidade de replicas em {compose_file}")

    return (replicas, nodes_per_process)


COMPOSE_FILE = _early_compose_file()
REPLICAS, NODES_PER_PROCESS = detect_topology(COMPOSE_FILE)
SERVER_NODES = REPLICAS * NODES_PER_PROCESS   # quantidade de nos de server do compose, sem override
TOTAL_NODES = SERVER_NODES + 1                # os servers mais o bootstrap

# env.Vars le o ambiente no import, entao o SERVER_COUNT precisa estar setado antes;
# cada grupo troca o env.Vars.SERVER_COUNT pela quantidade de nos dele
os.environ["SERVER_COUNT"] = str(TOTAL_NODES)

import requests  # noqa: E402

from core.cli import get_near  # noqa: E402
from core.shared import env  # noqa: E402
from core.client.consumer import Client  # noqa: E402
from core.client.fill import store_chunks  # noqa: E402
from core.client.recall import RECALL_KS, measure_recall, percentile  # noqa: E402
from core.client.utils import write_json  # noqa: E402

EMBEDDINGS = {
    # sufixo -> (pasta dos embeddings, bytes por arquivo, ground truth de hamming)
    "1024": (REPO_ROOT / "data" / "embeddings-1024", 128, REPO_ROOT / "data" / "top500"),
    "2048": (REPO_ROOT / "data" / "embeddings-2048", 256, REPO_ROOT / "data" / "top500-2048"),
}
CLIP_GROUND_TRUTH = REPO_ROOT / "data" / "top500-clip"

REPLICATION_FACTORS = (2, 1)
K_SIZES = (10, 20)
# quantidade de nos de server (o bootstrap fica de fora); o primeiro valor e o do compose
NODE_COUNTS = (SERVER_NODES, 64, 32)
# os grupos rodados antes de a quantidade de nos entrar na matriz usavam esta rede
LEGACY_SERVER_NODES = 128
# e esta quantidade de nos por container (resultados sem o campo valem como 8)
LEGACY_NODES_PER_CONTAINER = 8
DEFAULT_NODES_PER_CONTAINER = 1
# PUTs simultaneos no fill (o padrao do store_chunks, usado em todas as rodadas antigas)
DEFAULT_FILL_WORKERS = 64

OVERRIDE_TEMPLATE = """# gerado por tools/bench_matrix.py -- grupo {label}
# {group}
services:
  bootstrap:
    environment:
      LISTEN_PORT: "{base_port}"
      K_SIZE: "{k_size}"
      REPLICATION_FACTOR: "{replication_factor}"
      SERVER_COUNT: "{total_nodes}"
      # o bootstrap ganha uma pasta de persistencia so dele: com a pasta
      # compartilhada ele disputa slot com os servers, pega a porta de outro no
      # (o node.txt manda na porta) e a rede sobe sem ninguem na porta 4000
      PERSIST_PATH: /app/{bootstrap_persist}
  server:
    environment:
      LISTEN_PORT: "{server_port}"
      BOOTSTRAP_ADDR: "localhost:{base_port}"
      K_SIZE: "{k_size}"
      REPLICATION_FACTOR: "{replication_factor}"
      SERVER_COUNT: "{total_nodes}"
      NODES_PER_PROC: "{nodes_per_container}"
    deploy:
      replicas: {replicas}
"""


# --------------------------------------------------------------------------- log


LOG_SINKS: List[Path] = []


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    for sink in LOG_SINKS:
        try:
            with open(sink, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass


def display_path(path: Path) -> str:
    """Caminho relativo ao repositorio quando da, absoluto quando esta fora dele."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


# ------------------------------------------------------------------------ config


@dataclass(frozen=True)
class Config:
    embeddings: str
    replication_factor: int
    k_size: int
    server_nodes: int
    nodes_per_container: int = DEFAULT_NODES_PER_CONTAINER
    base_port: int = BOOTSTRAP_PORT

    @property
    def label(self) -> str:
        return f"emb{self.embeddings}_rf{self.replication_factor}_k{self.k_size}_n{self.server_nodes}"

    @property
    def legacy_label(self) -> str | None:
        """Nome da pasta dos grupos rodados antes de a quantidade de nos entrar na matriz."""
        if self.server_nodes != LEGACY_SERVER_NODES or self.nodes_per_container != LEGACY_NODES_PER_CONTAINER:
            return None
        return f"emb{self.embeddings}_rf{self.replication_factor}_k{self.k_size}"

    @property
    def group(self) -> str:
        return (
            f"embeddings-{self.embeddings} | "
            f"REPLICATION_FACTOR={self.replication_factor} | "
            f"K_SIZE={self.k_size} | "
            f"nos={self.server_nodes} ({self.nodes_per_container} por container)"
        )

    @property
    def total_nodes(self) -> int:
        return self.server_nodes + 1

    @property
    def replicas(self) -> int:
        return self.server_nodes // self.nodes_per_container

    def http_ports(self) -> List[int]:
        return [self.base_port + HTTP_PORT_OFFSET + index for index in range(self.total_nodes)]

    @property
    def embeddings_dir(self) -> Path:
        return EMBEDDINGS[self.embeddings][0]

    @property
    def embedding_bytes(self) -> int:
        return EMBEDDINGS[self.embeddings][1]

    @property
    def hamming_ground_truth(self) -> Path:
        return EMBEDDINGS[self.embeddings][2]

    @property
    def chunks_per_image(self) -> int:
        return self.embedding_bytes // (KEY_SIZE_BITS // 8)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "group": self.group,
            "embeddings": f"embeddings-{self.embeddings}",
            "embeddings_dir": display_path(self.embeddings_dir),
            "embedding_bits": self.embedding_bytes * 8,
            "keys_per_image": self.chunks_per_image,
            "replication_factor": self.replication_factor,
            "k_size": self.k_size,
            "server_nodes": self.server_nodes,
            "total_nodes": self.total_nodes,
            "replicas": self.replicas,
            "nodes_per_process": self.nodes_per_container,
            "base_port": self.base_port,
            "http_base_port": self.base_port + HTTP_PORT_OFFSET,
            "hamming_ground_truth": display_path(self.hamming_ground_truth),
            "clip_ground_truth": display_path(CLIP_GROUND_TRUTH),
        }


def build_configs(
    node_counts: Sequence[int],
    embeddings_list: Sequence[str],
    replication_factors: Sequence[int],
    k_sizes: Sequence[int],
    nodes_per_container: int = DEFAULT_NODES_PER_CONTAINER,
    base_port: int = BOOTSTRAP_PORT,
) -> List[Config]:
    # com os valores padrao, a primeira combinacao e a configuracao padrao do
    # repositorio (rede do compose, embeddings-1024, REPLICATION_FACTOR=2, K_SIZE=10)
    for nodes in node_counts:
        if nodes <= 0 or nodes % nodes_per_container:
            raise SystemExit(
                f"--nodes {nodes}: a quantidade de nos precisa ser multiplo de "
                f"--nodes-per-container ({nodes_per_container})"
            )
    return [
        Config(
            embeddings=embeddings,
            replication_factor=replication_factor,
            k_size=k_size,
            server_nodes=nodes,
            nodes_per_container=nodes_per_container,
            base_port=base_port,
        )
        for nodes in node_counts
        for embeddings in embeddings_list
        for replication_factor in replication_factors
        for k_size in k_sizes
    ]


# -------------------------------------------------------------------- identidades


def _read_node_file(path: Path) -> Tuple[str | None, int | None]:
    try:
        parts = path.read_text(encoding="utf-8").split()
    except OSError:
        return (None, None)
    node_id = parts[0] if parts else None
    port = int(parts[1]) if len(parts) > 1 else None
    return (node_id, port)


def load_or_build_identities(networks_dir: Path, seed: int, server_nodes: int) -> Dict[str, Any]:
    """
    Fixa os node ids e as portas dos nos da maior rede da bateria.

    O arquivo ../networks/node_identities.json e a fonte da verdade: enquanto ele
    existir, toda rede recriada sobe com exatamente os mesmos ids, mesmo depois
    de a persistencia ser apagada. As redes menores usam os primeiros servers do
    mapa (ver identities_for), entao uma rede de 32 nos e um pedaco da de 128.
    """
    path = networks_dir / "node_identities.json"
    if path.exists():
        identities = json.loads(path.read_text(encoding="utf-8"))
        available = len(identities["servers"])
        if available < server_nodes:
            # estende o mapa sem mexer nos ids que ja existem
            rng = random.Random(f"{seed}-{available}")
            used = {entry["node_id"] for entry in identities["servers"]} | {identities["bootstrap"]["node_id"]}
            for index in range(available, server_nodes):
                while True:
                    candidate = rng.getrandbits(NODE_ID_BITS).to_bytes(NODE_ID_BITS // 8, "big").hex()
                    if candidate not in used:
                        used.add(candidate)
                        break
                identities["servers"].append(
                    {"slot": f"slot_{index}", "node_id": candidate, "port": BOOTSTRAP_PORT + 1 + index}
                )
            identities["total_nodes"] = server_nodes + 1
            write_json(str(path), identities)
            log(f"{path} tinha {available} servers; estendido para {server_nodes} (ids antigos mantidos)")
        log(f"identidades carregadas de {path} ({len(identities['servers'])} servers + bootstrap)")
        return identities

    rng = random.Random(seed)
    used: set[str] = set()

    def new_id() -> str:
        while True:
            candidate = rng.getrandbits(NODE_ID_BITS).to_bytes(NODE_ID_BITS // 8, "big").hex()
            if candidate not in used:
                used.add(candidate)
                return candidate

    servers = []
    reused = 0
    for index in range(server_nodes):
        slot = f"slot_{index}"
        node_id, port = _read_node_file(PERSIST_DIR / slot / NODE_FILE_NAME)
        if node_id and node_id not in used:
            used.add(node_id)
            reused += 1
        else:
            node_id = new_id()
        servers.append({"slot": slot, "node_id": node_id, "port": port or (BOOTSTRAP_PORT + 1 + index)})

    bootstrap_id, _ = _read_node_file(BOOTSTRAP_PERSIST_DIR / "slot_0" / NODE_FILE_NAME)
    if not bootstrap_id or bootstrap_id in used:
        bootstrap_id = new_id()

    identities = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_nodes": server_nodes + 1,
        "bootstrap": {"slot": "slot_0", "node_id": bootstrap_id, "port": BOOTSTRAP_PORT},
        "servers": servers,
    }
    networks_dir.mkdir(parents=True, exist_ok=True)
    write_json(str(path), identities)
    log(f"mapa de identidades criado em {path} ({reused} ids reaproveitados da persistencia atual)")
    return identities


def identities_for(identities: Dict[str, Any], server_nodes: int) -> Dict[str, Any]:
    """Recorta o mapa de identidades para uma rede com server_nodes servers."""
    return {**identities, "total_nodes": server_nodes + 1, "servers": identities["servers"][:server_nodes]}


def reset_persistence(identities: Dict[str, Any], base_port: int = BOOTSTRAP_PORT) -> None:
    """
    Zera o conteudo dos nos e reescreve os node.txt com os ids fixos.

    As portas do mapa de identidades sao relativas ao BOOTSTRAP_PORT; com outro
    base_port todas andam pela mesma diferenca (o node.txt manda na porta do no).
    """
    shift = base_port - BOOTSTRAP_PORT
    expected = {entry["slot"] for entry in identities["servers"]}

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    for existing in PERSIST_DIR.iterdir():
        if existing.is_dir() and existing.name not in expected:
            # slots extras aparecem quando algum no corre para criar um slot novo
            shutil.rmtree(existing, ignore_errors=True)
        elif existing.is_file():
            existing.unlink(missing_ok=True)

    for entry in identities["servers"]:
        slot_dir = PERSIST_DIR / entry["slot"]
        slot_dir.mkdir(parents=True, exist_ok=True)
        (slot_dir / "state.pkl").unlink(missing_ok=True)
        (slot_dir / NODE_FILE_NAME).write_text(f"{entry['node_id']}\n{entry['port'] + shift}\n", encoding="utf-8")

    bootstrap = identities["bootstrap"]
    shutil.rmtree(BOOTSTRAP_PERSIST_DIR, ignore_errors=True)
    slot_dir = BOOTSTRAP_PERSIST_DIR / bootstrap["slot"]
    slot_dir.mkdir(parents=True, exist_ok=True)
    (slot_dir / NODE_FILE_NAME).write_text(f"{bootstrap['node_id']}\n{bootstrap['port'] + shift}\n", encoding="utf-8")

    log(
        f"persistencia zerada: {len(identities['servers'])} slots de server + 1 slot de bootstrap "
        f"(portas a partir de {base_port})"
    )


# ------------------------------------------------------------------------ docker


def compose_args(override: Path, *args: str) -> List[str]:
    return [
        "docker", "compose",
        "--project-directory", str(REPO_ROOT),
        "-f", str(COMPOSE_FILE),
        "-f", str(override),
        *args,
    ]


def run_compose(
    override: Path, *args: str, check: bool = True, timeout: int = 3600, quiet: bool = False
) -> subprocess.CompletedProcess:
    command = compose_args(override, *args)
    if not quiet:
        log(f"$ {' '.join(command)}")
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    if not quiet:
        for stream in (result.stdout, result.stderr):
            if stream and stream.strip():
                log(stream.strip()[-4000:])
    if check and result.returncode != 0:
        raise RuntimeError(f"docker compose falhou (exit {result.returncode}): {' '.join(args)}")
    return result


def write_override(config: Config, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        OVERRIDE_TEMPLATE.format(
            label=config.label,
            group=config.group,
            k_size=config.k_size,
            replication_factor=config.replication_factor,
            total_nodes=config.total_nodes,
            replicas=config.replicas,
            nodes_per_container=config.nodes_per_container,
            bootstrap_persist=BOOTSTRAP_PERSIST_DIRNAME,
            base_port=config.base_port,
            server_port=config.base_port + 1,
        ),
        encoding="utf-8",
    )
    return destination


def wait_http_ready(ports: Sequence[int], timeout: int) -> List[int]:
    """Espera todos os listeners HTTP responderem (400 em /get sem key)."""
    deadline = time.time() + timeout
    pending = set(ports)

    def probe(port: int) -> Tuple[int, bool]:
        try:
            response = requests.get(f"http://localhost:{port}/get", timeout=3)
            return (port, response.status_code in (200, 400, 404))
        except requests.RequestException:
            return (port, False)

    while pending and time.time() < deadline:
        with ThreadPoolExecutor(max_workers=32) as executor:
            for port, ok in executor.map(probe, sorted(pending)):
                if ok:
                    pending.discard(port)
        if pending:
            log(f"aguardando {len(pending)}/{len(ports)} nos subirem (ex.: {sorted(pending)[:5]})")
            time.sleep(5)
    return sorted(pending)


BOOTSTRAPPED_PATTERN = re.compile(r"found ([1-9]\d*) neighbors")
LISTENING_PATTERN = re.compile(r"Server listening on port")
UNHEALTHY_PATTERNS = ("Routing table is empty", "found 0 neighbors")


def wait_dht_ready(config: Config, override: Path, timeout: int, recent_window: str = "30s") -> bool:
    """
    Confirma que a rede se formou, lendo os logs dos containers de server.

    Cada no loga "Server listening on port" e, quando entra na rede,
    "Bootstrap via ... found N neighbors"; enquanto nao entra, o healer fica
    logando "Routing table is empty". A rede so e considerada pronta quando
    todos os nos entraram e a janela recente esta limpa.

    A sonda obvia -- pedir /get de uma chave conhecida -- nao serve: quando a
    chave tem mais de um valor o handler devolve 500 ("unhashable type: 'list'",
    o ValueSpiderCrawl monta um Counter com os valores), e a chave de metrica do
    bootstrap vira multi-valor no segundo report.
    """
    deadline = time.time() + timeout
    while True:
        logs = run_compose(override, "logs", "server", check=False, timeout=600, quiet=True).stdout or ""
        recent = run_compose(
            override, "logs", "--since", recent_window, "server", check=False, timeout=600, quiet=True
        ).stdout or ""

        listening = len(LISTENING_PATTERN.findall(logs))
        bootstrapped = len(BOOTSTRAPPED_PATTERN.findall(logs))
        unhealthy = sum(recent.count(pattern) for pattern in UNHEALTHY_PATTERNS)

        log(
            f"formando a rede: {listening}/{config.server_nodes} nos ouvindo, "
            f"{bootstrapped}/{config.server_nodes} entraram na rede, "
            f"{unhealthy} avisos de rota vazia nos ultimos {recent_window}"
        )
        if listening >= config.server_nodes and bootstrapped >= config.server_nodes and unhealthy == 0:
            break
        if time.time() >= deadline:
            return False
        time.sleep(10)

    # confirma que o caminho de busca responde de ponta a ponta; r=0 numa chave
    # aleatoria nao grava nada na rede e nao suja o resultado do recall
    probe_key = random.getrandbits(KEY_SIZE_BITS).to_bytes(KEY_SIZE_BITS // 8, "big").hex()
    port = random.choice(config.http_ports()[1:])
    try:
        response = requests.get(
            f"http://localhost:{port}/get_near", params={"key": probe_key, "r": 0}, timeout=60
        )
    except requests.RequestException as error:
        log(f"o caminho de busca nao respondeu na porta {port}: {type(error).__name__}: {error}")
        return False
    if response.status_code != 200:
        log(f"o caminho de busca respondeu {response.status_code} na porta {port}: {response.text[:200]}")
        return False

    log(f"rede formada: {bootstrapped} nos na rede e /get_near respondendo pela porta {port}")
    return True


# ------------------------------------------------------------------ ground truth


def load_tool_module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"_bench_{name}", REPO_ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_hamming_ground_truth(config: Config) -> None:
    """
    Gera data/top500-2048 quando ele nao existe.

    Reaproveita o tools/top500.py como esta: as constantes BITS/BYTES sao globais
    do modulo, entao basta troca-las na instancia importada -- nenhum arquivo do
    projeto e alterado.
    """
    output = config.hamming_ground_truth
    bits = config.embedding_bytes * 8
    expected = sum(1 for _ in config.embeddings_dir.glob("*.bin"))

    if output.is_dir():
        existing = sum(1 for _ in output.glob("*.json"))
        if existing >= expected * 0.99:
            log(f"ground truth de hamming ja existe: {output} ({existing} arquivos)")
            return
        log(f"{output} incompleto ({existing}/{expected}), regerando")

    log(f"gerando ground truth de {bits} bits em {output} (leva alguns minutos e ~2GB de RAM)")
    module = load_tool_module("top500")
    module.BITS = bits
    module.BYTES = config.embedding_bytes

    started = time.time()
    names, signatures = module.read_signatures(str(config.embeddings_dir))
    log(f"  {len(names)} embeddings lidos em {human_time(time.time() - started)}")
    started = time.time()
    neighbors, distances = module.top_k(signatures)
    log(f"  top-{module.K} calculado em {human_time(time.time() - started)}")
    started = time.time()
    module.write_files(str(output), names, neighbors, distances)
    log(f"  {len(names)} arquivos escritos em {human_time(time.time() - started)}")


# ---------------------------------------------------------------------- amostra


def scan_names(directory: Path, suffix: str, expected_size: int | None = None) -> set[str]:
    """
    Lista os nomes de uma pasta ja no formato do arquivo de embedding.

    Os ground truths guardam "<imagem>.jpg.bin.json", entao o .json sai fora e
    sobra exatamente o nome que a pasta de embeddings usa.
    """
    if not directory.is_dir():
        raise RuntimeError(f"pasta nao encontrada: {directory}")
    names: set[str] = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.endswith(suffix) or not entry.is_file():
                continue
            if expected_size is not None and entry.stat().st_size != expected_size:
                continue
            names.add(entry.name[: -len(suffix)] if suffix == ".json" else entry.name)
    return names


def build_query_sample(configs: Sequence[Config], amount: int, seed: int, cache_path: Path) -> List[str]:
    """
    Sorteia as imagens da bateria.

    A amostra e a mesma em todos os grupos (e fica gravada em disco): so entram
    imagens que existem em todas as pastas de embeddings usadas e nos dois
    ground truths, senao um grupo mediria recall sobre outro conjunto.
    """
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("amount") == amount and cached.get("seed") == seed and cached.get("sources"):
            log(f"amostra de {amount} imagens reaproveitada de {cache_path}")
            return cached["files"]

    sources: List[str] = []
    common: set[str] | None = None

    def intersect(names: set[str], origin: Path) -> None:
        nonlocal common
        sources.append(display_path(origin))
        common = names if common is None else (common & names)
        log(f"  {origin.name}: {len(names)} arquivos -> {len(common)} em comum")

    log("montando a amostra de consultas")
    for config in {config.embeddings: config for config in configs}.values():
        intersect(scan_names(config.embeddings_dir, ".bin", config.embedding_bytes), config.embeddings_dir)
        intersect(scan_names(config.hamming_ground_truth, ".json"), config.hamming_ground_truth)
    intersect(scan_names(CLIP_GROUND_TRUTH, ".json"), CLIP_GROUND_TRUTH)

    candidates = sorted(common or set())
    if len(candidates) < amount:
        raise RuntimeError(f"so ha {len(candidates)} imagens com embedding e ground truth completos, precisava de {amount}")

    files = random.Random(seed).sample(candidates, amount)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(cache_path), {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "amount": amount,
        "seed": seed,
        "candidates": len(candidates),
        "sources": sources,
        "sampling": "random.Random(seed).sample -- sem reposicao, mesma amostra em todos os grupos",
        "files": files,
    })
    log(f"amostra de {amount} imagens sorteada entre {len(candidates)} candidatas e salva em {cache_path}")
    return files


# ---------------------------------------------------------------------- medicoes


async def run_fill(
    config: Config, urls: List[str], out_dir: Path, hard_limit: int, workers: int = DEFAULT_FILL_WORKERS
) -> Dict[str, Any]:
    log(f"fill de {config.embeddings_dir} (limite de {hard_limit} chunks, {workers} requisicoes simultaneas)")
    started = time.time()
    submitted, elapsed = await store_chunks(
        str(config.embeddings_dir),
        urls,
        hard_limit=hard_limit,
        max_workers=workers,
        hash_size_bits=KEY_SIZE_BITS,
        failure_report_path=str(out_dir / "fill_failed_requests.json"),
    )
    report = json.loads((out_dir / "fill_failed_requests.json").read_text(encoding="utf-8"))
    failures = report.get("failure_count", 0)
    result = {
        "base_path": display_path(config.embeddings_dir),
        "hash_size_bits": KEY_SIZE_BITS,
        "hard_limit": hard_limit,
        "workers": workers,
        "chunks_submitted": submitted,
        "chunks_failed": failures,
        "chunks_ok": submitted - failures,
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_human": human_time(elapsed),
        "chunks_per_second": round(submitted / elapsed, 2) if elapsed else None,
        "failure_report": display_path(out_dir / "fill_failed_requests.json"),
    }
    log(
        f"fill terminado: {result['chunks_ok']}/{submitted} chunks em {result['elapsed_human']} "
        f"({result['chunks_per_second']} chunks/s, {failures} falhas) "
        f"[relogio: {human_time(time.time() - started)}]"
    )
    return result


async def query_once(client: Client, config: Config, name: str, radius: int, timeout: float):
    path = str(config.embeddings_dir / name)
    return await asyncio.wait_for(
        get_near(client, ["get_near_files", path, str(KEY_SIZE_BITS), str(radius)], write_output=False),
        timeout=timeout,
    )


async def run_sequential(
    client: Client, config: Config, files: Sequence[str], radius: int, timeout: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    started = time.time()

    for index, name in enumerate(files, start=1):
        query_start = time.perf_counter()
        try:
            grouped, times = await query_once(client, config, name, radius, timeout)
        except Exception as error:  # timeout, conexao recusada, json invalido...
            failures.append({"file": name, "error": f"{type(error).__name__}: {error}"})
            log(f"  [{index}/{len(files)}] {name}: FALHOU ({type(error).__name__})")
            continue
        records.append({
            "name": name,
            "grouped": grouped,
            "times": times,
            "wall_seconds": time.perf_counter() - query_start,
        })
        if index % 50 == 0 or index == len(files):
            done = len(records)
            average = sum(record["wall_seconds"] for record in records) / max(done, 1)
            log(
                f"  [{index}/{len(files)}] buscas sequenciais, "
                f"{average:.3f}s por busca, {len(failures)} falhas, "
                f"decorrido {human_time(time.time() - started)}"
            )
    return records, failures


async def run_parallel(
    client: Client, config: Config, files: Sequence[str], radius: int, concurrency: int, timeout: float
) -> Dict[str, Any]:
    log(f"lote paralelo: {len(files)} buscas com {concurrency} simultaneas")
    semaphore = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    failures = 0

    async def one(name: str) -> None:
        nonlocal failures
        async with semaphore:
            query_start = time.perf_counter()
            try:
                await query_once(client, config, name, radius, timeout)
            except Exception:
                failures += 1
                return
            latencies.append(time.perf_counter() - query_start)

    started = time.perf_counter()
    await asyncio.gather(*(one(name) for name in files))
    wall = time.perf_counter() - started

    result = {
        "concurrency": concurrency,
        "queries": len(files),
        "queries_ok": len(latencies),
        "queries_failed": failures,
        "wall_seconds": round(wall, 3),
        "queries_per_second": round(len(latencies) / wall, 3) if wall else None,
        "latency": latency_stats(latencies),
    }
    log(
        f"lote paralelo: {result['queries_per_second']} buscas/s em {human_time(wall)} "
        f"({failures} falhas)"
    )
    return result


def latency_stats(values: Sequence[float]) -> Dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    average = sum(ordered) / len(ordered)
    return {
        "avg": round(average, 4),
        "std": round((sum((value - average) ** 2 for value in ordered) / len(ordered)) ** 0.5, 4),
        "p50": round(percentile(ordered, 0.50), 4),
        "p90": round(percentile(ordered, 0.90), 4),
        "p99": round(percentile(ordered, 0.99), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def summarize_recall(
    config: Config,
    records: Sequence[Dict[str, Any]],
    ground_truth: Path,
    radius: int,
    seed: int,
    sample_path: Path,
) -> Dict[str, Any]:
    """
    Mede o recall das respostas ja coletadas contra um ground truth.

    As duas medicoes (hamming e clip) reusam as mesmas respostas da rede, entao a
    diferenca entre elas e so a referencia -- e a rede so e consultada uma vez.
    """
    results = []
    for record in records:
        result, _, _ = measure_recall(
            str(config.embeddings_dir / record["name"]),
            record["grouped"],
            record["times"],
            ground_truth_path=str(ground_truth),
        )
        results.append(result)

    if not results:
        return {"params": {"ground_truth_path": str(ground_truth)}, "results": []}

    def stats(field: str) -> Dict[str, float]:
        values = [result[field] for result in results]
        average = sum(values) / len(values)
        return {
            "avg": average,
            "std": (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5,
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "min": min(values),
            "max": max(values),
        }

    fields = (
        ["top500_found"]
        + [f"recall@{k}" for k in RECALL_KS]
        + [f"hamming_recall@{k}" for k in RECALL_KS]
        + ["returned", "request_time"]
    )
    return {
        "params": {
            "group": config.group,
            "base_path": display_path(config.embeddings_dir),
            "amount": len(results),
            "r": radius,
            "ground_truth_path": display_path(ground_truth),
            "seed": seed,
            "sample_file": display_path(sample_path),
            "replication_factor": config.replication_factor,
            "k_size": config.k_size,
            "server_nodes": config.server_nodes,
        },
        "coverage": sum(result["coverage"] for result in results) / len(results),
        **{field: stats(field) for field in fields},
        "results": results,
    }


# --------------------------------------------------------------------- execucao


def stored_nodes_per_container(summary: Dict[str, Any]) -> int:
    """Nos por container de um resultado gravado; os antigos, sem o campo, usavam 8."""
    return int(summary.get("config", {}).get("nodes_per_process", LEGACY_NODES_PER_CONTAINER))


def set_aside(path: Path, reason: str) -> None:
    """Move uma pasta de outra rodada para <pai>/_substituidos/ em vez de apaga-la."""
    if not path.exists():
        return
    destination = path.parent / "_substituidos" / f"{path.name}_{reason}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    log(f"{path} guardado em {destination}")


def backup_network(config: Config, networks_dir: Path, identities: Dict[str, Any], summary: Dict[str, Any]) -> Path:
    destination = networks_dir / config.label
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    shutil.copytree(PERSIST_DIR, destination / "persist")
    shutil.copytree(BOOTSTRAP_PERSIST_DIR, destination / BOOTSTRAP_PERSIST_DIRNAME)
    write_json(str(destination / "node_identities.json"), identities)
    write_json(str(destination / "meta.json"), summary)

    size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    log(f"persistencia copiada para {destination} ({size / 1e6:.1f} MB)")
    return destination


async def run_config(config: Config, options: argparse.Namespace, identities: Dict[str, Any], files: Sequence[str]) -> Dict[str, Any]:
    identities = identities_for(identities, config.server_nodes)
    env.Vars.SERVER_COUNT = config.total_nodes  # tamanho do pool de conexoes do fill
    out_dir = options.results_dir / config.label

    # resultado de outro arranjo de nos por container nao e sobrescrito: vai para _substituidos/
    previous_summary = out_dir / "summary.json"
    if previous_summary.exists():
        previous = stored_nodes_per_container(json.loads(previous_summary.read_text(encoding="utf-8")))
        if previous != config.nodes_per_container:
            set_aside(out_dir, f"{previous}-por-container")
            set_aside(options.networks_dir / config.label, f"{previous}-por-container")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_log = out_dir / "run.log"
    LOG_SINKS.append(config_log)
    started_at = datetime.now(timezone.utc)
    started = time.time()

    try:
        log("=" * 78)
        log(f"GRUPO {config.label} -- {config.group}")
        log("=" * 78)

        override = write_override(config, out_dir / "compose.override.yml")
        write_json(str(out_dir / "config.json"), config.as_dict())

        log("derrubando qualquer rede que esteja de pe")
        run_compose(override, "down", "--remove-orphans", "-t", "30", check=False)

        reset_persistence(identities, config.base_port)

        log(
            f"subindo {config.total_nodes} nos ({config.replicas} containers x {config.nodes_per_container} "
            f"= {config.server_nodes} servers + bootstrap)"
        )
        up_args = ["up", "-d"]
        if options.build:
            up_args.append("--build")
        run_compose(override, *up_args)

        missing = wait_http_ready(config.http_ports(), options.ready_timeout)
        if missing:
            raise RuntimeError(f"{len(missing)} nos nao responderam em {options.ready_timeout}s: {missing[:10]}")
        log(f"todos os {config.total_nodes} nos responderam no HTTP")

        if not wait_dht_ready(config, override, options.dht_timeout):
            raise RuntimeError(f"a rede nao se formou em {options.dht_timeout}s (veja os logs dos containers de server)")

        log(f"esperando {options.settle}s para as tabelas de rotas assentarem")
        time.sleep(options.settle)

        urls = [f"http://localhost:{port}" for port in config.http_ports()]
        hard_limit = options.fill_limit or (
            options.fill_images * config.chunks_per_image if options.fill_images else 10 ** 9
        )
        fill = await run_fill(config, urls, out_dir, hard_limit, options.fill_workers)

        client = Client(urls=urls)
        try:
            log(f"rodando {len(files)} buscas sequenciais (r={options.radius})")
            records, query_failures = await run_sequential(client, config, files, options.radius, options.query_timeout)
            if not records:
                raise RuntimeError("nenhuma busca sequencial terminou com sucesso")

            hamming = summarize_recall(config, records, config.hamming_ground_truth, options.radius, options.seed, options.sample_path)
            clip = summarize_recall(config, records, CLIP_GROUND_TRUTH, options.radius, options.seed, options.sample_path)
            write_json(str(out_dir / "recall_hamming.json"), hamming)
            write_json(str(out_dir / "recall_clip.json"), clip)
            log(
                f"recall hamming ({config.hamming_ground_truth.name}): coverage={hamming['coverage']:.4f}, "
                f"recall@10={hamming['recall@10']['avg']:.4f}, hamming_recall@10={hamming['hamming_recall@10']['avg']:.4f}"
            )
            log(
                f"recall clip (top500-clip): coverage={clip['coverage']:.4f}, "
                f"recall@10={clip['recall@10']['avg']:.4f}, hamming_recall@10={clip['hamming_recall@10']['avg']:.4f}"
            )

            sequential_latencies = [record["times"]["request_time"].total_seconds() for record in records]
            sequential_wall = sum(record["wall_seconds"] for record in records)
            speed = {
                "params": {
                    "queries": len(files),
                    "radius": options.radius,
                    "keys_per_query": config.chunks_per_image,
                    "query_timeout_seconds": options.query_timeout,
                },
                "fill": fill,
                "sequential": {
                    "queries": len(files),
                    "queries_ok": len(records),
                    "queries_failed": len(query_failures),
                    "wall_seconds": round(sequential_wall, 3),
                    "queries_per_second": round(len(records) / sequential_wall, 3) if sequential_wall else None,
                    "request_latency": latency_stats(sequential_latencies),
                    "wall_latency": latency_stats([record["wall_seconds"] for record in records]),
                    "avg_results_per_query": sum(len(record["grouped"]) for record in records) / len(records),
                },
                "parallel": None,
                "failures": query_failures,
            }
            if options.concurrency > 0:
                speed["parallel"] = await run_parallel(
                    client, config, files, options.radius, options.concurrency, options.query_timeout
                )
            write_json(str(out_dir / "speed.json"), speed)
        finally:
            await client.close()

        log(f"esperando {options.settle_before_down}s para os nos gravarem a persistencia")
        time.sleep(options.settle_before_down)
        log("derrubando a rede")
        run_compose(override, "down", "--remove-orphans", "-t", "60", check=False)

        summary = {
            "label": config.label,
            "group": config.group,
            "status": "ok",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total_seconds": round(time.time() - started, 1),
            "total_human": human_time(time.time() - started),
            "config": config.as_dict(),
            "nodes": {"total": config.total_nodes, "servers": config.server_nodes, "bootstrap": 1},
            "fill": fill,
            "speed": {key: value for key, value in speed.items() if key != "failures"},
            "recall_hamming": {key: value for key, value in hamming.items() if key != "results"},
            "recall_clip": {key: value for key, value in clip.items() if key != "results"},
        }
        summary["backup_path"] = str(backup_network(config, options.networks_dir, identities, summary))
        write_json(str(out_dir / "summary.json"), summary)
        log(f"GRUPO {config.label} concluido em {summary['total_human']}")
        return summary
    finally:
        LOG_SINKS.remove(config_log)


def write_matrix_summary(summaries: Sequence[Dict[str, Any]], results_dir: Path) -> None:
    if not summaries:
        return
    write_json(str(results_dir / "matrix_summary.json"), {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "groups": list(summaries),
    })

    rows = []
    for summary in summaries:
        if summary.get("status") != "ok":
            rows.append({
                "grupo": summary["label"],
                "status": summary.get("status"),
                "erro": summary.get("error", ""),
            })
            continue
        config = summary["config"]
        fill = summary["fill"]
        hamming = summary["recall_hamming"]
        clip = summary["recall_clip"]
        sequential = summary["speed"]["sequential"]
        parallel = summary["speed"].get("parallel") or {}
        rows.append({
            "grupo": summary["label"],
            "status": "ok",
            "nos": config.get("server_nodes", summary.get("nodes", {}).get("servers")),
            "nos_por_container": stored_nodes_per_container(summary),
            "embeddings": config["embeddings"],
            "replication_factor": config["replication_factor"],
            "k_size": config["k_size"],
            "fill_chunks": fill["chunks_submitted"],
            "fill_falhas": fill["chunks_failed"],
            "fill_tempo": fill["elapsed_human"],
            "fill_chunks_s": fill["chunks_per_second"],
            "fill_workers": fill.get("workers", DEFAULT_FILL_WORKERS),
            "buscas": sequential["queries_ok"],
            "resultados_por_busca": round(sequential["avg_results_per_query"], 2),
            "hamming_coverage": round(hamming["coverage"], 4),
            "hamming_recall@10": round(hamming["recall@10"]["avg"], 4),
            "hamming_recall@100": round(hamming["recall@100"]["avg"], 4),
            "hamming_rerank@10": round(hamming["hamming_recall@10"]["avg"], 4),
            "clip_coverage": round(clip["coverage"], 4),
            "clip_recall@10": round(clip["recall@10"]["avg"], 4),
            "clip_recall@100": round(clip["recall@100"]["avg"], 4),
            "clip_rerank@10": round(clip["hamming_recall@10"]["avg"], 4),
            "latencia_avg": sequential["request_latency"]["avg"],
            "latencia_p90": sequential["request_latency"]["p90"],
            "latencia_p99": sequential["request_latency"]["p99"],
            "seq_buscas_s": sequential["queries_per_second"],
            "par_buscas_s": parallel.get("queries_per_second"),
            "par_concorrencia": parallel.get("concurrency"),
            "backup": summary.get("backup_path", ""),
        })

    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path = results_dir / "matrix_summary.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    log(f"resumo da matriz em {path} e em {results_dir / 'matrix_summary.json'}")


# ------------------------------------------------------------------------- main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bateria de fill + recall + velocidade para cada combinacao de parametros da rede.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queries", type=int, default=1000, help="quantidade de buscas por grupo")
    parser.add_argument("--radius", type=int, default=12, help="raio de hamming das buscas")
    parser.add_argument("--seed", type=int, default=42, help="seed do sorteio da amostra e dos node ids novos")
    parser.add_argument("--concurrency", type=int, default=16, help="buscas simultaneas no lote de vazao (0 desliga)")
    parser.add_argument("--fill-images", type=int, default=None,
                        help="preenche so as N primeiras imagens da pasta (padrao: dataset inteiro)")
    parser.add_argument("--fill-limit", type=int, default=None,
                        help="limite bruto de chunks no fill; tem prioridade sobre --fill-images")
    parser.add_argument("--fill-workers", type=int, default=DEFAULT_FILL_WORKERS,
                        help="PUTs simultaneos no fill; a vazao do fill depende dele, entao compare "
                             "grupos rodados com o mesmo valor")
    parser.add_argument("--nodes", default=",".join(str(nodes) for nodes in NODE_COUNTS),
                        help=f"quantidades de nos de server da matriz, separadas por virgula "
                             f"(multiplos de --nodes-per-container)")
    parser.add_argument("--nodes-per-container", type=int, default=DEFAULT_NODES_PER_CONTAINER,
                        help="nos de server por container (as rodadas antigas usavam 8)")
    parser.add_argument("--base-port", type=int, default=BOOTSTRAP_PORT,
                        help=f"porta kademlia do bootstrap; os servers usam as seguintes e o HTTP "
                             f"de cada no fica em +{HTTP_PORT_OFFSET}")
    parser.add_argument("--embeddings", default=",".join(EMBEDDINGS),
                        help=f"pastas de embeddings da matriz, separadas por virgula ({', '.join(EMBEDDINGS)})")
    parser.add_argument("--replication-factors", "--rf", dest="replication_factors",
                        default=",".join(str(value) for value in REPLICATION_FACTORS),
                        help="valores de REPLICATION_FACTOR da matriz, separados por virgula")
    parser.add_argument("--k-sizes", "--k", dest="k_sizes", default=",".join(str(value) for value in K_SIZES),
                        help="valores de K_SIZE da matriz, separados por virgula")
    parser.add_argument("--only", default=None, help="lista de grupos separada por virgula (ex.: emb1024_rf2_k10_n128)")
    parser.add_argument("--skip", default=None, help="lista de grupos a pular, separada por virgula")
    parser.add_argument("--force", action="store_true", help="refaz grupos que ja tem summary.json")
    parser.add_argument("--no-build", dest="build", action="store_false", help="nao passa --build no primeiro up")
    parser.add_argument("--skip-ground-truth", action="store_true",
                        help="nao gera data/top500-2048 mesmo que ele nao exista")
    parser.add_argument("--compose-file", type=Path, default=COMPOSE_FILE,
                        help="compose usado para subir a rede; a topologia sai dele")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "data" / "results" / "bench")
    parser.add_argument("--networks-dir", type=Path, default=REPO_ROOT.parent / "networks")
    parser.add_argument("--ready-timeout", type=int, default=600, help="segundos para todos os nos responderem no HTTP")
    parser.add_argument("--dht-timeout", type=int, default=600, help="segundos para a DHT se formar")
    parser.add_argument("--settle", type=int, default=30, help="pausa depois que a rede sobe")
    parser.add_argument("--settle-before-down", type=int, default=90,
                        help="pausa antes do down, para os nos gravarem o state.pkl (PERSIST_SAVE_INTERVAL=60)")
    parser.add_argument("--query-timeout", type=float, default=180.0, help="timeout de cada busca, em segundos")
    parser.add_argument("--keep-last-persist", action="store_true",
                        help="nao limpa a persistencia do ultimo grupo ao terminar")
    parser.add_argument("--dry-run", action="store_true", help="mostra o plano e sai")
    options = parser.parse_args(argv)
    # o compose ja foi resolvido no import (a topologia depende dele), entao
    # o valor do argparse so serve para o --help
    options.compose_file = COMPOSE_FILE

    def int_list(flag: str, raw: str) -> List[int]:
        try:
            values = [int(value) for value in raw.split(",") if value.strip()]
        except ValueError:
            parser.error(f"{flag} invalido: {raw!r}")
        if not values:
            parser.error(f"{flag} vazio")
        if any(value <= 0 for value in values):
            parser.error(f"{flag} so aceita inteiros positivos: {raw!r}")
        return list(dict.fromkeys(values))  # sem repetidos, na ordem dada

    options.nodes = int_list("--nodes", options.nodes)
    options.replication_factors = int_list("--replication-factors", options.replication_factors)
    options.k_sizes = int_list("--k-sizes", options.k_sizes)
    if options.nodes_per_container <= 0:
        parser.error("--nodes-per-container precisa ser positivo")
    if options.fill_workers <= 0:
        parser.error("--fill-workers precisa ser positivo")
    largest = max(options.nodes) + 1
    if largest > HTTP_PORT_OFFSET:
        parser.error(f"com {largest} nos as portas kademlia invadiriam a faixa HTTP (+{HTTP_PORT_OFFSET})")
    if options.base_port < 1024 or options.base_port + HTTP_PORT_OFFSET + largest - 1 > 65535:
        parser.error(f"--base-port {options.base_port}: a faixa de portas precisa ficar entre 1024 e 65535")
    if options.nodes_per_container > 1 and "NODES_PER_PROC" not in COMPOSE_FILE.read_text(encoding="utf-8"):
        parser.error(f"{COMPOSE_FILE.name} roda um no por container; use o docker-compose.multi.yml")

    # aceita "1024" ou "embeddings-1024"
    embeddings = [value.strip().removeprefix("embeddings-") for value in options.embeddings.split(",") if value.strip()]
    unknown = [value for value in embeddings if value not in EMBEDDINGS]
    if unknown or not embeddings:
        parser.error(f"--embeddings invalido: {options.embeddings!r} (opcoes: {', '.join(EMBEDDINGS)})")
    options.embeddings = list(dict.fromkeys(embeddings))
    return options


async def main_async(options: argparse.Namespace) -> int:
    options.results_dir.mkdir(parents=True, exist_ok=True)
    options.networks_dir.mkdir(parents=True, exist_ok=True)
    LOG_SINKS.append(options.results_dir / "bench_matrix.log")

    configs = build_configs(
        options.nodes, options.embeddings, options.replication_factors, options.k_sizes,
        options.nodes_per_container, options.base_port,
    )

    def matches(config: Config, labels: set[str]) -> bool:
        return config.label in labels or config.legacy_label in labels

    if options.only:
        wanted = {label.strip() for label in options.only.split(",") if label.strip()}
        unknown = {label for label in wanted if not any(matches(config, {label}) for config in configs)}
        if unknown:
            raise SystemExit(f"grupo desconhecido: {sorted(unknown)}")
        configs = [config for config in configs if matches(config, wanted)]
    if options.skip:
        skipped = {label.strip() for label in options.skip.split(",") if label.strip()}
        configs = [config for config in configs if not matches(config, skipped)]

    # resultados ja prontos entram no resumo da matriz junto com os grupos novos
    summaries_by_label: Dict[str, Dict[str, Any]] = {}
    pending = []
    for config in configs:
        done = None
        for label in (config.label, config.legacy_label):
            if label and (options.results_dir / label / "summary.json").exists():
                done = options.results_dir / label / "summary.json"
                break
        if done and not options.force:
            previous = json.loads(done.read_text(encoding="utf-8"))
            if stored_nodes_per_container(previous) == config.nodes_per_container:
                log(f"grupo {config.label} ja tem {display_path(done)}, pulando (use --force para refazer)")
                summaries_by_label[config.label] = previous
                continue
            log(
                f"grupo {config.label}: {display_path(done)} foi rodado com "
                f"{stored_nodes_per_container(previous)} nos por container, vai ser refeito com "
                f"{config.nodes_per_container}"
            )
        pending.append(config)

    def ordered_summaries() -> List[Dict[str, Any]]:
        return [summaries_by_label[config.label] for config in configs if config.label in summaries_by_label]

    log(f"{len(pending)} grupos na fila: {[config.label for config in pending]}")
    log(
        f"compose: {COMPOSE_FILE.name} -> {options.nodes_per_container} nos por container; "
        f"redes de {options.nodes} servers (+ bootstrap)"
    )
    largest = max(options.nodes) + 1
    log(
        f"portas kademlia {options.base_port}-{options.base_port + largest - 1}, "
        f"HTTP {options.base_port + HTTP_PORT_OFFSET}-{options.base_port + HTTP_PORT_OFFSET + largest - 1} "
        f"(na maior rede)"
    )

    if options.dry_run:
        for config in pending:
            log(
                f"  {config.label}: {config.group} -> {config.replicas} containers, "
                f"ground truth {config.hamming_ground_truth.name} + top500-clip"
            )
        return 0
    if not pending:
        write_matrix_summary(ordered_summaries(), options.results_dir)
        return 0

    identities = load_or_build_identities(options.networks_dir, options.seed, max(config.server_nodes for config in pending))

    pre_backup = options.networks_dir / f"_persist_antes_da_bateria_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    if PERSIST_DIR.exists():
        shutil.copytree(PERSIST_DIR, pre_backup / "persist")
        log(f"persistencia atual guardada em {pre_backup} antes de qualquer alteracao")

    if not options.skip_ground_truth:
        for config in {config.embeddings: config for config in pending}.values():
            ensure_hamming_ground_truth(config)

    options.sample_path = options.results_dir / f"query_sample_{options.queries}_seed{options.seed}.json"
    files = build_query_sample(pending, options.queries, options.seed, options.sample_path)

    summaries: List[Dict[str, Any]] = []
    for position, config in enumerate(pending, start=1):
        log(f"### grupo {position}/{len(pending)}")
        try:
            summary = await run_config(config, options, identities, files)
        except Exception as error:
            log(f"GRUPO {config.label} FALHOU: {type(error).__name__}: {error}")
            summary = {
                "label": config.label,
                "group": config.group,
                "status": "erro",
                "error": f"{type(error).__name__}: {error}",
                "config": config.as_dict(),
            }
            write_json(str(options.results_dir / config.label / "failure.json"), summary)
            override = options.results_dir / config.label / "compose.override.yml"
            if override.exists():
                run_compose(override, "down", "--remove-orphans", "-t", "30", check=False)
        finally:
            options.build = False  # a imagem so precisa ser construida uma vez
        summaries.append(summary)
        summaries_by_label[config.label] = summary
        write_matrix_summary(ordered_summaries(), options.results_dir)

    if not options.keep_last_persist:
        reset_persistence(identities, options.base_port)
        log("persistencia local zerada; as redes ficaram guardadas em " + str(options.networks_dir))

    write_matrix_summary(ordered_summaries(), options.results_dir)
    ok = sum(1 for summary in summaries if summary.get("status") == "ok")
    log(f"bateria terminada: {ok}/{len(summaries)} grupos concluidos")
    return 0 if ok == len(summaries) else 1


def main() -> int:
    options = parse_args()
    try:
        return asyncio.run(main_async(options))
    except KeyboardInterrupt:
        log("interrompido pelo usuario; a rede pode ter ficado de pe (docker compose down para derrubar)")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
