"""
Monitor de CPU e memoria para o tools/bench_matrix.py.

Roda num processo separado (para nao disputar o event loop do cliente) e, a
cada `interval` segundos, grava em um CSV uma linha por container da rede, uma
por processo do cliente e uma da maquina:

  - containers: le os contadores do cgroup do container (cpu.stat e
    memory.stat/memory.current); se o cgroup nao for v2 ou nao estiver
    acessivel, cai para o /proc do processo principal do container;
  - cliente: o processo do bench_matrix e os filhos dele (os processos do fill),
    pelo /proc;
  - maquina: /proc/stat, /proc/loadavg e /proc/meminfo.

cpu_pct e por cento de um nucleo (100 = um nucleo inteiro); na linha da maquina
e a soma de todos os nucleos (3500 = 35 nucleos ocupados). mem_mb e a memoria
anonima do container (sem cache de arquivo) ou o RSS do processo.

So funciona em Linux; em outro sistema o bench_matrix desliga o monitor.

Nada daqui chama o Docker durante a coleta: a lista de containers chega pronta
do bench_matrix (discover_containers), que roda um docker inspect uma vez so.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Sequence, Tuple

PHASES = ("subida", "fill", "cache", "sequencial", "recall", "paralelo", "persistencia")
CSV_FIELDS = ("time", "elapsed_s", "phase", "kind", "name", "cpu_pct", "mem_mb", "load1")
CGROUP_ROOT = "/sys/fs/cgroup"


def supported() -> bool:
    return os.path.isfile("/proc/stat") and os.path.isdir(CGROUP_ROOT)


# ------------------------------------------------------------------ descoberta


def discover_containers(container_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """Nome, pid principal e pasta do cgroup de cada container (um docker inspect so)."""
    ids = [cid for cid in container_ids if cid.strip()]
    if not ids:
        return []
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.Id}}\t{{.Name}}\t{{.State.Pid}}", *ids],
        capture_output=True, text=True, timeout=300,
    )
    containers = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        cid, name, pid = parts[0], parts[1].lstrip("/"), int(parts[2] or 0)
        containers.append({"id": cid, "name": name, "pid": pid, "cgroup": cgroup_dir(pid)})
    return containers


def cgroup_dir(pid: int) -> str | None:
    """Pasta do cgroup v2 do processo, se os contadores estiverem legiveis."""
    if pid <= 0:
        return None
    try:
        with open(f"/proc/{pid}/cgroup", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("0::"):  # cgroup v2: uma linha so, "0::/caminho"
            path = os.path.join(CGROUP_ROOT, line[3:].lstrip("/"))
            if os.access(os.path.join(path, "cpu.stat"), os.R_OK):
                return path
    return None


# --------------------------------------------------------------------- leitura


CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def cgroup_sample(path: str) -> Tuple[float | None, float | None]:
    """(segundos de CPU acumulados, bytes de memoria anonima) do cgroup."""
    cpu = None
    stat = _read(os.path.join(path, "cpu.stat"))
    if stat:
        for line in stat.splitlines():
            if line.startswith("usage_usec "):
                cpu = int(line.split()[1]) / 1e6
                break
    memory = None
    mem_stat = _read(os.path.join(path, "memory.stat"))
    if mem_stat:
        for line in mem_stat.splitlines():
            if line.startswith("anon "):
                memory = float(line.split()[1])
                break
    if memory is None:
        current = _read(os.path.join(path, "memory.current"))
        memory = float(current) if current and current.strip().isdigit() else None
    return cpu, memory


def proc_sample(pid: int) -> Tuple[float | None, float | None]:
    """(segundos de CPU acumulados, RSS em bytes) do processo."""
    stat = _read(f"/proc/{pid}/stat")
    if not stat:
        return None, None
    fields = stat[stat.rfind(")") + 2:].split()
    cpu = (int(fields[11]) + int(fields[12])) / CLOCK_TICKS  # utime + stime
    statm = _read(f"/proc/{pid}/statm")
    memory = int(statm.split()[1]) * PAGE_SIZE if statm else None
    return cpu, memory


def client_processes(root_pid: int, exclude: int) -> Dict[int, str]:
    """O processo do bench_matrix e os filhos diretos dele (fill), menos o monitor."""
    found = {root_pid: "cliente"}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return found
    for entry in entries:
        if not entry.isdigit() or int(entry) in (root_pid, exclude):
            continue
        stat = _read(f"/proc/{entry}/stat")
        if not stat:
            continue
        fields = stat[stat.rfind(")") + 2:].split()
        if int(fields[1]) != root_pid:  # ppid
            continue
        cmdline = (_read(f"/proc/{entry}/cmdline") or "").replace("\0", " ")
        if "multiprocessing" in cmdline:
            found[int(entry)] = f"cliente-filho-{entry}"
    return found


def machine_sample() -> Tuple[float | None, float | None, float | None]:
    """(segundos de CPU ocupados somando todos os nucleos, memoria usada em bytes, load1)."""
    busy = None
    stat = _read("/proc/stat")
    if stat:
        values = [int(value) for value in stat.splitlines()[0].split()[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
        busy = (sum(values[:8]) - idle) / CLOCK_TICKS
    used = None
    meminfo = _read("/proc/meminfo")
    if meminfo:
        info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in meminfo.splitlines() if ":" in line}
        if "MemTotal" in info and "MemAvailable" in info:
            used = float(info["MemTotal"] - info["MemAvailable"])
    load = _read("/proc/loadavg")
    load1 = float(load.split()[0]) if load else None
    return busy, used, load1


# ---------------------------------------------------------------------- coleta


def run(
    csv_path: str,
    containers: List[Dict[str, Any]],
    root_pid: int,
    interval: float,
    phase_value,
    stop_event,
) -> None:
    """Alvo do processo do monitor: coleta ate o stop_event ser setado."""
    started = time.time()
    previous: Dict[Tuple[str, str], Tuple[float, float]] = {}
    own_pid = os.getpid()

    def cpu_pct(key: Tuple[str, str], seconds: float | None, now: float) -> float | None:
        if seconds is None:
            return None
        last = previous.get(key)
        previous[key] = (seconds, now)
        if last is None or now <= last[1]:
            return None
        return round(max(0.0, (seconds - last[0]) / (now - last[1]) * 100), 1)

    def mb(value: float | None) -> float | None:
        return None if value is None else round(value / 2 ** 20, 1)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        while True:
            now = time.time()
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            elapsed = round(now - started, 1)
            phase = PHASES[phase_value.value] if 0 <= phase_value.value < len(PHASES) else "?"
            rows = []

            for container in containers:
                if container["cgroup"]:
                    seconds, memory = cgroup_sample(container["cgroup"])
                else:
                    seconds, memory = proc_sample(container["pid"])
                pct = cpu_pct(("no", container["name"]), seconds, now)
                rows.append((stamp, elapsed, phase, "no", container["name"], pct, mb(memory), None))

            for pid, name in client_processes(root_pid, own_pid).items():
                seconds, memory = proc_sample(pid)
                pct = cpu_pct(("cliente", str(pid)), seconds, now)
                rows.append((stamp, elapsed, phase, "cliente", name, pct, mb(memory), None))

            busy, used, load1 = machine_sample()
            pct = cpu_pct(("maquina", "maquina"), busy, now)
            rows.append((stamp, elapsed, phase, "maquina", "maquina", pct, mb(used), load1))

            # a primeira amostra de cada origem nao tem CPU (precisa de duas leituras)
            writer.writerows(row for row in rows if row[5] is not None or row[3] == "maquina")
            handle.flush()

            if stop_event.wait(interval):
                return


# --------------------------------------------------------------------- resumo


def _stats(values: Sequence[float]) -> Dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "avg": round(sum(ordered) / len(ordered), 1),
        "p95": round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)], 1),
        "max": round(ordered[-1], 1),
    }


def summarize(csv_path: str) -> Dict[str, Any]:
    """Resumo por fase do CSV: cliente, nos (inclusive o mais quente) e maquina."""
    by_phase: Dict[str, List[Dict[str, str]]] = {}
    with open(csv_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_phase.setdefault(row["phase"], []).append(row)

    def number(value: str) -> float | None:
        return float(value) if value not in ("", None) else None

    summary: Dict[str, Any] = {}
    for phase in [phase for phase in PHASES if phase in by_phase]:
        rows = by_phase[phase]
        nodes = [row for row in rows if row["kind"] == "no" and number(row["cpu_pct"]) is not None]
        clients = [row for row in rows if row["kind"] == "cliente" and number(row["cpu_pct"]) is not None]
        machine = [row for row in rows if row["kind"] == "maquina"]

        per_node: Dict[str, List[float]] = {}
        for row in nodes:
            per_node.setdefault(row["name"], []).append(float(row["cpu_pct"]))
        hottest = max(per_node.items(), key=lambda item: sum(item[1]) / len(item[1]), default=None)

        client_total: Dict[str, float] = {}
        for row in clients:
            client_total[row["time"]] = client_total.get(row["time"], 0.0) + float(row["cpu_pct"])

        last_time = max((row["time"] for row in rows if row["kind"] == "no"), default=None)
        node_memory = [
            float(row["mem_mb"]) for row in rows
            if row["kind"] == "no" and row["time"] == last_time and number(row["mem_mb"]) is not None
        ]

        summary[phase] = {
            "samples": len({row["time"] for row in rows}),
            "client_cpu_pct": _stats([float(row["cpu_pct"]) for row in clients]),
            "client_total_cpu_pct": _stats(list(client_total.values())),
            "node_cpu_pct": _stats([float(row["cpu_pct"]) for row in nodes]),
            "hottest_node": (
                {"name": hottest[0], "avg_cpu_pct": round(sum(hottest[1]) / len(hottest[1]), 1),
                 "max_cpu_pct": round(max(hottest[1]), 1)}
                if hottest else None
            ),
            "node_mem_mb": _stats(node_memory),
            "nodes_mem_total_mb": round(sum(node_memory), 1) if node_memory else None,
            "machine_cpu_pct": _stats([float(row["cpu_pct"]) for row in machine if number(row["cpu_pct"]) is not None]),
            "machine_load1_max": max((float(row["load1"]) for row in machine if number(row["load1"]) is not None), default=None),
            "machine_mem_used_mb_max": max((float(row["mem_mb"]) for row in machine if number(row["mem_mb"]) is not None), default=None),
        }
    return summary


if __name__ == "__main__":  # resumo de um resources.csv ja gravado
    import sys

    print(json.dumps(summarize(sys.argv[1]), indent=2, ensure_ascii=False))
