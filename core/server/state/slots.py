from io import TextIOWrapper
import os
from typing import NamedTuple, Optional, Tuple

try:
    import fcntl  # Unix-only
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None
    import msvcrt

class SlotResult(NamedTuple):
    path: str
    file: TextIOWrapper


def _try_lock(lock_file):
    if fcntl is not None:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return

    # Windows file locking requires locking an existing byte range.
    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)

def try_aquire_slot(path: str) -> Optional[SlotResult]:
    """
    Tenta adquirir um slot para armazenar um chunk.
    Retorna uma tupla com o caminho do slot e o arquivo de lock, ou None se não conseguiu.
    """
    slot_path = os.path.join(path, "state.pkl")

    # O_CREAT: cria se não existir | O_RDWR: leitura e escrita | (sem O_TRUNC: não apaga o conteúdo)
    fd = os.open(slot_path, os.O_CREAT | os.O_RDWR)
    lock_file = os.fdopen(fd, "r+b")

    try:
        _try_lock(lock_file)
        return SlotResult(path=slot_path, file=lock_file)
    except (BlockingIOError, OSError):
        lock_file.close()
        return None

def try_aquire_any_slot_or_create(path: str = "./persist/") -> Tuple[bool, SlotResult]:
    """
    Tenta adquirir qualquer slot disponível.
    Se não encontrar nenhum slot disponível, cria um novo.
    Retorna uma tupla com o caminho do slot e o arquivo de lock, ou None se não conseguiu.
    """
    os.makedirs(path, exist_ok=True)
    slots = os.listdir(path)
    for slot in slots:
        slot_path = os.path.join(path, slot)
        if os.path.isdir(slot_path):
            result = try_aquire_slot(slot_path)
            if result is not None:
                return (False, result)
            
    # Se não encontrou nenhum slot disponível, tenta criar um novo
    new_slot_path = os.path.join(path, f"slot_{len(slots)}")
    os.makedirs(new_slot_path, exist_ok=True)
    result = try_aquire_slot(new_slot_path)

    return (True, result)
    

