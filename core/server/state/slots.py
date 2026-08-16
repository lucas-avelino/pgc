import fcntl
from io import TextIOWrapper
import os
from typing import NamedTuple, Optional, Tuple

class SlotResult(NamedTuple):
    path: str
    file: TextIOWrapper

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
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return SlotResult(path=slot_path, file=lock_file)
    except BlockingIOError:
        lock_file.close()
        return None

def try_aquire_any_slot_or_create(path: str = "./persist/") -> Tuple[bool, SlotResult]:
    """
    Tenta adquirir qualquer slot disponível.
    Se não encontrar nenhum slot disponível, cria um novo.
    Retorna uma tupla com o caminho do slot e o arquivo de lock, ou None se não conseguiu.
    """
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
    

