import asyncio
import os
import pickle
import random

from core.shared import env
from core.server.data_structures.MIH import MIH_Storage
from core.server.state.slots import SlotResult, try_aquire_any_slot_or_create

class State:
    node_id: bytes
    storage: MIH_Storage
    slot_info: SlotResult

    def __init__(self, key_size_bits: int, persist_path: str = "./persist/", ):
        (is_new, slot_info) = try_aquire_any_slot_or_create(persist_path)
        if slot_info is None:
            raise RuntimeError("No available slot for state persistence.")
        self.slot_info = slot_info
        if not is_new:
            try:
                old_state = pickle.load(self.slot_info.file)
                node_id = old_state.get("node_id")
                if node_id is None:
                    print("No node ID found in the loaded state.")
                    is_new = True
                storage = MIH_Storage.from_snapshot(old_state.get("storage", {}), reset_age=True)
                print(
                    f"Loaded state from slot {slot_info.path}: "
                    f"node_id={node_id.hex() if node_id else 'new'}, "
                    f"{len(storage.data)} keys"
                )
            except Exception as e:
                print(f"Error loading state from slot, will create a new one, error: {slot_info.path}: {e}")
                is_new = True                   
        
        if is_new:
            print("No available slot for state persistence. Starting with a new state.")
            node_id = random.getrandbits(key_size_bits).to_bytes(key_size_bits // 8, "big")
            storage = MIH_Storage(m=env.Vars.MIH_PARTITION_SIZE)

        print(f"Using slot {slot_info.path} for state persistence.")
        self.node_id = node_id
        self.storage = storage
        self.slot_info = slot_info

    def save_state(self):
        """
        Salva o estado em um arquivo persistente.
        """
        state = {
            "node_id": self.node_id,
            "storage": self.storage.snapshot(),
        }
        try:
            self.slot_info.file.seek(0)
            pickle.dump(state, self.slot_info.file)
            self.slot_info.file.truncate()
            self.slot_info.file.flush()
        except Exception:
            print(f"Error saving state to slot {self.slot_info.path}.")

async def persist_regularly(state: State):
    while True:
        await asyncio.sleep(env.Vars.PERSIST_SAVE_INTERVAL)
        state.save_state()