import asyncio
from kademlia.network import Server
from core.shared import env

async def heal_bootstrap(server: Server, try_bootstrap):
    while True:
        await asyncio.sleep(env.Vars.BOOTSTRAP_HEAL_INTERVAL)
        if server.bootstrappable_neighbors():
            continue

        print(
            "Routing table is empty; trying to heal via "
            f"{env.Vars.BOOTSTRAP_ADDR}"
        )
        await try_bootstrap(server)