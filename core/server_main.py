import asyncio

from kademlia.network import Server

from shared import env
from server.state import states

port = -1

async def main():
    state = states.State(key_size_bits=KEY_SIZE_BITS, persist_path=env.Vars.PERSIST_PATH)
    state.save_state()  # Save state immediately after loading or creating it
    
    server = Server(
        ksize=env.Vars.K_SIZE,
        replication_factor=env.Vars.REPLICATION_FACTOR,
        node_id=state.node_id,
        storage=state.storage,
        key_size_bits=KEY_SIZE_BITS
    )

    port = await listen(server)

    metric_task = None
    http_task = None
    persist_task = None
    heal_task = None

    print(f"Metrics report interval: {env.Vars.METRICS_REPORT_INTERVAL}")
    print(f"Node id: {server.node.id.hex()}")
    try:
        if env.Vars.BOOTSTRAP_ADDR:
            await try_bootstrap(server)
            heal_task = asyncio.create_task(heal_bootstrap(server))
        if env.Vars.METRICS_REPORT_INTERVAL > 0:
            metric_task = asyncio.create_task(report_metrics(server))
        if env.Vars.ENABLE_HTTP:
            http_task = asyncio.create_task(public_listen(port + 1000, server))
        if env.Vars.PERSIST_PATH:
            state.save_state()
            persist_task = asyncio.create_task(persist_regularly(state))

        print(f"Server listening on port {port}")
        await asyncio.Event().wait()
    finally:
        if metric_task:
            metric_task.cancel()
        if http_task:
            http_task.cancel()
        if persist_task:
            persist_task.cancel()
        if heal_task:
            heal_task.cancel()
        if env.Vars.PERSIST_PATH:
            state.save_state()
        server.stop()

async def listen(server: Server) -> int:
    port = env.Vars.LISTEN_PORT
    while True:
        try:
            await server.listen(port)
            return port
        except OSError:
            if not env.Vars.AUTOCALC_PORT:
                raise
            port += 1
