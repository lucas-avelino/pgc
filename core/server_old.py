import asyncio
import os
import pickle
import tempfile
import time
from kademlia.network import Server
import env
from data_structures.MIH import MIH_Storage
from state.states import State


KEY_SIZE_BITS = 64 
port = -1

async def main():
    state = State(key_size_bits=KEY_SIZE_BITS, persist_path=env.Vars.PERSIST_PATH)
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


def load_persisted_state(path: str | None):
    if not path:
        return None, MIH_Storage(m=64)

    state_file = os.path.join(path, PERSIST_STATE_FILE)
    if not os.path.exists(state_file):
        return None, MIH_Storage(m=64)

    with open(state_file, "rb") as file:
        state = pickle.load(file)

    node_id = state.get("node_id")
    storage = MIH_Storage.from_snapshot(state.get("storage", {}), reset_age=True)
    print(
        f"Loaded persisted state from {state_file}: "
        f"node_id={node_id.hex() if node_id else 'new'}, "
        f"{len(storage.data)} keys"
    )
    return node_id, storage


async def persist_regularly(state: State):
    while True:
        await asyncio.sleep(env.Vars.PERSIST_SAVE_INTERVAL)
        state.save_state()


async def try_bootstrap(server: Server) -> bool:
    try:
        neighbors = await server.bootstrap([env.Vars.BOOTSTRAP_ADDR])
    except Exception as exc:
        print(f"Bootstrap failed via {env.Vars.BOOTSTRAP_ADDR}: {exc}")
        return False

    print(
        f"Bootstrap via {env.Vars.BOOTSTRAP_ADDR} found "
        f"{len(neighbors)} neighbors"
    )
    return True


async def heal_bootstrap(server: Server):
    while True:
        await asyncio.sleep(env.Vars.BOOTSTRAP_HEAL_INTERVAL)
        if server.bootstrappable_neighbors():
            continue

        print(
            "Routing table is empty; trying to heal via "
            f"{env.Vars.BOOTSTRAP_ADDR}"
        )
        await try_bootstrap(server)


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


# Will listem for get and puts via TCP outside of the Kademlia protocol. This is for testing and debugging purposes.
async def public_listen(port: int, server: Server):
    from aiohttp import web
    print(f"Starting HTTP server on port {port}")
    app = web.Application()

    # key in hex format, value in string format
    async def get(request):
        try:
            key_hex = request.query.get('key')
            if not key_hex:
                return web.json_response({'error': 'Missing key parameter'}, status=400)

            key = bytes.fromhex(key_hex)
            
            # Check local storage first
            value = server.storage.get(key)
            if value is not None:
                print(f"Found key in local storage: {key_hex}")
                return web.json_response({'key': key_hex, 'value': value})
            
            # Try to get from DHT network
            value = await server.get(key_hex)
            if value is None:
                print(f"Key not found in DHT either: {key_hex}")
                return web.json_response({'error': 'Key not found'}, status=404)
            
            return web.json_response({'key': key_hex, 'value': value})
        except ValueError as e:
            return web.json_response({'error': f'Invalid hex key: {str(e)}'}, status=400)
        except Exception as e:
            print(f"Error in HTTP GET: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_near(request):
        try:
            request_start = time.perf_counter()
            key_hex = request.query.get('key')
            radius = request.query.get('r')
            trace_enabled = request.query.get('trace', '').lower() in ('1', 'true', 'yes', 'on')
            if not key_hex:
                return web.json_response({'error': 'Missing key parameter'}, status=400)
            if radius is None:
                return web.json_response({'error': 'Missing r parameter'}, status=400)

            r = int(radius)
            bytes.fromhex(key_hex)

            if trace_enabled:
                results, trace = await server.get_hamming(key_hex, r, trace=True)
            else:
                results = await server.get_hamming(key_hex, r)
                trace = None

            response = {
                'key': key_hex,
                'r': r,
                'results': [
                    {'key': key, 'value': value, 'distance': distance}
                    for key, value, distance in results
                ],
            }
            if trace_enabled:
                trace['http_total_ms'] = round((time.perf_counter() - request_start) * 1000, 3)
                response['trace'] = trace
            return web.json_response(response)
        except ValueError as e:
            return web.json_response({'error': f'Invalid key or radius: {str(e)}'}, status=400)
        except Exception as e:
            print(f"Error in HTTP GET_NEAR: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def put(request):
        try:
            data = await request.json()
            key_hex = data.get('key')
            value = data.get('value')
            
            if not key_hex or value is None:
                return web.json_response({'error': 'Missing key or value'}, status=400)
            
            key = bytes.fromhex(key_hex)
            
            # Store in DHT
            await server.set(key, value)
            
            return web.json_response({'status': 'ok', 'key': key_hex})
        except ValueError as e:
            return web.json_response({'error': f'Invalid hex key: {str(e)}'}, status=400)
        except Exception as e:
            print(f"Error in HTTP PUT: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    app.router.add_get("/get", get)
    app.router.add_get("/get_near", get_near)
    app.router.add_post("/put", put)

    # Use AppRunner to avoid blocking the event loop
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    print(f"HTTP server started successfully on port {port}")



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down server...")
