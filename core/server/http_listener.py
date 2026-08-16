from aiohttp import web
import kademlia

class NodeHTTPListener:
    async def __init__(self, port: int, server: kademlia.network.Server):
      self.server = server

      app = web.Application()

      app.router.add_get("/get", self.get)
      app.router.add_get("/get_near", self.get_near)
      app.router.add_post("/put", self.put)
  
      # Use AppRunner to avoid blocking the event loop
      runner = web.AppRunner(app)
      await runner.setup()
      site = web.TCPSite(runner, '0.0.0.0', port)
      await site.start()
      print(f"HTTP server started successfully on port {port}")


    async def get(self, request):
      try:
          key_hex = request.query.get('key')
          if not key_hex:
              return web.json_response({'error': 'Missing key parameter'}, status=400)

          key = bytes.fromhex(key_hex)
          
          # Check local storage first
          value = self.server.storage.get(key)
          if value is not None:
              print(f"Found key in local storage: {key_hex}")
              return web.json_response({'key': key_hex, 'value': value})
          
          # Try to get from DHT network
          value = await self.server.get(key_hex)
          if value is None:
              print(f"Key not found in DHT either: {key_hex}")
              return web.json_response({'error': 'Key not found'}, status=404)
          
          return web.json_response({'key': key_hex, 'value': value})
      except ValueError as e:
          return web.json_response({'error': f'Invalid hex key: {str(e)}'}, status=400)
      except Exception as e:
          print(f"Error in HTTP GET: {e}")
          return web.json_response({'error': str(e)}, status=500)

    async def get_near(self, request):
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
                results, trace = await self.server.get_hamming(key_hex, r, trace=True)
            else:
                results = await self.server.get_hamming(key_hex, r)
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
    
    async def put(self, request):
        try:
            data = await request.json()
            key_hex = data.get('key')
            value = data.get('value')
            
            if not key_hex or value is None:
                return web.json_response({'error': 'Missing key or value'}, status=400)
            
            key = bytes.fromhex(key_hex)
            
            # Store in DHT
            await self.server.set(key, value)
            
            return web.json_response({'status': 'ok', 'key': key_hex})
        except ValueError as e:
            return web.json_response({'error': f'Invalid hex key: {str(e)}'}, status=400)
        except Exception as e:
            print(f"Error in HTTP PUT: {e}")
            return web.json_response({'error': str(e)}, status=500)
    