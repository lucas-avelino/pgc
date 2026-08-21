import asyncio
import logging
import random
import time

from rpcudp.protocol import RPCProtocol

from .node import Node
from .routing import RoutingTable
from .utils import TraceTimer

log = logging.getLogger(__name__)


class KademliaProtocol(RPCProtocol):
    def __init__(self, source_node, storage, ksize, replication_factor=3):
        RPCProtocol.__init__(self)
        self.router = RoutingTable(self, ksize, source_node)
        self.storage = storage
        self.source_node = source_node
        self.replication_factor = replication_factor

    def get_refresh_ids(self):
        """
        Get ids to search for to keep old buckets up to date.
        """
        ids = []
        for bucket in self.router.lonely_buckets():
            rid = random.randint(*bucket.range).to_bytes(20, byteorder="big")
            ids.append(rid)
        return ids

    def rpc_stun(self, sender):
        return sender

    def rpc_ping(self, sender, nodeid):
        source = Node(nodeid, sender[0], sender[1])
        self.welcome_if_new(source)
        return self.source_node.id

    def rpc_store(self, sender, nodeid, key, value):
        source = Node(nodeid, sender[0], sender[1])
        self.welcome_if_new(source)
        log.debug(
            "got a store request from %s, storing '%s'='%s'", sender, key.hex(), value
        )
        self.storage[key] = value
        return True

    def rpc_find_node(self, sender, nodeid, key):
        log.info("finding neighbors of %i in local table", int(nodeid.hex(), 16))
        source = Node(nodeid, sender[0], sender[1])
        self.welcome_if_new(source)
        node = Node(key)
        neighbors = self.router.find_neighbors(node, exclude=source)
        return list(map(tuple, neighbors))

    def rpc_find_value(self, sender, nodeid, key):
        source = Node(nodeid, sender[0], sender[1])
        self.welcome_if_new(source)
        value = self.storage.get(key, None)
        if value is None:
            return self.rpc_find_node(sender, nodeid, key)
        return {"value": value}
    
    async def rpc_find_value_hamming(self, sender, nodeid, key, r, trace=False):
        source = Node(nodeid, sender[0], sender[1])
        self.welcome_if_new(source)
        timer = TraceTimer(enabled=trace, node_id=self.source_node.id)
        start = time.perf_counter()
        if not hasattr(self.storage, "get_by_hamming"):
            response = {"results": []}
            if trace:
                response["trace"] = {
                    "total_ms": round((time.perf_counter() - start) * 1000, 3),
                    "events": [],
                    "storage_support": False,
                }
            return response

        loop = asyncio.get_running_loop()
        async with self.hamming_remote_sem:
            results = await loop.run_in_executor(
                self.hamming_remote_executor,
                self.storage.get_by_hamming,
                key,
                r,
                timer if trace else None,
                "remote_mih",
            )

            elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
            response = {"results": [(k.hex(), v, d) for k, v, d in results]}
            if trace:
                response["trace"] = timer.finish(
                    storage_support=True,
                    results=len(results),
                )
                
                response["trace"]["total_ms"] = elapsed_ms

        return response

    async def call_find_node(self, node_to_ask, node_to_find):
        address = (node_to_ask.ip, node_to_ask.port)
        result = await self.find_node(address, self.source_node.id, node_to_find.id)
        return self.handle_call_response(result, node_to_ask)

    async def call_find_value(self, node_to_ask, node_to_find):
        address = (node_to_ask.ip, node_to_ask.port)
        result = await self.find_value(address, self.source_node.id, node_to_find.id)
        return self.handle_call_response(result, node_to_ask)

    async def call_find_value_hamming(self, node_to_ask, node_to_find, r, trace=False):
        address = (node_to_ask.ip, node_to_ask.port)
        result = await self.find_value_hamming(
            address, self.source_node.id, node_to_find.id, r, trace
        )
        return self.handle_call_response(result, node_to_ask)

    async def call_ping(self, node_to_ask):
        address = (node_to_ask.ip, node_to_ask.port)
        result = await self.ping(address, self.source_node.id)
        return self.handle_call_response(result, node_to_ask)

    async def call_store(self, node_to_ask, key, value):
        address = (node_to_ask.ip, node_to_ask.port)
        result = await self.store(address, self.source_node.id, key, value)
        return self.handle_call_response(result, node_to_ask)

    def welcome_if_new(self, node):
        """
        Given a new node, send it all the keys/values it should be storing,
        then add it to the routing table.

        @param node: A new node that just joined (or that we just found out
        about).

        Process:
        For each key in storage, get k closest nodes.  If newnode is closer
        than the furtherst in that list, and the node for this server
        is closer than the closest in that list, then store the key/value
        on the new node (per section 2.5 of the paper)
        """
        if not self.router.is_new_node(node):
            return

        log.info("never seen %s before, adding to router", node)
        self.router.add_contact(node)

    def handle_call_response(self, result, node):
        """
        If we get a response, add the node to the routing table.  If
        we get no response, make sure it's removed from the routing table.
        """
        if not result[0]:
            log.warning("no response from %s, removing from router", node)
            self.router.remove_contact(node)
            return result

        log.info("got successful response from %s", node)
        self.welcome_if_new(node)
        return result
