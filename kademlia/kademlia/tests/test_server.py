import asyncio

import pytest

from kademlia.network import Server
from kademlia.node import Node
from kademlia.protocol import KademliaProtocol


@pytest.mark.asyncio
async def test_storing(bootstrap_node):
    server = Server()
    await server.listen(bootstrap_node[1] + 1)
    await server.bootstrap([bootstrap_node])
    await server.set("key", "value")
    result = await server.get("key")

    assert result == "value"

    server.stop()


@pytest.mark.asyncio
async def test_set_digest_uses_replication_factor(monkeypatch):
    key = (0).to_bytes(20, byteorder="big")
    server = Server(replication_factor=2, node_id=(2).to_bytes(20, byteorder="big"))

    nodes = [
        Node((1).to_bytes(20, byteorder="big"), "127.0.0.1", 8001),
        Node((3).to_bytes(20, byteorder="big"), "127.0.0.1", 8003),
        Node((4).to_bytes(20, byteorder="big"), "127.0.0.1", 8004),
    ]
    stored_on = []

    class FakeRouter:
        def find_neighbors(self, node):
            return nodes

    class FakeProtocol:
        router = FakeRouter()

        async def call_store(self, node, dkey, value):
            stored_on.append(node)
            return True

    class FakeSpider:
        def __init__(self, protocol, node, nearest, ksize, alpha):
            pass

        async def find(self):
            return nodes

    monkeypatch.setattr("kademlia.network.NodeSpiderCrawl", FakeSpider)
    server.protocol = FakeProtocol()

    result = await server.set_digest(key, "value")

    assert result is True
    assert server.storage.get(key) == "value"
    assert [node.port for node in stored_on] == [8001]


class TestSwappableProtocol:
    def test_default_protocol(self):
        """
        An ordinary Server object will initially not have a protocol, but will
        have a KademliaProtocol object as its protocol after its listen()
        method is called.
        """
        loop = asyncio.get_event_loop()
        server = Server()
        assert server.protocol is None
        loop.run_until_complete(server.listen(8469))
        assert isinstance(server.protocol, KademliaProtocol)
        server.stop()

    def test_custom_protocol(self):
        """
        A subclass of Server which overrides the protocol_class attribute will
        have an instance of that class as its protocol after its listen()
        method is called.
        """

        # Make a custom Protocol and Server to go with hit.
        class CoconutProtocol(KademliaProtocol):
            pass

        class HuskServer(Server):
            protocol_class = CoconutProtocol

        # An ordinary server does NOT have a CoconutProtocol as its protocol...
        loop = asyncio.get_event_loop()
        server = Server()
        loop.run_until_complete(server.listen(8469))
        assert not isinstance(server.protocol, CoconutProtocol)
        server.stop()

        # ...but our custom server does.
        husk_server = HuskServer()
        loop.run_until_complete(husk_server.listen(8469))
        assert isinstance(husk_server.protocol, CoconutProtocol)
        husk_server.stop()
