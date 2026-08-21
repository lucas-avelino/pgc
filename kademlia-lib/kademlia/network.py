"""
Package for interacting on the network at a high level.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import pickle
import random
import time

from .crawling import NodeSpiderCrawl, ProximityValueSpiderCrawl, ValueSpiderCrawl
from .node import Node
from .protocol import KademliaProtocol
from .storage import ForgetfulStorage
from .utils import TraceTimer

log = logging.getLogger(__name__)


def _normalize_key(key):
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return bytes.fromhex(key)
    raise TypeError("Key must be bytes or a hex string")


class Server:
    """
    High level view of a node instance.  This is the object that should be
    created to start listening as an active node on the network.
    """

    protocol_class = KademliaProtocol

    def __init__(
        self,
        ksize=20,
        replication_factor=3,
        alpha=3,
        node_id: bytes = None,
        key_size_bits=128,
        storage=None,
        hamming_workers=4,
        hamming_max_concurrency=8,
    ):
        """
        Create a server instance.  This will start listening on the given port.

        Args:
            ksize (int): The k parameter from the paper
            alpha (int): The alpha parameter from the paper
            node_id: The id for this node on the network.
            storage: An instance that implements the interface
                     :class:`~kademlia.storage.IStorage`
            replication_factor (int): The replication factor from the paper
            key_size_bits (int): The number of bits in the key space
        """
        if replication_factor < 1:
            raise ValueError("replication_factor must be at least 1")

        self.ksize = ksize
        self.replication_factor = replication_factor
        self.alpha = alpha
        self.storage = storage or ForgetfulStorage()
        self.key_size_bits = key_size_bits
        self.node = Node(node_id or random.getrandbits(key_size_bits).to_bytes(key_size_bits // 8, "big"))
        self.transport = None
        self.protocol = None
        self.refresh_loop = None
        self.save_state_loop = None
        self._hamming_executor = ThreadPoolExecutor(
            max_workers=hamming_workers,
            thread_name_prefix="kademlia-hamming",
        )
        self._hamming_sem = asyncio.Semaphore(hamming_max_concurrency)

    def stop(self):
        if self.transport is not None:
            self.transport.close()

        if self.refresh_loop:
            self.refresh_loop.cancel()

        if self.save_state_loop:
            self.save_state_loop.cancel()

        self._hamming_executor.shutdown(wait=False, cancel_futures=True)

    def _create_protocol(self):
        protocol = self.protocol_class(self.node, self.storage, self.ksize)
        protocol.replication_factor = self.replication_factor
        protocol.hamming_remote_executor = self._hamming_executor
        protocol.hamming_remote_sem = self._hamming_sem
        return protocol

    async def listen(self, port, interface="0.0.0.0"):
        """
        Start listening on the given port.

        Provide interface="::" to accept ipv6 address
        """
        loop = asyncio.get_event_loop()
        listen = loop.create_datagram_endpoint(
            self._create_protocol, local_addr=(interface, port)
        )
        log.info("Node %i listening on %s:%i", self.node.long_id, interface, port)
        self.transport, self.protocol = await listen
        # finally, schedule refreshing table
        self.refresh_table()

    def refresh_table(self, interval=3600):
        log.debug("Refreshing routing table")
        asyncio.ensure_future(self._refresh_table())
        loop = asyncio.get_event_loop()
        self.refresh_loop = loop.call_later(interval, self.refresh_table)

    async def _refresh_table(self):
        """
        Refresh buckets that haven't had any lookups in the last hour
        (per section 2.3 of the paper).
        """
        results = []
        for node_id in self.protocol.get_refresh_ids():
            node = Node(node_id)
            nearest = self.protocol.router.find_neighbors(node, self.alpha)
            spider = NodeSpiderCrawl(
                self.protocol, node, nearest, self.ksize, self.alpha
            )
            results.append(spider.find())

        # do our crawling
        await asyncio.gather(*results)

        # now republish keys older than one hour
        for dkey, value in self.storage.iter_older_than(3600):
            await self.set_digest(dkey, value)

    def bootstrappable_neighbors(self):
        """
        Get a :class:`list` of (ip, port) :class:`tuple` pairs suitable for
        use as an argument to the bootstrap method.

        The server should have been bootstrapped
        already - this is just a utility for getting some neighbors and then
        storing them if this server is going down for a while.  When it comes
        back up, the list of nodes can be used to bootstrap.
        """
        neighbors = self.protocol.router.find_neighbors(self.node)
        return [tuple(n)[-2:] for n in neighbors]

    async def bootstrap(self, addrs):
        """
        Bootstrap the server by connecting to other known nodes in the network.

        Args:
            addrs: A `list` of (ip, port) `tuple` pairs.  Note that only IP
                   addresses are acceptable - hostnames will cause an error.
        """
        log.debug("Attempting to bootstrap node with %i initial contacts", len(addrs))
        cos = list(map(self.bootstrap_node, addrs))
        gathered = await asyncio.gather(*cos)
        nodes = [node for node in gathered if node is not None]
        spider = NodeSpiderCrawl(
            self.protocol, self.node, nodes, self.ksize, self.alpha
        )
        return await spider.find()

    async def bootstrap_node(self, addr):
        result = await self.protocol.ping(addr, self.node.id)
        return Node(result[1], addr[0], addr[1]) if result[0] else None

    async def get(self, key):
        """
        Get a key if the network has it.

        Args:
            key: The key to look up.

        Returns:
            :class:`None` if not found, the value otherwise.
        """
        log.info("Looking up key %s", key)
        dkey = _normalize_key(key)
        # if this node has it, return it
        if self.storage.get(dkey) is not None:
            return self.storage.get(dkey)
        node = Node(dkey)
        nearest = self.protocol.router.find_neighbors(node)
        if not nearest:
            log.warning("There are no known neighbors to get key %s", key)
            return None
        spider = ValueSpiderCrawl(self.protocol, node, nearest, self.ksize, self.alpha)
        return await spider.find()

    async def get_hamming(self, key, r, trace=False):
        """
        Get nearby keys from the network by Hamming distance.

        Args:
            key: The key to look up.
            r: The Hamming distance radius to search within.

        Returns:
            A list of (key_hex, value, distance) results.
        """
        log.info("Looking up key %s with Hamming distance %d", key, r)
        timer = TraceTimer(enabled=trace, node_id=self.node.id)

        start = time.perf_counter()
        dkey = _normalize_key(key)
        timer.step("network.normalize_key", start)

        start = time.perf_counter()
        local_results = []
        if hasattr(self.storage, "get_by_hamming"):
            loop = asyncio.get_running_loop()
            async with self._hamming_sem:
                raw_local_results = await loop.run_in_executor(
                    self._hamming_executor,
                    self.storage.get_by_hamming,
                    dkey,
                    r,
                    timer if trace else None,
                    "local_mih",
                )
            local_results = [
                (found_key.hex(), value, distance)
                for found_key, value, distance in raw_local_results
            ]
        timer.step("network.local_hamming", start, results=len(local_results))

        start = time.perf_counter()
        node = Node(dkey)
        nearest = self.protocol.router.find_neighbors(node)
        timer.step("network.find_neighbors", start, neighbors=len(nearest))
        if not nearest:
            log.warning("There are no known neighbors to get key %s", key)
            if trace:
                return local_results, timer.finish(
                    local_results=len(local_results),
                    remote_results=0,
                    total_results=len(local_results),
                )
            return local_results

        spider = ProximityValueSpiderCrawl(
            self.protocol,
            node,
            nearest,
            self.ksize,
            self.alpha,
            r,
            trace=timer if trace else None,
        )
        start = time.perf_counter()
        remote_results = await spider.find()
        timer.step("network.remote_hamming", start, results=len(remote_results))

        start = time.perf_counter()
        deduped = {}
        for found_key, value, distance in local_results + remote_results:
            result_key = (found_key, value)
            current = deduped.get(result_key)
            if current is None or distance < current[1]:
                deduped[result_key] = (value, distance)

        results = [
            (found_key, value, distance)
            for (found_key, _), (value, distance) in sorted(
                deduped.items(), key=lambda item: item[1][1]
            )
        ]
        timer.step(
            "network.dedupe_and_sort",
            start,
            local_results=len(local_results),
            remote_results=len(remote_results),
            total_results=len(results),
        )
        if trace:
            return results, timer.finish(
                local_results=len(local_results),
                remote_results=len(remote_results),
                total_results=len(results),
            )
        return results

    async def set(self, key, value):
        """
        Set the given key to the given value in the network.
        """
        if not check_dht_value_type(value):
            raise TypeError("Value must be of type int, float, bool, str, or bytes")
        log.info("setting '%s' = '%s' on network", key, value)
        dkey = _normalize_key(key)
        return await self.set_digest(dkey, value)

    async def set_digest(self, dkey: bytes, value):
        """
        Set the given SHA1 digest key (bytes) to the given value in the
        network.
        """
        node = Node(dkey)

        nearest = self.protocol.router.find_neighbors(node)
        if not nearest:
            log.warning("There are no known neighbors to set key %s", dkey.hex())
            return False

        spider = NodeSpiderCrawl(self.protocol, node, nearest, self.ksize, self.alpha)
        nodes = sorted(await spider.find(), key=lambda n: n.distance_to(node))
        storage_nodes = nodes[: self.replication_factor]

        # If this node is close too, count it as one of the replicas.
        if (
            len(storage_nodes) < self.replication_factor
            or self.node.distance_to(node) < storage_nodes[-1].distance_to(node)
        ):
            self.storage[dkey] = value
            storage_nodes = sorted(
                storage_nodes + [self.node], key=lambda n: n.distance_to(node)
            )[: self.replication_factor]
            nodes = [n for n in storage_nodes if n.id != self.node.id]
        else:
            nodes = storage_nodes

        log.info("setting '%s' on %s", dkey.hex(), list(map(str, nodes)))

        results = [self.protocol.call_store(n, dkey, value) for n in nodes]
        remote_stored = any(await asyncio.gather(*results))
        # return true only if at least one store call succeeded
        stored_value = self.storage.get(dkey)
        stored_locally = (
            value in stored_value if isinstance(stored_value, list)
            else stored_value == value
        )
        return stored_locally or remote_stored

    def save_state(self, fname):
        """
        Save the state of this node (the alpha/ksize/id/immediate neighbors)
        to a cache file with the given fname.
        """
        log.info("Saving state to %s", fname)
        data = {
            "ksize": self.ksize,
            "replication_factor": self.replication_factor,
            "alpha": self.alpha,
            "id": self.node.id,
            "neighbors": self.bootstrappable_neighbors(),
        }
        if not data["neighbors"]:
            log.warning("No known neighbors, so not writing to cache.")
            return
        with open(fname, "wb") as file:
            pickle.dump(data, file)

    @classmethod
    async def load_state(cls, fname, port, interface="0.0.0.0"):
        """
        Load the state of this node (the alpha/ksize/id/immediate neighbors)
        from a cache file with the given fname and then bootstrap the node
        (using the given port/interface to start listening/bootstrapping).
        """
        log.info("Loading state from %s", fname)
        with open(fname, "rb") as file:
            data = pickle.load(file)
        svr = cls(
            ksize=data["ksize"],
            replication_factor=data.get("replication_factor", 3),
            alpha=data["alpha"],
            node_id=data["id"],
        )
        await svr.listen(port, interface)
        if data["neighbors"]:
            await svr.bootstrap(data["neighbors"])
        return svr

    def save_state_regularly(self, fname, frequency=600):
        """
        Save the state of node with a given regularity to the given
        filename.

        Args:
            fname: File name to save retularly to
            frequency: Frequency in seconds that the state should be saved.
                        By default, 10 minutes.
        """
        self.save_state(fname)
        loop = asyncio.get_event_loop()
        self.save_state_loop = loop.call_later(
            frequency, self.save_state_regularly, fname, frequency
        )


def check_dht_value_type(value):
    """
    Checks to see if the type of the value is a valid type for
    placing in the dht.
    """
    typeset = [int, float, bool, str, bytes]
    return type(value) in typeset
