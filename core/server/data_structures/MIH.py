import operator
import time
from itertools import takewhile
from collections import OrderedDict, defaultdict

from kademlia.storage import IStorage

from core.shared import env

KEY_SIZE = env.Vars.KEY_SIZE

class MIH_Storage(IStorage):
    def __init__(self, ttl=604800, m=4):
        self.data = OrderedDict()
        self.ttl = ttl
        self.m = m
        self.pieces = [defaultdict(set) for _ in range(m)]

        base, rem = divmod(KEY_SIZE, m)
        self._piece_sizes = [base + 1 if i < rem else base for i in range(m)]
        self._piece_offsets = []
        acc = 0
        for size in self._piece_sizes:
            self._piece_offsets.append(acc)
            acc += size 

    def shuffle_bits(self, key: bytes):
        total_bits = len(key) * 8
        bits = [0] * total_bits
        read = 0
        for byte in key:
            for i in range(7, -1, -1):
                bucket = read % self.m
                current = read // self.m
                pos = self._piece_offsets[bucket] + current
                bits[pos] = (byte >> i) & 1
                read += 1

        out = bytearray(len(key))
        for idx, bit in enumerate(bits):
            if bit:
                out[idx // 8] |= 1 << (7 - idx % 8)
        return bytes(out)


    def _split(self, key: bytes):
        total_bits = len(key) * 8
        value = int.from_bytes(key, "big")
        pieces = []
        offset_from_end = total_bits
        for size in self._piece_sizes:
            offset_from_end -= size
            mask = (1 << size) - 1
            pieces.append((value >> offset_from_end) & mask)
        return pieces

    def _index(self, key: bytes):
        for i, piece in enumerate(self._split(key)):
            self.pieces[i][piece].add(key)

    def _deindex(self, key: bytes):
        for i, piece in enumerate(self._split(key)):
            bucket = self.pieces[i].get(piece)
            if bucket is not None:
                bucket.discard(key)
                if not bucket:
                    del self.pieces[i][piece]

    def __setitem__(self, key, value):
        key = self.shuffle_bits(key)
        is_new = key not in self.data
        if key in self.data:
            _, values = self.data.pop(key)
            if value not in values:
                values.append(value)
        else:
            values = [value]
        self.data[key] = (time.monotonic(), values)
        if is_new:
            self._index(key)
        

    def snapshot(self):
        now = time.monotonic()
        return {
            "ttl": self.ttl,
            "m": self.m,
            "entries": [
                {
                    "key": key,
                    "age": max(0, now - birthday),
                    "values": values,
                }
                for key, (birthday, values) in self.data.items()
            ],
        }

    @classmethod
    def from_snapshot(cls, snapshot, reset_age=False):
        storage = cls(ttl=snapshot.get("ttl", 604800), m=snapshot.get("m", 4))
        now = time.monotonic()
        for entry in snapshot.get("entries", []):
            key = entry["key"]
            values = entry.get("values", [])
            if not values:
                continue
            birthday = now if reset_age else now - entry.get("age", 0)
            storage.data[key] = (birthday, list(values))
            storage._index(key)
        storage.cull()
        return storage

    def cull(self):
        for key, _ in self.iter_older_than(self.ttl):
            self._deindex(key)
            if key in self.data:
                del self.data[key]

    def get(self, key, default=None):
        key = self.shuffle_bits(key)
        
        if key in self.data:
            return self[key]
        return default

    def __getitem__(self, key):
        
        return self.data[key][1]

    def get_by_hamming(self, query: bytes, r: int, trace=None, trace_prefix="mih"):
        query = self.shuffle_bits(query)
        if r < 0:
            raise ValueError("r must be >= 0")

        start = time.perf_counter()
        
        if trace is not None:
            trace.step(f"{trace_prefix}.cull", start, entries=len(self.data))

        start = time.perf_counter()
        if r >= self.m:
            candidates = set(self.data.keys())
            candidate_source = "full_scan"
        else:
            candidates = set()
            for i, piece in enumerate(self._split(query)):
                candidates |= self.pieces[i].get(piece, set())
            candidate_source = "mih_buckets"
        if trace is not None:
            trace.step(
                f"{trace_prefix}.candidate_selection",
                start,
                source=candidate_source,
                candidates=len(candidates),
                partitions=self.m,
            )

        start = time.perf_counter()
        results = {}
        comparisons = 0
        for key in candidates:
            entry = self.data.get(key)
            if entry is None:
                continue
            comparisons += 1
            d = self._hamming(query, key)
            if d <= r:
                current = results.get(key)
                if current is None or d < current[1]:
                    results[key] = (entry[1], d)
        if trace is not None:
            trace.step(
                f"{trace_prefix}.hamming_filter",
                start,
                comparisons=comparisons,
                matched_keys=len(results),
            )

        start = time.perf_counter()
        output = []
        for key, (values, distance) in sorted(
            results.items(), key=lambda item: item[1][1]
        ):
            output.extend((key, value, distance) for value in values)
        if trace is not None:
            trace.step(f"{trace_prefix}.expand_values", start, results=len(output))
        return output

    @staticmethod
    def _hamming(a: bytes, b: bytes) -> int:
        assert len(a) == len(b), "keys must have the same size"
        return sum((x ^ y).bit_count() for x, y in zip(a, b))

    def __repr__(self):
        
        return repr(self.data)

    def iter_older_than(self, seconds_old):
        min_birthday = time.monotonic() - seconds_old
        zipped = self._triple_iter()
        matches = takewhile(lambda r: min_birthday >= r[1], zipped)
        return [
            (key, value)
            for key, _, values in matches
            for value in values
        ]

    def _triple_iter(self):
        ikeys = self.data.keys()
        ibirthday = map(operator.itemgetter(0), self.data.values())
        ivalues = map(operator.itemgetter(1), self.data.values())
        return zip(ikeys, ibirthday, ivalues)

    def __iter__(self):
        
        return (
            (key, value)
            for key, (_, values) in self.data.items()
            for value in values
        )
