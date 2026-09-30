import operator
import time
from itertools import takewhile
from collections import defaultdict

from kademlia.storage import IStorage

from core.shared import env

KEY_SIZE = env.Vars.KEY_SIZE

def values_as_list(stored) -> list:
    return stored if type(stored) is list else [stored]

def compact_values(values):
    return values[0] if len(values) == 1 else list(values)

def round_robin_buckets(bucket_sizes):
    offsets, total = [], 0
    for size in bucket_sizes:
        offsets.append(total)
        total += size

    perm = [0] * total
    used = [0] * len(bucket_sizes)
    value = 0
    while value < total:
        for i, size in enumerate(bucket_sizes):
            if used[i] < size:
                perm[offsets[i] + used[i]] = value
                used[i] += 1
                value += 1
    return perm

class MIH_Storage(IStorage):
    def __init__(self, ttl=604800, m=4):
        self.data = {}
        self.ttl = ttl
        self.m = m
        self.pieces = [defaultdict(list) for _ in range(m)]

        base, rem = divmod(KEY_SIZE, m)
        self._piece_sizes = [base + 1 if i < rem else base for i in range(m)]
        self._piece_offsets = []
        acc = 0
        for size in self._piece_sizes:
            self._piece_offsets.append(acc)
            acc += size 
        self._permut = round_robin_buckets(self._piece_sizes)
        

    def shuffle_bits(self, key: bytes) -> bytes:
        perm = self._permut
        result = bytearray((len(perm) + 7) // 8)
        for out_pos, in_pos in enumerate(perm):
            if key[in_pos >> 3] & (1 << (7 - (in_pos & 7))):
                result[out_pos >> 3] |= 1 << (7 - (out_pos & 7))
        return bytes(result)



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
            self.pieces[i][piece].append(key)

    def _deindex(self, key: bytes):
        for i, piece in enumerate(self._split(key)):
            bucket = self.pieces[i].get(piece)
            if bucket is not None:
                try:
                    bucket.remove(key)
                except ValueError:
                    continue
                if not bucket:
                    del self.pieces[i][piece]

    def __setitem__(self, key, value):
        key = self.shuffle_bits(key)
        is_new = key not in self.data
        if is_new:
            stored = value
        else:
            _, stored = self.data.pop(key)
            if type(stored) is list:
                if value not in stored:
                    stored.append(value)
            elif stored != value:
                stored = [stored, value]
        self.data[key] = (time.monotonic(), stored)
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
                    "values": values_as_list(stored),
                }
                for key, (birthday, stored) in self.data.items()
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
            storage.data[key] = (birthday, compact_values(values))
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
        
        return values_as_list(self.data[key][1])

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
                candidates.update(self.pieces[i].get(piece, ()))
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
                    results[key] = (values_as_list(entry[1]), d)
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
        return (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).bit_count()

    def __repr__(self):
        
        return repr(self.data)

    def iter_older_than(self, seconds_old):
        min_birthday = time.monotonic() - seconds_old
        zipped = self._triple_iter()
        matches = takewhile(lambda r: min_birthday >= r[1], zipped)
        return [
            (key, value)
            for key, _, stored in matches
            for value in values_as_list(stored)
        ]

    def _triple_iter(self):
        ikeys = self.data.keys()
        ibirthday = map(operator.itemgetter(0), self.data.values())
        ivalues = map(operator.itemgetter(1), self.data.values())
        return zip(ikeys, ibirthday, ivalues)

    def __iter__(self):
        
        return (
            (key, value)
            for key, (_, stored) in self.data.items()
            for value in values_as_list(stored)
        )
