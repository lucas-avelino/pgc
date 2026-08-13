"""
General catchall for functions that don't make sense as methods.
"""

import asyncio
import hashlib
import operator
import time


async def gather_dict(dic):
    cors = list(dic.values())
    results = await asyncio.gather(*cors)
    return dict(zip(dic.keys(), results))


class TraceTimer:
    def __init__(self, enabled=False, node_id=None):
        self.enabled = enabled
        self.started_at = time.perf_counter()
        self.events = []
        self.node_id = node_id.hex() if hasattr(node_id, "hex") else node_id

    def _event_name(self, name):
        if not self.node_id:
            return name
        return f"{self.node_id}.{name}"

    def record(self, name, duration_s, **data):
        if not self.enabled:
            return
        event = {
            "name": self._event_name(name),
            "duration_ms": round(duration_s * 1000, 3),
        }
        event.update(data)
        self.events.append(event)

    def step(self, name, start, **data):
        self.record(name, time.perf_counter() - start, **data)

    def finish(self, **data):
        if not self.enabled:
            return None
        trace = {
            "total_ms": round((time.perf_counter() - self.started_at) * 1000, 3),
            "events": self.events,
        }
        if self.node_id:
            trace["node_id"] = self.node_id
        trace.update(data)
        return trace

def shared_prefix(args):
    """
    Find the shared prefix between the strings.

    For instance:

        sharedPrefix(['blahblah', 'blahwhat'])

    returns 'blah'.
    """
    i = 0
    while i < min(map(len, args)):
        if len(set(map(operator.itemgetter(i), args))) != 1:
            break
        i += 1
    return args[0][:i]


def bytes_to_bit_string(bites):
    bits = [bin(bite)[2:].rjust(8, "0") for bite in bites]
    return "".join(bits)
