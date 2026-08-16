import asyncio

from kademlia.network import Server

from core.shared import env


async def report_metrics(server: Server):
    while True:
        await asyncio.sleep(env.Vars.METRICS_REPORT_INTERVAL)
        storage = server.storage
        if hasattr(storage, "cull"):
            storage.cull()

        data = getattr(storage, "data", {})
        key_count = len(data)
        value_count = sum(
            len(values) if isinstance(values, list) else 1
            for _, values in data.values()
        )
        duplicate_values = value_count - key_count

        node_id = server.node.id.hex()
        print(
            f"[{node_id}] Metrics report: "
            f"{key_count} keys, {value_count} values, "
            f"{duplicate_values} extra values on repeated keys"
        )

        pieces = getattr(storage, "pieces", None)
        if pieces:
            bucket_count = sum(len(piece) for piece in pieces)
            index_refs = sum(len(bucket) for piece in pieces for bucket in piece.values())
            avg_refs_per_partition = index_refs / len(pieces)
            avg_bucket_size = index_refs / bucket_count if bucket_count else 0
            max_bucket_size = max(
                (len(bucket) for piece in pieces for bucket in piece.values()),
                default=0,
            )
            print(
                f"[{node_id}] MIH index: "
                f"{len(pieces)} partitions, {bucket_count} occupied buckets, "
                f"{index_refs} key refs, "
                f"{avg_refs_per_partition:.2f} refs/partition, "
                f"{avg_bucket_size:.2f} avg bucket size, "
                f"{max_bucket_size} max bucket size"
            )

        await server.set(server.node.id, len(server.storage.data))