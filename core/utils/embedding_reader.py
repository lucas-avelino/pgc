
def embedding_file_to_keys(file: str, key_size_bits: int) -> list[str]:
    keys = []
    chunk_size = key_size_bits // 8
    with open(file, "rb") as f:
        data = f.read()
        for i in range(0, len(data), chunk_size):
            keys.append(data[i:i + chunk_size].hex())
    return keys
