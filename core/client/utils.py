import json


def write_file(file_name: str, data: str) -> None:
    with open(file_name, "wb") as f:
        f.write(data.encode("utf-8"))

def write_json(file_name: str, data: dict) -> None:
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)