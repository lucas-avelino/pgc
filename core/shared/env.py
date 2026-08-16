import os
import socket


class ConfigNames:
  BOOTSTRAP_ADDR = 'BOOTSTRAP_ADDR'
  LISTEN_PORT = 'LISTEN_PORT'
  AUTOCALC_PORT = 'AUTOCALC_PORT'
  K_SIZE = 'K_SIZE'
  METRICS_REPORT_INTERVAL = 'METRICS_REPORT_INTERVAL'
  REPLICATION_FACTOR = 'REPLICATION_FACTOR'
  ENABLE_HTTP = 'ENABLE_HTTP'
  PERSIST_PATH = 'PERSIST_PATH'
  PERSIST_SAVE_INTERVAL = 'PERSIST_SAVE_INTERVAL'
  BOOTSTRAP_HEAL_INTERVAL = 'BOOTSTRAP_HEAL_INTERVAL'
  KEY_SIZE = 'KEY_SIZE'

def _get_bool(name: str, default: bool = False) -> bool:
  value = os.getenv(name)
  if value is None:
    return default
  return value.lower() in {'1', 'true', 'yes', 'on'}


def _parse_addr(value: str | None) -> tuple[str, int] | None:
  if not value:
    return None

  host, separator, port = value.rpartition(':')
  if not separator or not host or not port:
    raise ValueError(f"{ConfigNames.BOOTSTRAP_ADDR} must be in host:port format")

  return (socket.gethostbyname(host), int(port))


class Vars:
  BOOTSTRAP_ADDR: tuple[str, int] | None = _parse_addr(os.getenv(ConfigNames.BOOTSTRAP_ADDR))
  LISTEN_PORT: int = int(os.getenv(ConfigNames.LISTEN_PORT) or '4001' )
  AUTOCALC_PORT: bool = _get_bool(ConfigNames.AUTOCALC_PORT)
  K_SIZE: int = int(os.getenv(ConfigNames.K_SIZE) or '20')
  METRICS_REPORT_INTERVAL: int = int(os.getenv(ConfigNames.METRICS_REPORT_INTERVAL) or '30')
  REPLICATION_FACTOR: int = int(os.getenv(ConfigNames.REPLICATION_FACTOR) or '3')
  ENABLE_HTTP: bool = _get_bool(ConfigNames.ENABLE_HTTP, default=True)
  PERSIST_PATH: str | None = os.getenv(ConfigNames.PERSIST_PATH)
  PERSIST_SAVE_INTERVAL: int = int(os.getenv(ConfigNames.PERSIST_SAVE_INTERVAL) or '5')
  BOOTSTRAP_HEAL_INTERVAL: int = int(os.getenv(ConfigNames.BOOTSTRAP_HEAL_INTERVAL) or '10')
  KEY_SIZE: int = 64
