"""Hospital sintético anonimizado — prontuários, exames, protocolos."""

from .db import DB_PATH, connect, init_db

__all__ = ["DB_PATH", "connect", "init_db"]
