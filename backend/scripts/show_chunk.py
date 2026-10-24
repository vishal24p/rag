"""Print one indexed chunk by its database row ID."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg  # noqa: E402
from psycopg import sql  # noqa: E402

from src.config import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Show an indexed chunk by ID")
    parser.add_argument("chunk_id", type=int, help="Chunk row ID")
    args = parser.parse_args()

    with psycopg.connect(settings.db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT id, document_id, chunk_index, content FROM {} WHERE id = %s"
                ).format(sql.Identifier(settings.chunks_table)),
                (args.chunk_id,),
            )
            chunk = cur.fetchone()

    if chunk is None:
        print(f"Chunk {args.chunk_id} not found", file=sys.stderr)
        return 1

    chunk_id, document_id, chunk_index, content = chunk
    print(f"Chunk ID: {chunk_id}")
    print(f"Document ID: {document_id}")
    print(f"Chunk index: {chunk_index}\n")
    print(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
