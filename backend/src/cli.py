# backend/src/cli.py
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def ingest(args: argparse.Namespace) -> None:
    """Ingest documents from a file or directory."""
    from backend.src.api.deps import get_ingestion_processor

    processor = get_ingestion_processor()
    path = Path(args.path)

    if path.is_file():
        doc_id = processor.process_file(path)
        logger.info("Ingested %s -> %s", path, doc_id)
    elif path.is_dir():
        count = processor.process_directory(path)
        logger.info("Ingested %d files from %s", count, path)
    else:
        logger.error("Path not found: %s", path)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="intellidocs",
        description="IntelliDocs CLI -- document ingestion and management",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the vector store")
    ingest_parser.add_argument(
        "path",
        type=str,
        help="Path to a file or directory to ingest",
    )
    ingest_parser.set_defaults(func=ingest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
