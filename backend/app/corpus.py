from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.models import ChunkRecord, DocumentSource


def load_sources(path: Path) -> list[DocumentSource]:
    payload = json.loads(path.read_text())
    return [DocumentSource.model_validate(item) for item in payload]


def load_demo_chunks(path: Path) -> list[ChunkRecord]:
    payload = json.loads(path.read_text())
    return [ChunkRecord.model_validate(item) for item in payload]


def load_normalized_chunks(root: Path) -> list[ChunkRecord]:
    if not root.exists():
        return []
    chunks: list[ChunkRecord] = []
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped:
                chunks.append(ChunkRecord.model_validate_json(stripped))
    return chunks


def load_corpus_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def group_sources_by_family(sources: list[DocumentSource]) -> dict[str, list[DocumentSource]]:
    grouped: dict[str, list[DocumentSource]] = defaultdict(list)
    for source in sources:
        grouped[source.doc_family].append(source)
    return dict(grouped)
