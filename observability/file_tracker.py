from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from config.settings import LOGS_DIR

CHUNK_TRACKING_FILE = LOGS_DIR / "chunked_files.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_tracking_payload() -> Dict[str, Any]:
    if not CHUNK_TRACKING_FILE.is_file():
        return {}

    try:
        payload = json.loads(CHUNK_TRACKING_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def build_chunk_tracking_entry(file_path: Path, source_docs: List[Any], processed_docs: List[Any]) -> Dict[str, Any]:
    extracted_characters = sum(len(doc.page_content) for doc in source_docs)
    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "pages_or_sections_extracted": len(source_docs),
        "characters_extracted": extracted_characters,
        "chunks_created": len(processed_docs),
        "tracked_at": _now_iso(),
    }


def build_embedding_tracking_entry(
    *,
    file_name: str,
    file_path: str,
    file_size_bytes: int,
    chunk_count: int,
    embedding_model: str,
    embedding_start_time: str,
    embedding_end_time: str,
    embedding_duration_ms: int,
    status: str,
) -> Dict[str, Any]:
    return {
        "file_name": file_name,
        "file_path": file_path,
        "file_size_bytes": file_size_bytes,
        "chunk_count": chunk_count,
        "embedding_model": embedding_model,
        "embedding_start_time": embedding_start_time,
        "embedding_end_time": embedding_end_time,
        "embedding_duration_ms": embedding_duration_ms,
        "status": status,
    }


def _write_tracking_payload(payload: Dict[str, Any]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNK_TRACKING_FILE, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)


def save_chunk_tracking_report(entries: List[Dict[str, Any]], status: str = "completed") -> None:
    existing = _load_tracking_payload()
    payload = {
        "generated_at": _now_iso(),
        "status": status,
        "file_count": len(entries),
        "total_characters_extracted": sum(entry.get("characters_extracted", 0) for entry in entries),
        "total_chunks_created": sum(entry.get("chunks_created", 0) for entry in entries),
        "files": entries,
        "embedding_files": existing.get("embedding_files", []),
    }
    _write_tracking_payload(payload)


def append_embedding_tracking_entry(entry: Dict[str, Any]) -> None:
    existing = _load_tracking_payload()
    embedding_files = existing.get("embedding_files", [])
    if not isinstance(embedding_files, list):
        embedding_files = []

    embedding_files = [item for item in embedding_files if isinstance(item, dict)]
    embedding_files.append(entry)

    payload = {
        "generated_at": existing.get("generated_at") or _now_iso(),
        "status": existing.get("status", "completed"),
        "file_count": existing.get("file_count", len(existing.get("files", [])) if isinstance(existing.get("files", []), list) else 0),
        "total_characters_extracted": existing.get("total_characters_extracted", 0),
        "total_chunks_created": existing.get("total_chunks_created", 0),
        "files": existing.get("files", []) if isinstance(existing.get("files", []), list) else [],
        "embedding_files": embedding_files,
    }
    _write_tracking_payload(payload)


def initialize_chunk_tracking_report() -> None:
    save_chunk_tracking_report([], status="started")
