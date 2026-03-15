from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup


def load_sources(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def fetch_with_children(client: httpx.Client, source: dict, max_children: int) -> tuple[dict[str, str], str]:
    response = client.get(source["url"])
    response.raise_for_status()
    pages: dict[str, str] = {"root.html": response.text}

    if source["doc_type"] not in {"html", "product"}:
        return pages, hashlib.sha256(response.text.encode("utf-8")).hexdigest()

    soup = BeautifulSoup(response.text, "html.parser")
    child_count = 0
    for anchor in soup.find_all("a", href=True):
        if child_count >= max_children:
            break
        absolute = urljoin(source["url"], anchor["href"])
        if absolute == source["url"] or not absolute.startswith(source["crawl_prefix"]):
            continue
        try:
            child_response = client.get(absolute)
            child_response.raise_for_status()
        except Exception:
            continue
        child_count += 1
        pages[f"page-{child_count}.html"] = child_response.text
    return pages, hashlib.sha256(response.text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the offline NVIDIA corpus bundle into data/corpus.")
    parser.add_argument("--sources", default="data/sources/nvidia_sources.json")
    parser.add_argument("--corpus-root", default="data/corpus")
    parser.add_argument("--max-children", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source_path = project_root / args.sources
    corpus_root = project_root / args.corpus_root
    raw_html_root = corpus_root / "raw/html"
    raw_pdf_root = corpus_root / "raw/pdfs"
    normalized_root = corpus_root / "normalized"
    raw_html_root.mkdir(parents=True, exist_ok=True)
    raw_pdf_root.mkdir(parents=True, exist_ok=True)
    normalized_root.mkdir(parents=True, exist_ok=True)

    snapshot_id = datetime.now(UTC).strftime("%Y%m%d")
    fetched_at = datetime.now(UTC).isoformat()
    manifest: dict[str, object] = {
        "snapshot_id": snapshot_id,
        "retrieved_at": fetched_at,
        "sources": {},
        "errors": [],
    }

    client = httpx.Client(timeout=args.timeout, follow_redirects=True, headers={"User-Agent": "maistorage-agentic-rag/1.0"})
    for source in load_sources(source_path):
        try:
            html_root = raw_html_root / source["id"]
            pdf_root = raw_pdf_root / source["id"]
            html_root.mkdir(parents=True, exist_ok=True)
            pdf_root.mkdir(parents=True, exist_ok=True)

            pages, html_hash = fetch_with_children(client, source, args.max_children)
            local_url_map: dict[str, str] = {}
            for filename, html in pages.items():
                (html_root / filename).write_text(html)
                local_url_map[filename] = source["url"] if filename == "root.html" else source["crawl_prefix"]

            pdf_url = source.get("pdf_url")
            pdf_hash = None
            if pdf_url:
                pdf_response = client.get(pdf_url)
                pdf_response.raise_for_status()
                pdf_bytes = pdf_response.content
                (pdf_root / "source.pdf").write_bytes(pdf_bytes)
                pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

            manifest["sources"][source["id"]] = {
                "title": source["title"],
                "source_url": source["url"],
                "pdf_url": pdf_url,
                "retrieved_at": fetched_at,
                "snapshot_id": snapshot_id,
                "content_hash": html_hash,
                "pdf_hash": pdf_hash,
                "doc_version": source.get("doc_version"),
                "local_url_map": local_url_map,
            }
        except Exception as exc:
            manifest["errors"].append(f"{source['id']}: {exc}")

    (corpus_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
