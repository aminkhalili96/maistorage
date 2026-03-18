"""Quick sanity check: measure how many chunks the navigation filter drops."""
from __future__ import annotations

import json
from pathlib import Path

from app.services.chunking import _is_navigation_chunk


def test_filter_stats():
    demo_path = Path(__file__).resolve().parents[2] / "data" / "demo_chunks.json"
    chunks = json.loads(demo_path.read_text())
    total = len(chunks)
    nav_chunks = [c for c in chunks if _is_navigation_chunk(c.get("content", ""))]
    nav = len(nav_chunks)
    kept = total - nav

    print(f"\nTotal demo chunks : {total}")
    print(f"Nav/noise filtered: {nav} ({nav / total * 100:.1f}%)")
    print(f"Substantive kept  : {kept} ({kept / total * 100:.1f}%)")

    print("\n--- 3 example FILTERED chunks ---")
    for ex in nav_chunks[:3]:
        content = ex.get("content", "")
        print(
            f"  source={ex.get('source_id','?')} | "
            f"tokens≈{len(content.split())} | "
            f"preview: {content[:140]!r}"
        )

    assert kept >= 10, "Too many chunks filtered — check threshold"
    print("\nFilter check passed.")
