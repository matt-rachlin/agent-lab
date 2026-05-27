"""Synthetic-query retrieval eval using a local Ollama model.

Vendored from kb_builder.eval. Renamed to `eval_retrieval` so the module name
doesn't collide with `lab.eval` (the experiment-evaluation framework).

1. Sample N chunks weighted by section diversity.
2. Ask the local LLM to write a realistic user query for each.
3. Query the KB; check if the originating chunk appears in top-k.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ollama import Client
from tenacity import retry, stop_after_attempt, wait_exponential

from lab.rag._util import atomic_write_text, console, write_jsonl
from lab.rag.index import hybrid_query

DEFAULT_EVAL_MODEL = "qwen3:14b-q4_K_M"


@dataclass
class EvalQuery:
    question: str
    origin_chunk_id: str
    origin_doc_path: str
    origin_section: list[str]


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
def _gen_question(client: Client, chunk_text: str, section: list[str], model: str) -> str:
    sec = " / ".join(section) if section else "(none)"
    resp = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write a realistic search query a user would type into a knowledge-base "
                    "search to find the given passage. Output ONLY the question, no commentary."
                ),
            },
            {
                "role": "user",
                "content": f"Section: {sec}\nPassage:\n---\n{chunk_text[:1500]}\n---\nQuestion:",
            },
        ],
        options={"num_ctx": 4096, "temperature": 0.3},
    )
    text = (resp.get("message") or {}).get("content") or ""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line.lstrip("> ").strip().strip('"')


def _sample(rows: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    rnd = random.Random(seed)  # noqa: S311  # reason: deterministic sampling, not crypto
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for r in rows:
        key = tuple((r.get("section_path") or [])[:2])
        buckets.setdefault(key, []).append(r)
    keys = list(buckets.keys())
    rnd.shuffle(keys)
    picked: list[dict[str, Any]] = []
    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:
            if not buckets[k]:
                continue
            picked.append(buckets[k].pop())
            if len(picked) >= n:
                break
    return picked


def run_eval(
    kb_dir: Path, *, n: int = 20, k: int = 5, model: str = DEFAULT_EVAL_MODEL
) -> dict[str, Any]:
    import lancedb

    from lab.rag.index import TABLE_NAME

    db = lancedb.connect(str(kb_dir / "index"))
    if TABLE_NAME not in db.list_tables().tables:
        raise RuntimeError("no index — run build first")
    rows = db.open_table(TABLE_NAME).to_arrow().to_pylist()
    if not rows:
        raise RuntimeError("empty index")

    chosen = _sample(rows, n)
    client = Client(host="http://localhost:11434")
    queries: list[EvalQuery] = []
    results: list[dict[str, Any]] = []
    hits_at_1 = 0
    hits_at_5 = 0

    for i, row in enumerate(chosen, 1):
        try:
            q = _gen_question(client, row["text"], list(row.get("section_path") or []), model)
        except Exception as e:
            console.print(f"[red]q-gen failed[/] {row['chunk_id']}: {e}")
            continue
        if not q or len(q) < 8:
            continue
        queries.append(
            EvalQuery(
                question=q,
                origin_chunk_id=row["chunk_id"],
                origin_doc_path=row["doc_path"],
                origin_section=list(row.get("section_path") or []),
            )
        )
        hits = hybrid_query(kb_dir, q, k=k, alpha=0.5)
        ids = [h.chunk_id for h in hits]
        hit1 = ids[:1] == [row["chunk_id"]]
        hit5 = row["chunk_id"] in ids[:5]
        if hit1:
            hits_at_1 += 1
        if hit5:
            hits_at_5 += 1
        results.append(
            {
                "question": q,
                "origin_chunk_id": row["chunk_id"],
                "origin_doc": row["doc_path"],
                "origin_section": row.get("section_path") or [],
                "top_k_ids": ids,
                "top_k_titles": [h.title for h in hits],
                "hit_at_1": hit1,
                "hit_at_5": hit5,
            }
        )
        console.print(f"[dim]eval {i}/{len(chosen)} hit@1={hit1} hit@5={hit5}[/]")

    n_real = max(1, len(queries))
    summary = {
        "synthetic_query_count": len(queries),
        "retrieval_at_1": hits_at_1 / n_real,
        "retrieval_at_5": hits_at_5 / n_real,
        "k": k,
    }
    write_jsonl(kb_dir / "eval" / "queries.jsonl", [vars(q) for q in queries])
    atomic_write_text(
        kb_dir / "eval" / "results.json",
        json.dumps({"summary": summary, "details": results}, indent=2),
    )
    return summary
