"""Verify the pbs-agent-rag-v0.2 ground truth against the sealed bash KB.

Checks, per task in ground_truth.json:
  1. every hop has >= 1 supporting chunk (regex match over chunk text,
     mega whole-document chunks excluded);
  2. true multi-hop at the corpus level: NO single (non-mega) chunk
     supports ALL hops, so any complete citation set needs >= 2 chunks;
  3. collision safety: no chunk text contains the task's predicate
     substrings ("answer=...", "cite=01KSHSN"), so dumping raw hits into
     out.txt cannot satisfy the predicate;
  4. naive-retrieval multi-hop property: a single top-8 kb_query-style
     hybrid_query with the task's main question does NOT cover all hops.
     Suite-level requirement: >= 10 of 14 tasks must have incomplete
     coverage;
  5. adversarial tasks: the top hit for the naive question must NOT
     contain the answer token (the near-miss surfaces first) and must
     match the near-miss regex somewhere in the top 3.

Optionally cross-checks the suite YAML (--yaml) so predicates and ground
truth cannot drift apart.

Run on m-box from the lab repo so `lab.rag` resolves:
    cd /data/lab/code && uv run python /tmp/rag/verify_rag.py \
        --yaml tasks/pbs-agent-rag-v0.2/multihop.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

KB_DIR = Path("~/db/kb/bash").expanduser()
CHUNKS_PATH = KB_DIR / "chunks" / "chunks.enriched.jsonl"
MEGA_CHUNK_CHARS = 20_000
TOP_K = 8
MAX_FULLY_COVERED = 4  # >= 10 of 14 tasks must NOT be single-query coverable
CITE_PREFIX = "cite=01KSHSN"


def load_chunks() -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    with CHUNKS_PATH.open() as fh:
        for line in fh:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
    return chunks


def hop_support(chunks: dict[str, dict[str, Any]], regex: str) -> list[str]:
    pat = re.compile(regex, re.S)
    return [
        cid
        for cid, c in chunks.items()
        if len(c["text"]) <= MEGA_CHUNK_CHARS and pat.search(c["text"])
    ]


def check_yaml(gt_tasks: list[dict[str, Any]], yaml_path: Path, failures: list[str]) -> None:
    import yaml

    doc = yaml.safe_load(yaml_path.read_text())
    ytasks = {t["slug"]: t for t in doc.get("tasks", [])}
    if len(ytasks) != len(gt_tasks):
        failures.append(f"yaml has {len(ytasks)} tasks, ground truth has {len(gt_tasks)}")
    for gt in gt_tasks:
        yt = ytasks.get(gt["slug"])
        if yt is None:
            failures.append(f"{gt['slug']}: missing from yaml")
            continue
        pred = yt.get("success_predicate") or {}
        if pred.get("type") != "all_of":
            failures.append(f"{gt['slug']}: predicate type is not all_of")
            continue
        subs = [p.get("substring") for p in pred.get("predicates", [])]
        if gt["answer_substring"] not in subs:
            failures.append(f"{gt['slug']}: predicate missing {gt['answer_substring']!r}")
        if not any(s and s.startswith("cite=") for s in subs):
            failures.append(f"{gt['slug']}: predicate has no cite= substring check")
        tools = {t["name"] for t in yt.get("tools", [])}
        if not {"kb_query", "fs_write"} <= tools:
            failures.append(f"{gt['slug']}: tools must include kb_query and fs_write")
        # Every cite= predicate substring that pins an exact chunk id must
        # name a real chunk (checked later by caller via returned subs).
        gt["_yaml_cite_subs"] = [
            s for s in subs if s and s.startswith("cite=") and s != CITE_PREFIX
        ]


def main() -> int:
    ap = argparse.ArgumentParser()
    default_gt = str(Path(__file__).resolve().parent / "ground_truth.json")
    ap.add_argument("--ground-truth", default=default_gt)
    ap.add_argument("--yaml", default=None, help="suite yaml to cross-check")
    ap.add_argument("--no-retrieval", action="store_true", help="skip hybrid_query checks")
    args = ap.parse_args()

    gt = json.loads(Path(args.ground_truth).read_text())
    tasks = gt["tasks"]
    chunks = load_chunks()
    failures: list[str] = []

    if args.yaml:
        check_yaml(tasks, Path(args.yaml), failures)

    # ---- corpus-level checks -------------------------------------------
    supports: dict[str, list[list[str]]] = {}
    for t in tasks:
        slug = t["slug"]
        hop_sets: list[list[str]] = []
        for hop in t["hops"]:
            sup = hop_support(chunks, hop["regex"])
            hop_sets.append(sup)
            mark = "OK " if sup else "FAIL"
            print(f"[{mark}] {slug} / hop {hop['name']}: {len(sup)} supporting chunks")
            if len(sup) <= 4:
                for cid in sup:
                    c = chunks[cid]
                    print(f"        {cid}  {c.get('title') or c['source_url']}")
            if not sup:
                failures.append(f"{slug}: hop {hop['name']} has no supporting chunks")
        supports[slug] = hop_sets

        # single-chunk cover => not truly multi-hop
        all_sets = [set(s) for s in hop_sets]
        cover = set.intersection(*all_sets) if all_sets else set()
        if cover:
            names = ", ".join(sorted(cover))
            failures.append(f"{slug}: single chunk(s) cover ALL hops: {names}")
            print(f"[FAIL] {slug}: single-chunk cover by {names}")
        else:
            print(f"[OK ] {slug}: no single chunk covers all hops (>=2 cites required)")

        # collision safety
        for needle in (t["answer_substring"], CITE_PREFIX):
            hits = [cid for cid, c in chunks.items() if needle in c["text"]]
            if hits:
                failures.append(
                    f"{slug}: predicate substring {needle!r} occurs in corpus: {hits[:3]}"
                )

        # exact-id cite predicates must reference real chunks that support a hop
        for sub in t.get("_yaml_cite_subs", []):
            cid = sub.removeprefix("cite=")
            if cid not in chunks:
                failures.append(f"{slug}: predicate cites unknown chunk {cid}")
            elif not any(cid in s for s in hop_sets):
                failures.append(f"{slug}: predicate-pinned chunk {cid} supports no hop")

    # ---- retrieval checks ----------------------------------------------
    fully_covered = 0
    if not args.no_retrieval:
        from lab.rag.index import hybrid_query

        print("\n--- naive single-query top-8 coverage ---")
        for t in tasks:
            slug = t["slug"]
            hits = hybrid_query(KB_DIR, t["question"], k=TOP_K)
            top_ids = [h.chunk_id for h in hits]
            covered = [bool(set(sup) & set(top_ids)) for sup in supports[slug]]
            n_cov = sum(covered)
            full = n_cov == len(covered)
            fully_covered += int(full)
            tag = "COVERED(!)" if full else "incomplete"
            print(f"  {slug}: {n_cov}/{len(covered)} hops in top-{TOP_K} -> {tag}")

            if t.get("adversarial"):
                # `answer_regex` (when set) defines what counts as "the
                # answer is visible in this chunk" — needed when the literal
                # answer is a common word. Default: literal containment.
                ans_rx = t.get("answer_regex")
                top1 = chunks.get(top_ids[0]) if top_ids else None
                if ans_rx:
                    visible = bool(top1 and re.search(ans_rx, top1["text"], re.S | re.I))
                else:
                    visible = bool(top1 and t["answer"].lower() in top1["text"].lower())
                if visible:
                    failures.append(
                        f"{slug}: NOT adversarial — top-1 naive hit already shows the answer"
                    )
                else:
                    print("    adversarial OK: top-1 hit does not show the answer")
                nm = re.compile(t["near_miss"]["regex"], re.S)
                if not any(nm.search(chunks[cid]["text"]) for cid in top_ids[:3] if cid in chunks):
                    failures.append(f"{slug}: near-miss pattern absent from naive top-3")
                else:
                    print("    near-miss surfaces in naive top-3: OK")

        n_multi = len(tasks) - fully_covered
        print(
            f"\nmulti-hop property: {n_multi}/{len(tasks)} tasks NOT fully covered by one top-8 query"
        )
        if fully_covered > MAX_FULLY_COVERED:
            failures.append(
                f"only {n_multi} tasks pass the naive-coverage check; need >= {len(tasks) - MAX_FULLY_COVERED}"
            )

    print("\n=== verify_rag summary ===")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print(f"all checks passed for {len(tasks)} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
