"""Post-sweep citation audit for RAG tasks that emit answer=/cite= out.txt files.

The pbs-agent-rag-v0.2 suite makes agents write, to /workspace/out.txt:

    answer=<exact answer>
    cite=<chunk_id>          (one line per supporting chunk)

End-state predicates only check substrings; this script verifies the
*citations themselves*. The episode workspace is gone post-sweep, so the
final out.txt is reconstructed from the MinIO trajectory (the fs_write tool
calls' recorded args — same data access as scripts/trajectory_audit.py).
For every cite= line we check, against the KB's LanceDB index directly:

  * the chunk id EXISTS (else it is a fabricated citation);
  * the cited chunk plausibly SUPPORTS the answer — heuristic: the chunk
    text matches one of the task's ground-truth hop regexes (when
    tasks/<suite>/ground_truth.json is available) or, as a fallback,
    contains the answer's key tokens;
  * mega whole-document chunks (> 20k chars; a known bash-KB chunking
    artifact) are counted as 'weak' support, not valid — the agent only
    ever saw 1500 chars of them.

Optionally (--llm-check) a capped sample of citation/answer pairs is sent
to glm-5.1-cloud via the LiteLLM proxy for an entailment verdict.

Per-cell output: answer_correct (from the recorded end_state score),
citations_valid_frac, fabricated_citations, plus weak/unsupported counts.
Writes analysis/citation_check/<SLUG>/citation_check.csv and prints a
summary. Experiments with no RAG cite-format tasks (e.g. HARD-BENCH-002)
produce 0 cells and exit 0.

Usage (on m-box, from /data/lab/code):
    uv run python scripts/citation_check.py EXP-XXX [--llm-check]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from lab.core.llm import call_litellm_chat
from lab.core.minio_io import make_minio_client
from lab.core.settings import get_settings

PG_DSN = "dbname=lab host=/var/run/postgresql"
LITELLM_KEY_PATH = Path("/data/lab/services/litellm-master-key")
LLM_CHECK_MODEL = "glm-5.1-cloud"
DEFAULT_KB_ROOT = Path("~/db/kb").expanduser()
MEGA_CHUNK_CHARS = 20_000
ANSWER_RE = re.compile(r"^answer=(.*)$", re.MULTILINE)
CITE_RE = re.compile(r"^cite=(\S+)$", re.MULTILINE)

ENTAILMENT_SYSTEM_PROMPT = (
    "You judge whether a retrieved documentation passage supports a stated "
    "answer to a question. Respond with a single JSON object: "
    '{"supports": true | false, "rationale": "<one short sentence>"}. '
    "Never include any text outside the JSON."
)


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------


@dataclass
class Citation:
    chunk_id: str
    exists: bool = False
    mega: bool = False
    supports: bool = False
    llm_supports: bool | None = None


@dataclass
class Cell:
    model: str
    task: str
    seed: int
    score: float | None
    answer: str | None = None
    citations: list[Citation] = field(default_factory=list)
    out_txt_found: bool = False
    note: str = ""

    @property
    def fabricated(self) -> int:
        return sum(1 for c in self.citations if not c.exists)

    @property
    def valid_frac(self) -> float | None:
        if not self.citations:
            return None
        valid = sum(1 for c in self.citations if c.exists and c.supports and not c.mega)
        return valid / len(self.citations)


# --------------------------------------------------------------------------
# episode loading (same access pattern as scripts/trajectory_audit.py)
# --------------------------------------------------------------------------

EPISODE_SQL = """
    select m.litellm_id as model, t.slug as task, er.seed, er.trace_path,
           t.suite, t.payload,
           (al.turns->'score_breakdown'->'end_state'->>'value')::float as score
    from experiment_runs er
    join experiments e on e.experiment_id = er.experiment_id
    join models m on m.model_id = er.model_id
    join tasks t on t.task_id = er.task_id
    left join agent_logs al on al.run_id = er.run_id
    where e.slug = %s
    order by m.litellm_id, t.slug, er.seed
"""


def is_rag_cite_task(payload: dict[str, Any]) -> bool:
    """A task qualifies when it asks for the answer=/cite= out.txt format."""

    tools = {t.get("name") for t in payload.get("tools") or []}
    if "kb_query" not in tools:
        return False
    return "cite=" in str(payload.get("input", ""))


def load_cells(slug: str) -> list[tuple[Cell, str]]:
    """Return (cell, trace_path) for every RAG cite-format episode."""

    out: list[tuple[Cell, str]] = []
    with psycopg.connect(PG_DSN, row_factory=dict_row) as conn:
        for r in conn.execute(EPISODE_SQL, (slug,)):
            payload = r["payload"] or {}
            if not is_rag_cite_task(payload):
                continue
            cell = Cell(
                model=str(r["model"]),
                task=str(r["task"]),
                seed=int(r["seed"]),
                score=float(r["score"]) if r["score"] is not None else None,
            )
            out.append((cell, str(r["trace_path"] or "")))
    return out


def reconstruct_out_txt(trace_blob: bytes) -> str | None:
    """Replay fs_write calls targeting out.txt from the trajectory turns."""

    content: str | None = None
    for line in trace_blob.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "turn":
            continue
        for tc in rec.get("tool_calls") or []:
            if tc.get("tool") != "fs_write" or tc.get("error"):
                continue
            args = tc.get("args") or {}
            if not isinstance(args, dict) or args.get("_truncated"):
                # >4KB writes are stored as a preview marker; out.txt files
                # in this suite are tiny, so treat this as unparseable.
                continue
            path = str(args.get("path") or "")
            if path.lstrip("/").removeprefix("workspace/") != "out.txt":
                continue
            body = str(args.get("content") or "")
            if args.get("mode") == "append" and content is not None:
                content += body
            else:
                content = body
    return content


def fetch_trace(client: Any, trace_path: str) -> bytes | None:
    if not trace_path.startswith("s3://"):
        return None
    bucket, key = trace_path.removeprefix("s3://").split("/", 1)
    try:
        resp = client.get_object(bucket, key)
        blob = resp.read()
        resp.close()
        resp.release_conn()
    except Exception:
        return None
    return blob


# --------------------------------------------------------------------------
# KB access (LanceDB, read-only) + ground truth
# --------------------------------------------------------------------------


def load_kb_chunks(kb_root: Path, kb_name: str) -> dict[str, str]:
    """chunk_id -> text for the whole KB, straight from LanceDB."""

    import lancedb

    db = lancedb.connect(str(kb_root / kb_name / "index"))
    table = db.open_table("chunks")
    arrow = table.to_arrow()
    ids = arrow.column("chunk_id").to_pylist()
    texts = arrow.column("text").to_pylist()
    return dict(zip(ids, [t or "" for t in texts], strict=True))


def load_ground_truth(repo_root: Path) -> dict[str, dict[str, Any]]:
    """slug -> ground-truth record, from any tasks/*/ground_truth.json."""

    gt: dict[str, dict[str, Any]] = {}
    for path in sorted(repo_root.glob("tasks/*/ground_truth.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for task in doc.get("tasks", []):
            gt[task["slug"]] = task
    return gt


def chunk_supports(text: str, cell: Cell, gt_task: dict[str, Any] | None) -> bool:
    """Cheap support heuristic: hop-regex match, else answer-token overlap."""

    if gt_task:
        return any(re.search(hop["regex"], text, re.S) for hop in gt_task.get("hops", []))
    answer = cell.answer or ""
    tokens = [tok for tok in re.split(r"\W+", answer) if len(tok) >= 2]
    if not tokens:
        return False
    return any(tok.lower() in text.lower() for tok in tokens)


# --------------------------------------------------------------------------
# optional LLM entailment pass
# --------------------------------------------------------------------------


def llm_entailment(question: str, answer: str, chunk_text: str, litellm_key: str) -> bool | None:
    user = (
        f"Question: {question}\n\nStated answer: {answer}\n\n"
        f"Passage:\n{chunk_text[:3000]}\n\n"
        "Does the passage support (at least one necessary fact of) the answer?"
    )
    try:
        resp, _latency = call_litellm_chat(
            settings=get_settings(),
            litellm_key=litellm_key,
            model=LLM_CHECK_MODEL,
            messages=[
                {"role": "system", "content": ENTAILMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            # glm-5.1-cloud is a reasoning model: the budget must cover the
            # hidden reasoning tokens or `content` comes back empty with
            # finish_reason=length.
            max_tokens=2000,
        )
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            # Fall back to the reasoning trace if the final answer was cut.
            m = re.search(r"\{.*\}", str(msg.get("reasoning_content") or ""), re.S)
        if not m:
            return None
        return bool(json.loads(m.group(0)).get("supports"))
    except Exception:
        return None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("slug", help="experiment slug, e.g. EXP-015")
    ap.add_argument("--kb-root", type=Path, default=DEFAULT_KB_ROOT)
    ap.add_argument("--kb-name", default="bash")
    ap.add_argument("--llm-check", action="store_true", help="entailment pass via glm-5.1-cloud")
    ap.add_argument("--max-llm-checks", type=int, default=20)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: analysis/citation_check/<SLUG>/ under the repo root",
    )
    args = ap.parse_args()

    cells = load_cells(args.slug)
    if not cells:
        print(f"{args.slug}: no RAG cite-format episodes found — 0 cells, nothing to check")
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    chunks = load_kb_chunks(args.kb_root, args.kb_name)
    gt = load_ground_truth(repo_root)
    print(f"{args.slug}: {len(cells)} RAG episodes; KB '{args.kb_name}' has {len(chunks)} chunks")

    client = make_minio_client()
    litellm_key = ""
    if args.llm_check:
        litellm_key = LITELLM_KEY_PATH.read_text().strip()
    llm_budget = args.max_llm_checks

    done: list[Cell] = []
    for cell, trace_path in cells:
        blob = fetch_trace(client, trace_path)
        if blob is None:
            cell.note = "trace_unavailable"
            done.append(cell)
            continue
        out_txt = reconstruct_out_txt(blob)
        if out_txt is None:
            cell.note = "no out.txt fs_write in trajectory"
            done.append(cell)
            continue
        cell.out_txt_found = True
        m = ANSWER_RE.search(out_txt)
        cell.answer = m.group(1).strip() if m else None
        gt_task = gt.get(cell.task)
        for cid in CITE_RE.findall(out_txt):
            cit = Citation(chunk_id=cid)
            text = chunks.get(cid)
            if text is not None:
                cit.exists = True
                cit.mega = len(text) > MEGA_CHUNK_CHARS
                cit.supports = chunk_supports(text, cell, gt_task)
                if args.llm_check and llm_budget > 0 and not cit.mega:
                    question = (gt_task or {}).get("question", cell.task)
                    cit.llm_supports = llm_entailment(
                        question, cell.answer or "", text, litellm_key
                    )
                    llm_budget -= 1
            cell.citations.append(cit)
        done.append(cell)

    out_dir = args.out_dir or (repo_root / "analysis" / "citation_check" / args.slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "citation_check.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "model",
                "task",
                "seed",
                "answer_correct",
                "answer",
                "n_citations",
                "citations_valid_frac",
                "fabricated_citations",
                "mega_citations",
                "llm_supported",
                "llm_checked",
                "note",
            ]
        )
        for cell in done:
            llm_checked = sum(1 for c in cell.citations if c.llm_supports is not None)
            llm_ok = sum(1 for c in cell.citations if c.llm_supports)
            w.writerow(
                [
                    cell.model,
                    cell.task,
                    cell.seed,
                    "" if cell.score is None else int(cell.score >= 1.0),
                    cell.answer or "",
                    len(cell.citations),
                    "" if cell.valid_frac is None else f"{cell.valid_frac:.2f}",
                    cell.fabricated,
                    sum(1 for c in cell.citations if c.mega),
                    llm_ok,
                    llm_checked,
                    cell.note,
                ]
            )

    n_with_cites = sum(1 for c in done if c.citations)
    fabricated_total = sum(c.fabricated for c in done)
    fracs = [c.valid_frac for c in done if c.valid_frac is not None]
    mean_frac = sum(fracs) / len(fracs) if fracs else float("nan")
    print(
        f"cells={len(done)} with_citations={n_with_cites} "
        f"mean_citations_valid_frac={mean_frac:.2f} fabricated_total={fabricated_total}"
    )
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
