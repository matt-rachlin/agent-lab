"""Judge-calibration study — LLM judges vs machine-verified ground truth.

The lab's agent benchmarks score episodes with deterministic end-state
predicates, which gives us something most eval stacks lack: machine-verified
ground truth at scale. This script measures how accurate LLM judges are
against that truth and characterizes their biases.

Population: all scored cells from HARD-BENCH-001, HARD-BENCH-002,
CODER-BENCH-001 and HARD-BENCH-CLOUD-001 (~364 episodes), stratified down to
--max-episodes with pass/fail balanced toward 60/40 where possible (seeded
RNG). Each sampled episode is rendered for judging — task statement plus the
FULL final conversation from the trajectory `messages` record (tool results
truncated at 2 KB each, truncation marked) — with the scoring predicate and
the ground-truth score withheld. Each judge answers:

  "Did the agent successfully complete the task?"
  -> JSON {"verdict": "pass"|"fail", "confidence": 0-100, "rationale": ...}

at temperature 0 over the LiteLLM proxy. JSON parse failures are retried once
with a stricter reminder; still-unparseable responses are recorded as abstain.
Raw responses are cached incrementally to a JSONL keyed (run_id, judge); the
script is resumable — already-judged pairs are skipped on re-run.

Outputs (in --out):
  judgments_cache.jsonl        raw judge responses (the resume cache)
  study_population.csv         the sampled episode manifest
  confusion_matrices.csv       per-judge TP/FP/TN/FN, accuracy, FPR-on-failures
  calibration.csv              per-judge confidence buckets vs empirical accuracy
  bias_length.csv              accuracy by episode-length tercile (verbosity bias)
  bias_subject_model.csv       accuracy/leniency by subject model (family bias)
  bias_family.csv              same-family vs other-family roll-up per judge
  bias_category.csv            accuracy by task category
  agreement.csv                pairwise inter-judge Cohen's kappa
  SUMMARY.md                   the headline tables

Cost: ~(judges x episodes) cloud calls on the LiteLLM cloud lanes; no local
GPU use. Re-runs only pay for missing pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import psycopg
from jobs_status import Job
from psycopg.rows import dict_row

from lab.core.llm import call_litellm_chat
from lab.core.minio_io import make_minio_client
from lab.core.settings import get_settings

PG_DSN = "dbname=lab host=/var/run/postgresql"
LITELLM_KEY_PATH = Path("/data/lab/services/litellm-master-key")

EXPERIMENT_SLUGS = (
    "HARD-BENCH-001",
    "HARD-BENCH-002",
    "CODER-BENCH-001",
    "HARD-BENCH-CLOUD-001",
)
DEFAULT_JUDGES = "glm-5.1-cloud,gpt-oss-120b-cloud,qwen3-coder-480b-cloud"

PASS_THRESHOLD = 1.0
TARGET_PASS_SHARE = 0.6  # aim for ~60/40 pass/fail in the sample
TOOL_RESULT_LIMIT = 2048  # bytes per tool result shown to the judge
TRANSCRIPT_LIMIT = 100_000  # global safety cap on rendered transcript chars
MAX_TRANSIENT_RETRIES = 3
RETRY_SLEEP_S = (5.0, 15.0, 30.0)
JUDGE_MAX_TOKENS = 8192
JUDGE_TIMEOUT_S = 300

CONF_BUCKETS = ((0, 50), (50, 70), (70, 80), (80, 90), (90, 101))

JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}")
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge of AI-agent benchmark episodes. You are given "
    "a task statement and the full conversation of an agent attempting that "
    "task (some long tool results are truncated and marked as such). Decide "
    "ONLY whether the agent successfully completed the task as stated. "
    "Respond with a single JSON object and nothing else: "
    '{"verdict": "pass" | "fail", "confidence": <integer 0-100>, '
    '"rationale": "<one short sentence>"}.'
)
JUDGE_QUESTION = (
    "Did the agent successfully complete the task? Respond JSON "
    '{"verdict": "pass" | "fail", "confidence": 0-100, "rationale": "<short>"}.'
)
STRICT_REMINDER = (
    "Your previous reply could not be parsed. Respond with ONLY one JSON "
    'object, no markdown fences, no extra text: {"verdict": "pass" or "fail", '
    '"confidence": integer 0-100, "rationale": short string}.'
)

MODEL_FAMILIES = {
    "gemma": "gemma",
    "qwen": "qwen",
    "devstral": "mistral",
    "glm": "glm",
    "gpt-oss": "gpt-oss",
    "deepseek": "deepseek",
    "kimi": "kimi",
    "llama": "llama",
    "phi": "phi",
    "granite": "granite",
    "hermes": "hermes",
}


def model_family(model: str) -> str:
    low = model.lower()
    for prefix, fam in MODEL_FAMILIES.items():
        if low.startswith(prefix):
            return fam
    return "other"


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------


@dataclass
class Episode:
    run_id: str
    experiment: str
    model: str
    task: str
    category: str
    seed: int
    score: float
    trace_path: str
    task_input: str
    transcript: str = ""
    transcript_chars: int = 0
    n_messages: int = 0
    length_tercile: str = ""

    @property
    def passed(self) -> bool:
        return self.score >= PASS_THRESHOLD


@dataclass
class Judgment:
    run_id: str
    judge: str
    verdict: str  # pass | fail | abstain | error
    confidence: int | None
    rationale: str
    attempts: int
    raw_response: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int

    @property
    def final(self) -> bool:
        return self.verdict in ("pass", "fail", "abstain")


# --------------------------------------------------------------------------
# loading & sampling
# --------------------------------------------------------------------------


def load_population() -> list[Episode]:
    sql = """
        select er.run_id, e.slug as experiment, m.litellm_id as model,
               t.slug as task, coalesce(t.category, '?') as category, er.seed,
               er.trace_path,
               coalesce(t.payload->>'input', '') as task_input,
               (al.turns->'score_breakdown'->'end_state'->>'value')::float
                   as score
        from experiment_runs er
        join experiments e on e.experiment_id = er.experiment_id
        join models m on m.model_id = er.model_id
        join tasks t on t.task_id = er.task_id
        left join agent_logs al on al.run_id = er.run_id
        where e.slug = any(%s)
        order by er.run_id
    """
    episodes: list[Episode] = []
    with psycopg.connect(PG_DSN, row_factory=dict_row) as conn:
        for r in conn.execute(sql, (list(EXPERIMENT_SLUGS),)):
            if r["score"] is None:
                continue
            episodes.append(
                Episode(
                    run_id=str(r["run_id"]),
                    experiment=str(r["experiment"]),
                    model=str(r["model"]),
                    task=str(r["task"]),
                    category=str(r["category"]),
                    seed=int(r["seed"]),
                    score=float(r["score"]),
                    trace_path=str(r["trace_path"] or ""),
                    task_input=str(r["task_input"]),
                )
            )
    return episodes


def stratified_sample(episodes: list[Episode], cap: int, seed: int) -> list[Episode]:
    """Cap at `cap`, balancing pass/fail toward 60/40 where possible."""
    rng = random.Random(seed)
    passes = [e for e in episodes if e.passed]
    fails = [e for e in episodes if not e.passed]
    if len(episodes) <= cap:
        return sorted(episodes, key=lambda e: e.run_id)
    want_fail = min(len(fails), round(cap * (1 - TARGET_PASS_SHARE)))
    want_pass = min(len(passes), cap - want_fail)
    if want_pass + want_fail < cap:  # one stratum short: backfill from the other
        want_fail = min(len(fails), cap - want_pass)
    sample = rng.sample(passes, want_pass) + rng.sample(fails, want_fail)
    return sorted(sample, key=lambda e: e.run_id)


# --------------------------------------------------------------------------
# trajectory rendering
# --------------------------------------------------------------------------


def fetch_messages(client: Any, trace_path: str) -> list[dict[str, Any]] | None:
    """Fetch the trajectory JSONL and return the `messages` record's messages."""
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
    for line in blob.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "messages":
            msgs = rec.get("messages")
            if isinstance(msgs, list):
                return msgs
    return None


def render_transcript(messages: list[dict[str, Any]]) -> str:
    """Render the conversation for the judge. Tool results > 2 KB are
    truncated (marked); ground truth / predicates never appear in messages."""
    parts: list[str] = []
    for m in messages:
        role = str(m.get("role") or "?")
        content = m.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, default=str) if content is not None else ""
        if role == "tool":
            fn = str(m.get("function") or "tool")
            if len(content) > TOOL_RESULT_LIMIT:
                total = len(content)
                content = (
                    content[:TOOL_RESULT_LIMIT]
                    + f"\n...[tool result truncated: showing {TOOL_RESULT_LIMIT}"
                    + f" of {total} chars]"
                )
            err = m.get("error")
            tag = f"[TOOL RESULT {fn}{' — ERROR' if err else ''}]"
            parts.append(f"{tag}:\n{content}")
            continue
        block = f"[{role.upper()}]:\n{content}" if content.strip() else f"[{role.upper()}]: (empty)"
        calls = m.get("tool_calls") or []
        for tc in calls:
            fn_obj = tc.get("function") or {} if isinstance(tc, dict) else {}
            name = fn_obj.get("name") or "?"
            args = str(fn_obj.get("arguments") or "")[:400]
            block += f"\n[CALLS TOOL {name}({args})]"
        parts.append(block)
    text = "\n\n".join(parts)
    if len(text) > TRANSCRIPT_LIMIT:
        half = TRANSCRIPT_LIMIT // 2
        text = (
            text[:half]
            + f"\n\n...[transcript middle truncated: {len(text)} chars total]...\n\n"
            + text[-half:]
        )
    return text


def assign_length_terciles(episodes: list[Episode]) -> None:
    ordered = sorted(e.transcript_chars for e in episodes)
    lo = ordered[len(ordered) // 3]
    hi = ordered[2 * len(ordered) // 3]
    for e in episodes:
        e.length_tercile = (
            "short"
            if e.transcript_chars <= lo
            else "medium"
            if e.transcript_chars <= hi
            else "long"
        )


# --------------------------------------------------------------------------
# judging
# --------------------------------------------------------------------------


def parse_judgment(text: str) -> tuple[str, int | None, str] | None:
    """Return (verdict, confidence, rationale) or None if unparseable."""
    cleaned = text.strip()
    fence = FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    for m in JSON_OBJ_RE.finditer(cleaned):
        try:
            obj = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or "verdict" not in obj:
            continue
        verdict = str(obj.get("verdict", "")).strip().lower()
        if verdict not in ("pass", "fail"):
            continue
        confidence: int | None = None
        try:
            confidence = max(0, min(100, int(float(obj.get("confidence")))))
        except (TypeError, ValueError):
            confidence = None
        return verdict, confidence, str(obj.get("rationale", ""))[:400]
    return None


def call_judge_once(
    settings: Any, key: str, judge: str, messages: list[dict[str, Any]]
) -> tuple[str, int, int, int]:
    """One proxy call with transient-error retries.
    Returns (content, latency_ms, prompt_tokens, completion_tokens)."""
    last_exc: Exception | None = None
    for attempt in range(MAX_TRANSIENT_RETRIES):
        try:
            resp, latency_ms = call_litellm_chat(
                settings=settings,
                litellm_key=key,
                model=judge,
                messages=messages,
                temperature=0.0,
                max_tokens=JUDGE_MAX_TOKENS,
                timeout=JUDGE_TIMEOUT_S,
            )
            content = str(resp["choices"][0]["message"].get("content") or "")
            usage = resp.get("usage") or {}
            return (
                content,
                latency_ms,
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
            )
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(RETRY_SLEEP_S[min(attempt, len(RETRY_SLEEP_S) - 1)])
    raise RuntimeError(f"{judge}: transient retries exhausted: {last_exc}")


def judge_episode(settings: Any, key: str, judge: str, ep: Episode) -> Judgment:
    user_msg = (
        "TASK STATEMENT:\n"
        f"{ep.task_input.strip()}\n\n"
        "AGENT CONVERSATION:\n"
        f"{ep.transcript}\n\n"
        f"{JUDGE_QUESTION}"
    )
    convo: list[dict[str, Any]] = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    total_prompt = total_completion = total_latency = 0
    raw_all: list[str] = []
    attempts = 0
    try:
        for strict in (False, True):
            attempts += 1
            if strict:
                convo = [
                    *convo,
                    {"role": "assistant", "content": raw_all[-1] or "(empty)"},
                    {"role": "user", "content": STRICT_REMINDER},
                ]
            content, latency_ms, pt, ct = call_judge_once(settings, key, judge, convo)
            raw_all.append(content)
            total_latency += latency_ms
            total_prompt += pt
            total_completion += ct
            parsed = parse_judgment(content)
            if parsed is not None:
                verdict, confidence, rationale = parsed
                return Judgment(
                    ep.run_id,
                    judge,
                    verdict,
                    confidence,
                    rationale,
                    attempts,
                    "\n---RETRY---\n".join(raw_all)[:4000],
                    total_latency,
                    total_prompt,
                    total_completion,
                )
        return Judgment(
            ep.run_id,
            judge,
            "abstain",
            None,
            "unparseable after strict retry",
            attempts,
            "\n---RETRY---\n".join(raw_all)[:4000],
            total_latency,
            total_prompt,
            total_completion,
        )
    except Exception as exc:
        return Judgment(
            ep.run_id,
            judge,
            "error",
            None,
            str(exc)[:300],
            attempts,
            "\n---RETRY---\n".join(raw_all)[:4000],
            total_latency,
            total_prompt,
            total_completion,
        )


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def load_cache(path: Path) -> dict[tuple[str, str], Judgment]:
    cache: dict[tuple[str, str], Judgment] = {}
    if not path.exists():
        return cache
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        j = Judgment(
            run_id=str(r["run_id"]),
            judge=str(r["judge"]),
            verdict=str(r["verdict"]),
            confidence=r.get("confidence"),
            rationale=str(r.get("rationale", "")),
            attempts=int(r.get("attempts", 1)),
            raw_response=str(r.get("raw_response", "")),
            latency_ms=int(r.get("latency_ms", 0)),
            prompt_tokens=int(r.get("prompt_tokens", 0)),
            completion_tokens=int(r.get("completion_tokens", 0)),
        )
        if j.final:  # error records are not final: re-run retries them
            cache[(j.run_id, j.judge)] = j
    return cache


def append_cache(path: Path, lock: threading.Lock, j: Judgment) -> None:
    rec = {
        "run_id": j.run_id,
        "judge": j.judge,
        "verdict": j.verdict,
        "confidence": j.confidence,
        "rationale": j.rationale,
        "attempts": j.attempts,
        "raw_response": j.raw_response,
        "latency_ms": j.latency_ms,
        "prompt_tokens": j.prompt_tokens,
        "completion_tokens": j.completion_tokens,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with lock, path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    abstain: int = 0

    def add(self, gt_pass: bool, verdict: str) -> None:
        if verdict == "abstain":
            self.abstain += 1
        elif verdict == "pass":
            self.tp += gt_pass
            self.fp += not gt_pass
        else:
            self.fn += gt_pass
            self.tn += not gt_pass

    @property
    def n_judged(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n_judged if self.n_judged else float("nan")

    @property
    def fpr_on_failures(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else float("nan")

    @property
    def fnr_on_passes(self) -> float:
        denom = self.tp + self.fn
        return self.fn / denom if denom else float("nan")


def confusion_for(pairs: list[tuple[Episode, Judgment]]) -> Confusion:
    c = Confusion()
    for ep, j in pairs:
        c.add(ep.passed, j.verdict)
    return c


def cohens_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = 0.0
    for label in ("pass", "fail"):
        pe += (a.count(label) / n) * (b.count(label) / n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fmt(v: float) -> str:
    return "-" if v != v else f"{v:.3f}"


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------


def split_rows(
    pairs_by_judge: dict[str, list[tuple[Episode, Judgment]]],
    keyfn: Any,
) -> list[dict[str, Any]]:
    """Per (judge, split-key) accuracy / leniency rows."""
    rows: list[dict[str, Any]] = []
    for judge in sorted(pairs_by_judge):
        groups: dict[str, list[tuple[Episode, Judgment]]] = defaultdict(list)
        for ep, j in pairs_by_judge[judge]:
            groups[str(keyfn(ep))].append((ep, j))
        for key in sorted(groups):
            grp = groups[key]
            c = confusion_for(grp)
            n_pass_verdicts = sum(1 for _, j in grp if j.verdict == "pass")
            n_gt_pass = sum(1 for ep, _ in grp if ep.passed)
            rows.append(
                {
                    "judge": judge,
                    "split": key,
                    "n": len(grp),
                    "gt_pass_rate": f"{n_gt_pass / len(grp):.3f}",
                    "judge_pass_rate": (
                        f"{n_pass_verdicts / c.n_judged:.3f}" if c.n_judged else "-"
                    ),
                    "accuracy": fmt(c.accuracy),
                    "fpr_on_failures": fmt(c.fpr_on_failures),
                    "abstains": c.abstain,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judges", default=DEFAULT_JUDGES, help="comma-separated judge lanes")
    ap.add_argument("--max-episodes", type=int, default=240)
    ap.add_argument("--out", default="analysis/judge-calibration/")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "judgments_cache.jsonl"
    settings = get_settings()
    key = LITELLM_KEY_PATH.read_text().strip()

    # ---- population & sample ------------------------------------------------
    population = load_population()
    sample = stratified_sample(population, args.max_episodes, args.seed)
    print(
        f"population: {len(population)} scored episodes "
        f"({sum(e.passed for e in population)} pass / "
        f"{sum(not e.passed for e in population)} fail) from {len(EXPERIMENT_SLUGS)} experiments"
    )
    print(
        f"sample: {len(sample)} episodes ({sum(e.passed for e in sample)} pass / "
        f"{sum(not e.passed for e in sample)} fail), seed={args.seed}"
    )

    # ---- fetch + render trajectories ----------------------------------------
    client = make_minio_client()
    kept: list[Episode] = []
    dropped = 0
    for ep in sample:
        msgs = fetch_messages(client, ep.trace_path)
        if not msgs:
            dropped += 1
            continue
        ep.n_messages = len(msgs)
        ep.transcript = render_transcript(msgs)
        ep.transcript_chars = len(ep.transcript)
        kept.append(ep)
    if dropped:
        print(f"dropped {dropped} episodes with missing/unreadable trajectories")
    sample = kept
    assign_length_terciles(sample)
    eps_by_id = {e.run_id: e for e in sample}

    write_csv(
        out_dir / "study_population.csv",
        [
            {
                "run_id": e.run_id,
                "experiment": e.experiment,
                "model": e.model,
                "model_family": model_family(e.model),
                "task": e.task,
                "category": e.category,
                "seed": e.seed,
                "ground_truth": "pass" if e.passed else "fail",
                "transcript_chars": e.transcript_chars,
                "n_messages": e.n_messages,
                "length_tercile": e.length_tercile,
            }
            for e in sample
        ],
    )

    # ---- judge (resumable, cached) -------------------------------------------
    cache = load_cache(cache_path)
    lock = threading.Lock()
    todo = [(ep, judge) for judge in judges for ep in sample if (ep.run_id, judge) not in cache]
    print(f"cache: {len(cache)} judged pairs; {len(todo)} calls to make")

    done_count = 0
    with Job(f"judge-calibration n={len(todo)} judges={len(judges)}") as job:
        bar = job.bar("judgments", total=max(len(todo), 1))
        if todo:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(judge_episode, settings, key, judge, ep): (ep, judge)
                    for ep, judge in todo
                }
                for fut in as_completed(futures):
                    ep, judge = futures[fut]
                    j = fut.result()
                    done_count += 1
                    if j.final:
                        append_cache(cache_path, lock, j)
                        cache[(ep.run_id, judge)] = j
                    bar.advance(1, message=f"{judge}/{ep.task} -> {j.verdict}")
                    print(
                        f"[{done_count}/{len(todo)}] {judge} <- {ep.model}/{ep.task}/s{ep.seed} "
                        f"-> {j.verdict}"
                        + (f" ({j.confidence})" if j.confidence is not None else "")
                        + (f" [{j.rationale[:80]}]" if j.verdict == "error" else ""),
                        flush=True,
                    )
        job.log(f"judging complete: {done_count} new, {len(cache)} total in cache")

    # ---- assemble pairs -------------------------------------------------------
    pairs_by_judge: dict[str, list[tuple[Episode, Judgment]]] = {j: [] for j in judges}
    for (run_id, judge), j in cache.items():
        ep = eps_by_id.get(run_id)
        if ep is not None and judge in pairs_by_judge:
            pairs_by_judge[judge].append((ep, j))
    missing = {j: len(sample) - len(pairs_by_judge[j]) for j in judges}
    if any(missing.values()):
        print(f"WARNING: unjudged pairs remain (errors): {missing} — re-run to retry")

    # ---- confusion matrices ---------------------------------------------------
    conf_rows: list[dict[str, Any]] = []
    for judge in judges:
        c = confusion_for(pairs_by_judge[judge])
        conf_rows.append(
            {
                "judge": judge,
                "n_judged": c.n_judged,
                "abstain": c.abstain,
                "TP": c.tp,
                "FP": c.fp,
                "TN": c.tn,
                "FN": c.fn,
                "accuracy": fmt(c.accuracy),
                "fpr_on_failures": fmt(c.fpr_on_failures),
                "fnr_on_passes": fmt(c.fnr_on_passes),
            }
        )
    write_csv(out_dir / "confusion_matrices.csv", conf_rows)

    # ---- calibration ------------------------------------------------------------
    cal_rows: list[dict[str, Any]] = []
    for judge in judges:
        for lo, hi in CONF_BUCKETS:
            grp = [
                (ep, j)
                for ep, j in pairs_by_judge[judge]
                if j.verdict in ("pass", "fail")
                and j.confidence is not None
                and lo <= j.confidence < hi
            ]
            n = len(grp)
            correct = sum((j.verdict == "pass") == ep.passed for ep, j in grp)
            mean_conf = sum(j.confidence or 0 for _, j in grp) / n if n else float("nan")
            cal_rows.append(
                {
                    "judge": judge,
                    "bucket": f"[{lo},{min(hi, 100)}{')' if hi <= 100 else ']'}",
                    "n": n,
                    "mean_confidence": fmt(mean_conf / 100) if n else "-",
                    "empirical_accuracy": f"{correct / n:.3f}" if n else "-",
                }
            )
    write_csv(out_dir / "calibration.csv", cal_rows)

    # ---- bias probes ---------------------------------------------------------
    length_rows = split_rows(pairs_by_judge, lambda e: e.length_tercile)
    subject_rows = split_rows(pairs_by_judge, lambda e: e.model)
    category_rows = split_rows(pairs_by_judge, lambda e: e.category)
    family_rows: list[dict[str, Any]] = []
    for judge in judges:
        jfam = model_family(judge)
        for label, pred in (("same_family", True), ("other_family", False)):
            grp = [
                (ep, j)
                for ep, j in pairs_by_judge[judge]
                if (model_family(ep.model) == jfam) is pred
            ]
            if not grp:
                continue
            c = confusion_for(grp)
            n_pass_verdicts = sum(1 for _, j in grp if j.verdict == "pass")
            family_rows.append(
                {
                    "judge": judge,
                    "judge_family": jfam,
                    "split": label,
                    "n": len(grp),
                    "gt_pass_rate": f"{sum(e.passed for e, _ in grp) / len(grp):.3f}",
                    "judge_pass_rate": (
                        f"{n_pass_verdicts / c.n_judged:.3f}" if c.n_judged else "-"
                    ),
                    "accuracy": fmt(c.accuracy),
                    "fpr_on_failures": fmt(c.fpr_on_failures),
                }
            )
    write_csv(out_dir / "bias_length.csv", length_rows)
    write_csv(out_dir / "bias_subject_model.csv", subject_rows)
    write_csv(out_dir / "bias_category.csv", category_rows)
    write_csv(out_dir / "bias_family.csv", family_rows)

    # ---- inter-judge agreement -------------------------------------------------
    agree_rows: list[dict[str, Any]] = []
    verdicts: dict[str, dict[str, str]] = {
        judge: {ep.run_id: j.verdict for ep, j in pairs_by_judge[judge]} for judge in judges
    }
    for i, ja in enumerate(judges):
        for jb in judges[i + 1 :]:
            common = [
                rid
                for rid in eps_by_id
                if verdicts[ja].get(rid) in ("pass", "fail")
                and verdicts[jb].get(rid) in ("pass", "fail")
            ]
            a = [verdicts[ja][rid] for rid in common]
            b = [verdicts[jb][rid] for rid in common]
            raw = (
                sum(x == y for x, y in zip(a, b, strict=True)) / len(common)
                if common
                else float("nan")
            )
            agree_rows.append(
                {
                    "judge_a": ja,
                    "judge_b": jb,
                    "n": len(common),
                    "raw_agreement": fmt(raw),
                    "cohens_kappa": fmt(cohens_kappa(a, b)),
                }
            )
    write_csv(out_dir / "agreement.csv", agree_rows)

    # ---- SUMMARY.md ----------------------------------------------------------
    total_calls = len(cache)
    total_prompt = sum(j.prompt_tokens for j in cache.values())
    total_completion = sum(j.completion_tokens for j in cache.values())
    lines = ["# Judge calibration study", ""]
    lines.append(
        f"Population: {len(population)} machine-scored episodes from "
        f"{', '.join(EXPERIMENT_SLUGS)}. Sampled {len(sample)} "
        f"({sum(e.passed for e in sample)} pass / {sum(not e.passed for e in sample)} fail, "
        f"seed {args.seed}); ground truth = deterministic end-state predicate."
    )
    lines.append("")
    lines.append(
        f"Judges: {', '.join(judges)}. {total_calls} judged pairs, "
        f"{total_prompt:,} prompt + {total_completion:,} completion tokens."
    )
    lines.append("")
    lines.append("## Confusion matrices (positive = judge says pass)")
    lines.append("")
    lines += md_table(
        conf_rows,
        [
            "judge",
            "n_judged",
            "abstain",
            "TP",
            "FP",
            "TN",
            "FN",
            "accuracy",
            "fpr_on_failures",
            "fnr_on_passes",
        ],
    )
    lines.append("")
    lines.append(
        "`fpr_on_failures` is the dangerous error: the judge passes an episode "
        "the predicate says failed."
    )
    lines.append("")
    lines.append("## Calibration (stated confidence vs empirical accuracy)")
    lines.append("")
    lines += md_table(cal_rows, ["judge", "bucket", "n", "mean_confidence", "empirical_accuracy"])
    lines.append("")
    lines.append("## Bias probes")
    lines.append("")
    lines.append("### Episode length terciles (verbosity bias)")
    lines.append("")
    lines += md_table(
        length_rows,
        ["judge", "split", "n", "gt_pass_rate", "judge_pass_rate", "accuracy", "fpr_on_failures"],
    )
    lines.append("")
    lines.append("### Subject model")
    lines.append("")
    lines += md_table(
        subject_rows,
        ["judge", "split", "n", "gt_pass_rate", "judge_pass_rate", "accuracy", "fpr_on_failures"],
    )
    lines.append("")
    lines.append("### Same-family vs other-family (self-preference)")
    lines.append("")
    lines += md_table(
        family_rows,
        [
            "judge",
            "judge_family",
            "split",
            "n",
            "gt_pass_rate",
            "judge_pass_rate",
            "accuracy",
            "fpr_on_failures",
        ],
    )
    lines.append("")
    lines.append("### Task category")
    lines.append("")
    lines += md_table(
        category_rows,
        ["judge", "split", "n", "gt_pass_rate", "judge_pass_rate", "accuracy", "fpr_on_failures"],
    )
    lines.append("")
    lines.append("## Inter-judge agreement")
    lines.append("")
    lines += md_table(agree_rows, ["judge_a", "judge_b", "n", "raw_agreement", "cohens_kappa"])
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_dir}/SUMMARY.md and CSVs")
    sys.exit(0)


if __name__ == "__main__":
    main()
