"""git fetcher: shallow-clone a repo, walk for doc-ish files, normalize each.

Vendored from kb_builder.fetchers.git.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import trafilatura

from lab.rag._plan import PlannedSource
from lab.rag.fetchers import FetchedDoc, FetcherContext, FetchResult, register

DOC_EXTS = {
    ".md",
    ".markdown",
    ".rst",
    ".txt",
    ".adoc",
    ".asciidoc",
    ".texi",
    ".info",
    ".1",
    ".5",
    ".7",
    ".8",
    ".html",
    ".htm",
}

MAX_FILES = 400
MAX_FILE_BYTES = 1_000_000  # 1 MB


def _is_doc_path(p: Path, paths_filter: list[str] | None) -> bool:
    if p.suffix.lower() not in DOC_EXTS:
        return False
    if paths_filter:
        return any(s in str(p) for s in paths_filter)
    # default heuristic: prefer Documentation, doc, docs, README, manual subtrees
    parts = {seg.lower() for seg in p.parts}
    if {"documentation", "doc", "docs", "manual"} & parts:
        return True
    if p.name.lower().startswith("readme"):
        return True
    return True  # other doc extensions are still considered


def _render(file: Path, body: bytes) -> str:
    ext = file.suffix.lower()
    if ext in {".md", ".markdown"}:
        return body.decode("utf-8", errors="replace")
    if ext == ".rst":
        text = body.decode("utf-8", errors="replace")
        # Minimal RST -> markdown: keep mostly verbatim. Headings translate roughly.
        return text
    if ext in {".txt", ".texi", ".info"}:
        return "```\n" + body.decode("utf-8", errors="replace") + "\n```"
    if ext in {".adoc", ".asciidoc"}:
        return body.decode("utf-8", errors="replace")
    if ext in {".html", ".htm"}:
        md = (
            trafilatura.extract(body.decode("utf-8", errors="replace"), output_format="markdown")
            or ""
        )
        return md
    if ext in {".1", ".5", ".7", ".8"}:
        # mandoc render
        try:
            p = subprocess.run(
                ["mandoc", "-Tutf8"], input=body, capture_output=True, timeout=10, check=False
            )
            if p.returncode == 0:
                return "```\n" + p.stdout.decode("utf-8", errors="replace") + "\n```"
        except FileNotFoundError:
            pass
        return "```\n" + body.decode("utf-8", errors="replace") + "\n```"
    return body.decode("utf-8", errors="replace")


def fetch(source: PlannedSource, ctx: FetcherContext) -> FetchResult:
    if not shutil.which("git"):
        return FetchResult(skipped=True, skipped_reason="git not on PATH")
    paths_filter = source.paths
    with tempfile.TemporaryDirectory(prefix="kb-git-") as tmp:
        tmp_path = Path(tmp)
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    source.url,
                    str(tmp_path / "repo"),
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as e:
            return FetchResult(
                skipped=True,
                skipped_reason=f"git clone failed: {e.stderr.decode(errors='replace')[:200]}",
            )
        except subprocess.TimeoutExpired:
            return FetchResult(skipped=True, skipped_reason="git clone timeout")

        repo_root = tmp_path / "repo"
        docs: list[FetchedDoc] = []
        count = 0
        for file in repo_root.rglob("*"):
            if not file.is_file():
                continue
            if not _is_doc_path(file, paths_filter):
                continue
            if file.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                body = file.read_bytes()
            except Exception:  # noqa: S112  # reason: skip unreadable files; partial-KB is fine
                continue
            try:
                ctx.budget.add_page()
            except Exception:
                break
            rel = file.relative_to(repo_root).as_posix()
            md = _render(file, body)
            if len(md.strip()) < 80:
                continue
            doc_url = f"{source.url}#{rel}"
            docs.append(
                FetchedDoc(
                    url=doc_url,
                    raw_bytes=body,
                    raw_ext=file.suffix or ".txt",
                    body_markdown=f"# {rel}\n\n{md}\n",
                    title=rel,
                    license=source.license,
                    extra={"repo": source.url, "path": rel},
                )
            )
            count += 1
            if count >= MAX_FILES:
                break

    if not docs:
        return FetchResult(skipped=True, skipped_reason="no doc files found")
    return FetchResult(docs=docs)


register("git", fetch)
