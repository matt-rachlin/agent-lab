"""Add a paper to ~/research/papers/ cache.

Usage:
    uv run python tools/add_paper.py 2404.16130                  # arXiv id
    uv run python tools/add_paper.py arxiv:2404.16130            # arXiv id (explicit)
    uv run python tools/add_paper.py 10.1145/3589334.3645708     # DOI
    uv run python tools/add_paper.py https://arxiv.org/abs/2404.16130
    uv run python tools/add_paper.py 2404.16130 --refresh-metadata
    uv run python tools/add_paper.py 2404.16130 --stub           # write stub PDF (offline)

For each paper, creates ~/research/papers/<id>/ containing:
    paper.pdf      The downloaded PDF (or stub bytes with --stub)
    meta.yaml      Doc-meta frontmatter + paper-specific fields
    notes.md       Blank notes scaffold with frontmatter

Idempotent: re-running on the same id is a no-op unless --refresh-metadata
is given. Metadata is fetched from:
    - arXiv API   (for arxiv ids — http://export.arxiv.org/api/query)
    - Crossref    (for DOIs    — https://api.crossref.org/works/<doi>)
    - URL scrape  (best effort)

The directory and slug used on disk is "paper-<id>" with non-filesystem-safe
characters replaced (e.g. paper-2404.16130, paper-10.1145-3589334.3645708).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from rich.console import Console

RESEARCH_DIR = Path.home() / "research"
PAPERS_DIR = RESEARCH_DIR / "papers"

# Recognized arxiv id forms:
#   2404.16130
#   2404.16130v2
#   arxiv:2404.16130
#   cs.CL/0301001  (very old style; we accept it)
ARXIV_RE = re.compile(
    r"^(?:arxiv[:/])?(?P<id>(?:\d{4}\.\d{4,5})(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})$",
    re.IGNORECASE,
)

# Recognized DOI forms: 10.NNNN/...
DOI_RE = re.compile(r"^(?:doi[:/])?(?P<doi>10\.\d{4,9}/\S+)$", re.IGNORECASE)

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


# --- parsers ---


@dataclass
class ParsedRef:
    """Parsed paper identifier."""

    kind: str  # "arxiv" | "doi" | "url"
    raw: str  # the original input
    canonical_id: str  # canonical id (arxiv id sans v-suffix, or DOI, or url)
    slug: str  # filesystem-safe slug for the directory


def parse_identifier(s: str) -> ParsedRef:
    """Detect whether s is an arXiv id, DOI, or URL; return canonical form.

    Raises ValueError if the input matches none of the patterns.
    """
    s = s.strip()
    if not s:
        raise ValueError("empty identifier")

    # URL? Try to peel out an arxiv id or DOI first.
    if s.startswith(("http://", "https://")):
        u = urlparse(s)
        host = (u.netloc or "").lower()
        path = u.path or ""
        # arxiv: /abs/<id> or /pdf/<id>(.pdf)
        if "arxiv.org" in host:
            m = re.search(r"/(?:abs|pdf)/([^/]+?)(?:\.pdf)?$", path)
            if m:
                aid = m.group(1)
                # strip vN suffix for canonical id
                aid_canon = re.sub(r"v\d+$", "", aid)
                return ParsedRef(
                    kind="arxiv",
                    raw=s,
                    canonical_id=aid_canon,
                    slug=_arxiv_slug(aid_canon),
                )
        # DOI URL: dx.doi.org/... or doi.org/...
        if "doi.org" in host:
            doi = path.lstrip("/")
            if doi:
                return ParsedRef(
                    kind="doi",
                    raw=s,
                    canonical_id=doi,
                    slug=_doi_slug(doi),
                )
        # generic URL fallback
        slug = "url-" + re.sub(r"[^a-z0-9]+", "-", host + path).strip("-").lower()[:60]
        return ParsedRef(kind="url", raw=s, canonical_id=s, slug=slug)

    m = ARXIV_RE.match(s)
    if m:
        aid = m.group("id")
        aid_canon = re.sub(r"v\d+$", "", aid)
        return ParsedRef(
            kind="arxiv",
            raw=s,
            canonical_id=aid_canon,
            slug=_arxiv_slug(aid_canon),
        )

    m = DOI_RE.match(s)
    if m:
        doi = m.group("doi")
        return ParsedRef(kind="doi", raw=s, canonical_id=doi, slug=_doi_slug(doi))

    raise ValueError(f"unrecognized identifier: {s!r}")


def _arxiv_slug(aid: str) -> str:
    """arxiv:2404.16130 -> paper-2404-16130; cs.CL/0301001 -> paper-cs-cl-0301001.

    Slug doubles as the doc_id used by m docs (kebab-case enforced).
    """
    safe = re.sub(r"[^a-z0-9]+", "-", aid.lower()).strip("-")
    return f"paper-{safe}"


def _doi_slug(doi: str) -> str:
    """10.1145/3589334.3645708 -> paper-10-1145-3589334-3645708."""
    safe = re.sub(r"[^a-z0-9]+", "-", doi.lower()).strip("-")
    return f"paper-{safe}"


# --- metadata fetchers ---


@dataclass
class PaperMeta:
    """The paper-specific subset of meta.yaml (separate from doc-meta header)."""

    title: str = "TODO: title"
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = "arxiv"
    abstract: str = ""
    arxiv_id: str | None = None
    doi: str | None = None
    source_url: str = ""
    tags: list[str] = field(default_factory=list)


def fetch_arxiv_meta(arxiv_id: str, *, fetcher: Any | None = None) -> PaperMeta:
    """Fetch paper metadata from arXiv. Returns a PaperMeta.

    `fetcher` is a callable(url) -> bytes (injected for tests). If None we
    use urllib.request, which means we depend on network.
    """
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    raw = _http_get(url, fetcher=fetcher)
    return _parse_arxiv_atom(raw, arxiv_id=arxiv_id)


def fetch_doi_meta(doi: str, *, fetcher: Any | None = None) -> PaperMeta:
    """Fetch paper metadata from Crossref. Returns a PaperMeta."""
    import json

    url = f"https://api.crossref.org/works/{doi}"
    raw = _http_get(url, fetcher=fetcher)
    data = json.loads(raw.decode("utf-8"))
    msg = data.get("message", {})
    title_list = msg.get("title", [])
    title = title_list[0] if title_list else "TODO: title"
    authors = []
    for a in msg.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        full = f"{given} {family}".strip()
        if full:
            authors.append(full)
    year = None
    issued = msg.get("issued", {}).get("date-parts", [[None]])
    if issued and issued[0] and issued[0][0]:
        year = int(issued[0][0])
    venue = msg.get("container-title", [""])
    venue_str = venue[0] if venue else ""
    abstract = msg.get("abstract", "") or ""
    return PaperMeta(
        title=title,
        authors=authors,
        year=year,
        venue=venue_str or "crossref",
        abstract=abstract,
        doi=doi,
        source_url=msg.get("URL", f"https://doi.org/{doi}"),
        tags=[],
    )


def _parse_arxiv_atom(raw: bytes, *, arxiv_id: str) -> PaperMeta:
    """Parse arXiv Atom XML response into PaperMeta."""
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)  # noqa: S314 — arxiv API, trusted source
    entry = root.find("atom:entry", ns)
    if entry is None:
        raise ValueError(f"no entry for arxiv id {arxiv_id!r}")
    title_el = entry.find("atom:title", ns)
    title = (title_el.text or "").strip().replace("\n ", " ") if title_el is not None else ""
    summary_el = entry.find("atom:summary", ns)
    abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
    published_el = entry.find("atom:published", ns)
    year = None
    if published_el is not None and published_el.text:
        try:
            year = int(published_el.text[:4])
        except ValueError:
            year = None
    authors: list[str] = []
    for a in entry.findall("atom:author", ns):
        name_el = a.find("atom:name", ns)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())
    return PaperMeta(
        title=title or "TODO: title",
        authors=authors,
        year=year,
        venue="arxiv",
        abstract=abstract,
        arxiv_id=arxiv_id,
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
        tags=[],
    )


def _http_get(url: str, *, fetcher: Any | None = None) -> bytes:
    """Tiny urllib wrapper so tests can inject a fetcher."""
    if fetcher is not None:
        return fetcher(url)  # type: ignore[no-any-return]
    from urllib.request import Request, urlopen

    req = Request(url, headers={"User-Agent": "lab-add-paper/1.0"})  # noqa: S310
    with urlopen(req, timeout=20) as resp:  # noqa: S310 — trusted hosts only
        return resp.read()  # type: ignore[no-any-return]


# --- writers ---


def render_meta_yaml(ref: ParsedRef, pm: PaperMeta) -> str:
    """Render the paper-specific meta.yaml.

    This file contains only the paper-specific fields (authors, year, abstract,
    etc.). Doc-meta frontmatter for `m docs scan` lives in the sibling notes.md
    file, which has 'depends_on: paper-<id>' linking to this paper.
    """
    lines: list[str] = []
    lines.append(f"doc_id: {ref.slug}")
    lines.append(f"title: {_yaml_scalar(pm.title)}")
    if pm.arxiv_id:
        # Quote — bare 2404.16130 would parse as a float in YAML.
        lines.append(f'arxiv_id: "{pm.arxiv_id}"')
    else:
        lines.append("arxiv_id: null")
    if pm.doi:
        lines.append(f'doi: "{pm.doi}"')
    else:
        lines.append("doi: null")
    lines.append("authors:")
    if pm.authors:
        for a in pm.authors:
            lines.append(f"  - {_yaml_scalar(a)}")
    else:
        lines.append("  []")
    if pm.year is not None:
        lines.append(f"year: {pm.year}")
    else:
        lines.append("year: null")
    lines.append(f"venue: {_yaml_scalar(pm.venue)}")
    if pm.source_url:
        lines.append(f"source_url: {_yaml_scalar(pm.source_url)}")
    lines.append("abstract: |")
    if pm.abstract:
        for line in pm.abstract.splitlines():
            lines.append(f"  {line.rstrip()}")
    else:
        lines.append("  (no abstract)")
    if pm.tags:
        lines.append("tags:")
        for t in pm.tags:
            lines.append(f"  - {t}")
    else:
        lines.append("tags: []")
    lines.append("")
    return "\n".join(lines)


def render_notes_md(ref: ParsedRef, pm: PaperMeta, *, today: date | None = None) -> str:
    """Render the indexable notes.md with doc-meta frontmatter."""
    today = today or date.today()
    # The notes doc IS the canonical research-zone paper doc.
    lines = ["---"]
    lines.append(f"doc_id: {ref.slug}")
    lines.append(f"title: {_yaml_scalar(pm.title)}")
    lines.append("zone: research")
    lines.append("kind: paper")
    lines.append("status: active")
    lines.append("owner: m")
    lines.append(f"created: {today.isoformat()}")
    lines.append(f"last_updated: {today.isoformat()}")
    lines.append(f"last_verified: {today.isoformat()}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {pm.title}")
    lines.append("")
    if pm.authors:
        lines.append(f"**Authors:** {', '.join(pm.authors)}")
    if pm.year:
        lines.append(f"**Year:** {pm.year}")
    if pm.venue:
        lines.append(f"**Venue:** {pm.venue}")
    if pm.arxiv_id:
        lines.append(f"**arXiv:** [{pm.arxiv_id}](https://arxiv.org/abs/{pm.arxiv_id})")
    if pm.doi:
        lines.append(f"**DOI:** [{pm.doi}](https://doi.org/{pm.doi})")
    lines.append("")
    lines.append("See `meta.yaml` for structured metadata and `paper.pdf` for the full text.")
    lines.append("")
    if pm.abstract:
        lines.append("## Abstract")
        lines.append("")
        lines.append(pm.abstract)
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("(write a few sentences)")
    lines.append("")
    lines.append("## Key claims")
    lines.append("")
    lines.append("- ")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append("- ")
    lines.append("")
    lines.append("## Open questions")
    lines.append("")
    lines.append("- ")
    lines.append("")
    return "\n".join(lines)


def _yaml_scalar(s: str) -> str:
    """Render a string as a YAML scalar — quote if it contains anything tricky."""
    if not s:
        return '""'
    # Need quoting if it contains characters YAML treats specially when bare.
    if re.search(r"[:#\[\]{},&*!|>'\"%@`]|^[\-\?]| {2,}|^\s|\s$", s):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


# --- main ---


def add_paper(
    identifier: str,
    *,
    papers_dir: Path | None = None,
    refresh_metadata: bool = False,
    stub_pdf: bool = False,
    pdf_bytes: bytes | None = None,
    meta_fetcher: Any | None = None,
    today: date | None = None,
) -> tuple[Path, bool]:
    """Add (or refresh) a paper. Returns (paper_dir, created)."""
    papers_dir = papers_dir or PAPERS_DIR
    ref = parse_identifier(identifier)
    paper_dir = papers_dir / ref.slug
    meta_path = paper_dir / "meta.yaml"
    notes_path = paper_dir / "notes.md"
    pdf_path = paper_dir / "paper.pdf"

    existed = paper_dir.exists()
    if existed and not refresh_metadata:
        # Idempotent no-op. Don't overwrite anything.
        return paper_dir, False

    paper_dir.mkdir(parents=True, exist_ok=True)

    # Fetch metadata
    if ref.kind == "arxiv":
        pm = fetch_arxiv_meta(ref.canonical_id, fetcher=meta_fetcher)
    elif ref.kind == "doi":
        pm = fetch_doi_meta(ref.canonical_id, fetcher=meta_fetcher)
    else:
        # URL fallback — minimal metadata
        pm = PaperMeta(
            title="TODO: title (from URL)",
            source_url=ref.canonical_id,
            tags=[],
        )

    # Write meta.yaml (overwrite on refresh; create on first run)
    meta_path.write_text(render_meta_yaml(ref, pm))

    # notes.md — only create if missing (don't clobber hand-edits)
    if not notes_path.exists():
        notes_path.write_text(render_notes_md(ref, pm, today=today))

    # PDF
    if not pdf_path.exists():
        if pdf_bytes is not None:
            pdf_path.write_bytes(pdf_bytes)
        elif stub_pdf:
            pdf_path.write_bytes(b"%PDF-1.4\n%stub - paper PDF not yet downloaded\n%%EOF\n")
        # otherwise: leave it absent (the script could fetch from arxiv pdf URL,
        # but we keep the network-touching path out of the default flow).

    return paper_dir, not existed


@app.command()
def main(
    identifier: str = typer.Argument(..., help="arXiv id, DOI, or URL"),
    refresh_metadata: bool = typer.Option(False, "--refresh-metadata"),
    stub: bool = typer.Option(False, "--stub", help="Write a stub PDF (no network fetch)"),
    papers_dir: Path = typer.Option(PAPERS_DIR, "--papers-dir"),
) -> None:
    """CLI entrypoint."""
    try:
        ref = parse_identifier(identifier)
    except ValueError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(2) from None
    console.print(f"[bold]parsed:[/bold] kind={ref.kind} id={ref.canonical_id} slug={ref.slug}")
    paper_dir, created = add_paper(
        identifier,
        papers_dir=papers_dir,
        refresh_metadata=refresh_metadata,
        stub_pdf=stub,
    )
    state = "created" if created else ("refreshed" if refresh_metadata else "already-exists")
    console.print(f"[green]{state}:[/green] {paper_dir}")


if __name__ == "__main__":
    app()
