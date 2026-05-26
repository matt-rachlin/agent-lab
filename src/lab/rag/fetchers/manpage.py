"""manpage fetcher: uses local `man` / `mandoc`.

Vendored from kb_builder.fetchers.manpage.
"""

from __future__ import annotations

import re
import shutil
import subprocess

from lab.rag._plan import PlannedSource
from lab.rag.fetchers import FetchedDoc, FetcherContext, FetchResult, register


def _parse_man_url(url: str) -> tuple[str, str | None]:
    # man:NAME(N) or man:NAME or just "bash(1)"
    m = re.match(r"^(?:man:)?([A-Za-z0-9_.+-]+)(?:\((\d+)\))?$", url.strip())
    if not m:
        raise ValueError(f"unparseable manpage url: {url!r}")
    return m.group(1), m.group(2)


def _run_man(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run `man` with width/pager pinned, but inherit PATH and MANPATH so
    locally-installed manpages (e.g. from linuxbrew) remain discoverable.
    """
    import os

    env = os.environ.copy()
    env["MANWIDTH"] = "120"
    env["PAGER"] = "cat"
    return subprocess.run(args, env=env, capture_output=True, text=True, check=False)


def _man_text(name: str, section: str | None) -> tuple[str, str | None]:
    """Return (text, resolved_section). If section is given but missing,
    fall back to letting `man` pick whatever section is installed.
    """
    args = ["man", section, name] if section else ["man", name]
    p = _run_man(args)
    if p.returncode != 0 and section:
        # Try without section
        p2 = _run_man(["man", name])
        if p2.returncode == 0:
            # Find which section it landed in via `man -w` (whatis path)
            whereis = _run_man(["man", "-w", name])
            resolved: str | None = None
            if whereis.returncode == 0 and whereis.stdout.strip():
                m = re.search(r"/man(\d+\w*)/", whereis.stdout.strip())
                if m:
                    resolved = m.group(1)
            return re.sub(r".\x08", "", p2.stdout), resolved
        raise RuntimeError(p.stderr.strip() or p2.stderr.strip() or f"man {section} {name} failed")
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or f"man {name} failed")
    return re.sub(r".\x08", "", p.stdout), section


def _to_markdown(text: str, name: str, section: str | None) -> str:
    sec = f"({section})" if section else ""
    out = [f"# {name}{sec} — manpage\n"]
    in_block = False
    for line in text.splitlines():
        stripped = line.rstrip()
        # ALL-CAPS section heading at column 0
        if re.match(r"^[A-Z][A-Z0-9 _-]{2,}$", stripped) and not line.startswith(" "):
            if in_block:
                out.append("```")
                in_block = False
            out.append(f"\n## {stripped.strip().title()}\n")
            continue
        # subsection (indented capitalized line)
        if re.match(r"^   [A-Z][A-Za-z0-9 _-]{2,}$", line) and not line.startswith("       "):
            if in_block:
                out.append("```")
                in_block = False
            out.append(f"\n### {line.strip()}\n")
            continue
        out.append(line)
    if in_block:
        out.append("```")
    return "\n".join(out)


def fetch(source: PlannedSource, ctx: FetcherContext) -> FetchResult:
    if not shutil.which("man"):
        return FetchResult(skipped=True, skipped_reason="man not on PATH")
    try:
        name, section = _parse_man_url(source.url)
        text, resolved = _man_text(name, section)
    except Exception as e:
        return FetchResult(skipped=True, skipped_reason=f"manpage error: {e}")
    effective_section = resolved or section
    body = _to_markdown(text, name, effective_section)
    sec = f"({effective_section})" if effective_section else ""
    doc = FetchedDoc(
        url=f"man:{name}{sec}",
        raw_bytes=text.encode("utf-8"),
        raw_ext=".txt",
        body_markdown=body,
        title=f"{name}{sec} manpage",
        license="varies (local manpage)",
    )
    ctx.budget.add_page()
    return FetchResult(docs=[doc])


register("manpage", fetch)
