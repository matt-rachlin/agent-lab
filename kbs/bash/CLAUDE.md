---
doc_id: kbs-bash-claude
title: kbs/bash — agent notes
zone: lab
kind: claude
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
tags:
- lab
- rag
- agent
- claude
---
# `kbs/bash/` — agent notes

## When to query this KB

- User question references bash builtins, parameter expansion, redirection,
  process substitution, traps, job control, conditional expressions, arrays,
  associative arrays, brace expansion, `select`, or `getopts`.
- ShellCheck warning codes (`SC2xxx`) — the wiki is indexed under sources.
- Quoting rules, IFS pitfalls, scripting style.

## When NOT to query

- POSIX-sh portability questions where bash extensions are off the table —
  this KB is bash-flavored.
- Zsh, fish, dash idioms — not in scope.
- Generic shell-tool questions (jq, awk, sed) — those need their own KBs.

## Retrieval tips

- The index uses qwen3-embedding:8b-q8_0 (4096-dim). Prefer phrase-level
  queries over single keywords; the embeddings cluster well for
  multi-word technical phrases.
- HyPE-style augmented chunks live in `chunks/chunks.enriched.jsonl` —
  the `lab.rag` retriever picks the right file automatically.

## Known gaps

- Bash 5.3 features (added late 2025) may be undercovered if the GNU manual
  snapshot pre-dates them. Check `manifest.yaml` `last_refreshed_at`.
- ShellCheck wiki coverage stops at the rules present on the indexed commit;
  newer SC codes won't be there until a refresh.

## Citation style in answers

When citing this KB in a response, prefer the original source URL from the
chunk metadata (it's preserved in the chunk JSON). Don't cite the KB index
hash — cite the upstream document.
