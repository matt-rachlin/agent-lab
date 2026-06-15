# conf/_archive/

Historical backup copies of llama-swap.yaml and litellm-config.yaml from
prior phases (nemotron experiments, qwen30 ollama variant, etc.). Kept
for reproducibility of past sweep configs; not consumed by any live
service.

Per the P1.G3 .gitignore rule, `*.bak-*` files are now ignored — these
files predated that rule and are preserved deliberately. The pattern
going forward is `git stash`/branches for in-flight serving-config
changes, not `.bak-*` copies.
