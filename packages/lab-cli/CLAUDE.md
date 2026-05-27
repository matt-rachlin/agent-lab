# lab-cli

CLI entrypoint + state management:
- `lab.cli` — Typer app, all `lab <subcommand>` handlers
- `lab.experiment` — pre-registration plans, slug/heading helpers
- `lab.finding` — `findings/YYYY-MM-DD-*.md` lifecycle
- `lab.models.register` — local-model id parsing/registration

## Gotchas
- This package transitively depends on everything; importing it eagerly costs ~1s of import time. Subcommands lazy-import where possible.
- `models/register.py` houses local-model id parsing (`_split_variant_quant`, etc.) that tests poke at directly.
