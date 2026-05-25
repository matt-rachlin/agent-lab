"""Podman + gVisor sandbox. Implementation lands in 6b."""

from __future__ import annotations

from typing import Any


class Sandbox:
    def __init__(self, image: str, network: str = "none", **kwargs: Any) -> None:
        raise NotImplementedError("6b — not yet implemented")

    def start(self) -> None:
        raise NotImplementedError("6b — not yet implemented")

    def exec(self, cmd: list[str], timeout: int | None = None) -> tuple[int, str, str]:
        raise NotImplementedError("6b — not yet implemented")

    def stop(self) -> None:
        raise NotImplementedError("6b — not yet implemented")
