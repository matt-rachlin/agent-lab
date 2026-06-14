"""Stage 0b #10 — Ed25519 signing + the finding human-gate (ADR-008)."""

import pytest

from lab.core.signing import generate_keypair, sign, verify
from lab.core.trust import record_transition


def test_signing_roundtrip_and_tamper_detection():
    priv, pub = generate_keypair()
    sig = sign(priv, "promote run X to verified")
    assert verify(pub, "promote run X to verified", sig)
    assert not verify(pub, "promote run Y to verified", sig)


def test_finding_requires_human_or_signature():
    with pytest.raises(ValueError, match="finding"):
        record_transition("any-run", "finding", actor="system:verifier")
