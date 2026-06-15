"""Ed25519 signing for ADR-008 promotions and the action audit.

The private key is held OUTSIDE the agent's environment (research-agent-stage0
D5); the agent can verify but cannot sign verified/finding promotions.
"""

from __future__ import annotations

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def generate_keypair() -> tuple[bytes, bytes]:
    """(private_pem, public_pem). Store the private PEM off the agent host."""
    priv = Ed25519PrivateKey.generate()
    priv_pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub_pem = priv.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem


def sign(private_pem: bytes, message: str) -> str:
    priv = serialization.load_pem_private_key(private_pem, password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        raise TypeError("not an Ed25519 private key")
    return priv.sign(message.encode()).hex()


def verify(public_pem: bytes, message: str, signature_hex: str) -> bool:
    pub = serialization.load_pem_public_key(public_pem)
    if not isinstance(pub, Ed25519PublicKey):
        raise TypeError("not an Ed25519 public key")
    try:
        pub.verify(bytes.fromhex(signature_hex), message.encode())
    except (InvalidSignature, ValueError):
        return False
    return True
