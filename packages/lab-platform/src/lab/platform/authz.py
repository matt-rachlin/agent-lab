"""Action authorization — ADR-013 enforcement layer.

The Lab Agent Runtime (LAR, ADR-012) already had a minimal gate: a per-run
allow-set over side-effect classes. ADR-013 upgrades that single boolean into a
**tiered** decision per (actor, capability, side-effect class):

    auto -> allow          | runtime executes without asking
    human-approve          | runtime pauses, asks a human, executes on approval
    dry-run                | execute against a shadow; never touch the real target
    deny                   | refuse

This module owns the *policy* (which class maps to which tier) and the *decision*
machinery; the LAR owns *enforcement* (calling `decide`, then executing / asking /
shadowing / refusing, and auditing the decision via `record_action`).

Default gate by class (ADR-013 §2):
    read           -> allow
    external_read  -> allow   (egress allowlist is enforced elsewhere / #13)
    write_local    -> require_approval, unless auto-granted for (actor, class)
    irreversible   -> require_approval, NEVER auto in v0

Fail-closed: the default approver denies. require_approval with no wired human
approver therefore resolves to "do not execute" — never a silent allow.

LIVE vs STUBBED
---------------
LIVE here (pure, no I/O, fully unit-tested):
  * the tier policy (`DefaultPolicy.decide`) and its ADR-013 defaults,
  * the auto-grant set model (`AuthzPolicy.grants`),
  * the approval-hook contract (`ApprovalCallback`, fail-closed default),
  * the earned-autonomy ratchet *rule* (`Ratchet.record` / `should_ratchet` /
    incident reset), operating over an in-memory `RatchetStore`.

STUBBED (documented seam, NOT implemented here):
  * a *persistent* ratchet store. The live ratchet state in this module is the
    in-memory `InMemoryRatchetStore`, which is per-process and lost on restart.
    A production ratchet must persist counts on the append-only audit chain and
    require an Ed25519-signed promotion to actually flip a gate (ADR-013 §3). That
    needs a DB table + migration and is deliberately NOT fabricated here. The
    `RatchetStore` Protocol is the seam a DB-backed implementation plugs into; the
    promotion-signature check is also out of scope for v0 policy code.
  * the human-approve transport (ntfy / Bridge -> signed approval). Here it is
    abstracted behind `ApprovalCallback`; the real signed-approval flow is wired
    by the runtime/Bridge, not by this policy module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# ADR-013 side-effect classes (kept loose as str to match the LAR's `SideEffect`
# Literal without a hard import cycle; the runtime validates the concrete values).
SideEffect = Literal["read", "external_read", "write_local", "irreversible"]

#: The four authorization tiers (ADR-013 §2). "allow" is the resolved form of the
#: spec's `auto` gate; the others map 1:1 to the named gates.
Decision = Literal["allow", "require_approval", "dry_run", "deny"]

#: Classes that auto-execute by default (ADR-013 default gate).
_AUTO_BY_DEFAULT: frozenset[str] = frozenset({"read", "external_read"})

#: Reversible-but-mutating; require_approval until earned via the ratchet.
_EARNABLE: frozenset[str] = frozenset({"write_local"})

#: Never auto in v0 — always require_approval regardless of grants/ratchet.
_NEVER_AUTO: frozenset[str] = frozenset({"irreversible"})


# --------------------------------------------------------------------------- #
# Approval hook
# --------------------------------------------------------------------------- #

#: A human-approval hook. Receives a request dict (actor / tool / class / args /
#: capability) and returns True to approve, False to deny. Fail-closed: the
#: default approver (`deny_approver`) always returns False, so an unwired
#: require_approval never silently executes.
ApprovalCallback = Callable[[dict[str, Any]], bool]


def deny_approver(_request: dict[str, Any]) -> bool:
    """Default approver: deny everything (fail-closed)."""
    return False


# --------------------------------------------------------------------------- #
# Earned-autonomy ratchet (ADR-013 §3)
# --------------------------------------------------------------------------- #


@dataclass
class RatchetState:
    """Per-(actor, class, workflow) clean-streak counter.

    `clean_streak` is the number of consecutive approved actions with zero
    overrides/incidents. `ratcheted` records that the streak crossed the
    threshold AND (in production) a signed promotion flipped the gate; in this
    v0 policy module the signed-promotion step is stubbed (see module docstring),
    so `ratcheted` reflects only the streak condition.
    """

    clean_streak: int = 0
    ratcheted: bool = False


@runtime_checkable
class RatchetStore(Protocol):
    """Seam for ratchet persistence. The live default is `InMemoryRatchetStore`;
    a DB-backed store (audit-chain + signed promotion) plugs in here."""

    def get(self, actor: str, side_effect: str, workflow: str) -> RatchetState: ...

    def put(self, actor: str, side_effect: str, workflow: str, state: RatchetState) -> None: ...


@dataclass
class InMemoryRatchetStore:
    """In-memory ratchet store (per-process, lost on restart). STUB for the
    persistent store — fine for tests and dry-runs, not for production grants."""

    _states: dict[tuple[str, str, str], RatchetState] = field(default_factory=dict)

    def get(self, actor: str, side_effect: str, workflow: str) -> RatchetState:
        return self._states.get((actor, side_effect, workflow), RatchetState())

    def put(self, actor: str, side_effect: str, workflow: str, state: RatchetState) -> None:
        self._states[(actor, side_effect, workflow)] = state


@dataclass
class Ratchet:
    """The earned-autonomy *rule*, over a `RatchetStore`.

    N consecutive clean approved actions of a class on a workflow -> the gate may
    ratchet require_approval -> allow. ANY incident (override / kill / safety veto)
    resets the streak and revokes the ratchet (ADR-013 §3). `irreversible` never
    ratchets in v0.
    """

    threshold: int = 5
    store: RatchetStore = field(default_factory=InMemoryRatchetStore)

    def record_clean(self, actor: str, side_effect: str, workflow: str) -> RatchetState:
        """Record one clean, approved action; ratchet if the threshold is met."""
        if side_effect in _NEVER_AUTO:
            # irreversible never ratchets; keep a streak for audit but never flip.
            st = self.store.get(actor, side_effect, workflow)
            st = RatchetState(clean_streak=st.clean_streak + 1, ratcheted=False)
            self.store.put(actor, side_effect, workflow, st)
            return st
        st = self.store.get(actor, side_effect, workflow)
        streak = st.clean_streak + 1
        st = RatchetState(clean_streak=streak, ratcheted=streak >= self.threshold)
        self.store.put(actor, side_effect, workflow, st)
        return st

    def record_incident(self, actor: str, side_effect: str, workflow: str) -> RatchetState:
        """Any incident resets the streak and revokes the ratchet."""
        st = RatchetState(clean_streak=0, ratcheted=False)
        self.store.put(actor, side_effect, workflow, st)
        return st

    def is_ratcheted(self, actor: str, side_effect: str, workflow: str) -> bool:
        """True if this (actor, class, workflow) has earned auto. Never True for
        `irreversible` (v0)."""
        if side_effect in _NEVER_AUTO:
            return False
        return self.store.get(actor, side_effect, workflow).ratcheted


# --------------------------------------------------------------------------- #
# Policy + Authorizer
# --------------------------------------------------------------------------- #


@runtime_checkable
class Authorizer(Protocol):
    """Resolves a side-effect class to an ADR-013 tier for a given actor/tool."""

    def decide(self, actor: str, tool_name: str, side_effect: str, capability: str) -> Decision: ...


@dataclass
class AuthzPolicy:
    """Default ADR-013 policy.

    `grants` is the per-(actor, side-effect class) auto-grant set: an explicit,
    deny-by-default override letting an earnable class (`write_local`) resolve to
    `allow` without going through the ratchet — e.g. an operator-signed grant.
    `irreversible` is never granted auto in v0 even if present in `grants`.

    `ratchet`, when set, lets an *earned* streak flip `write_local` ->
    `require_approval` to `allow` automatically (ADR-013 §3). The (actor, class,
    workflow) key uses `workflow` (default ""); pass the deployed workflow id to
    scope ratchets per-workflow as the ADR requires.

    `force_dry_run` resolves any otherwise-allowed/approved write to `dry_run`
    (shadow execution) — the safe default for un-trusted rollout.
    """

    grants: set[tuple[str, str]] = field(default_factory=set)
    ratchet: Ratchet | None = None
    workflow: str = ""
    force_dry_run: bool = False

    def _auto_granted(self, actor: str, side_effect: str) -> bool:
        if side_effect in _NEVER_AUTO:  # irreversible: never auto in v0
            return False
        if (actor, side_effect) in self.grants:
            return True
        return self.ratchet is not None and self.ratchet.is_ratcheted(
            actor, side_effect, self.workflow
        )

    def decide(self, actor: str, tool_name: str, side_effect: str, capability: str) -> Decision:
        # Unknown class -> deny (fail-closed).
        if side_effect not in (_AUTO_BY_DEFAULT | _EARNABLE | _NEVER_AUTO):
            return "deny"
        # read / external_read: auto.
        if side_effect in _AUTO_BY_DEFAULT:
            return "allow"
        # From here it's a mutating class. Optional global dry-run shadow.
        if self.force_dry_run:
            return "dry_run"
        # write_local: allow if granted/earned, else require_approval.
        # irreversible: always require_approval (never auto in v0).
        if self._auto_granted(actor, side_effect):
            return "allow"
        return "require_approval"


def default_authorizer() -> AuthzPolicy:
    """ADR-013 defaults, no grants, no ratchet, fail-closed."""
    return AuthzPolicy()
