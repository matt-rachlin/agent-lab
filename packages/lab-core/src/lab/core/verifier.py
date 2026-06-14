"""ADR-008 adversarial verifier: minimum refutation battery + verdict.

Tries to REFUTE a reliability_confirmed candidate; only an unbroken result
reaches `verified`. Thresholds are the conservative v0 from
docs/protocols/research-agent-stage0 (Resolved decisions): >=16 seeds, >=5
prompt variants, >=2 independent re-grade paths, >=2 anchors per model-class.
The battery EXECUTION (re-running seeds/variants/anchors — GPU/cloud-heavy) is
injected via `battery_runner`; the verdict logic here is pure and tested.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from lab.core.trust import record_transition


@dataclass
class BatteryConfig:
    min_seeds: int = 16
    min_prompt_variants: int = 5
    min_regrade_paths: int = 2
    min_anchors_per_class: int = 2
    min_classes: int = 2


@dataclass
class BatteryResult:
    n_seeds: int
    seed_effect_holds: bool
    n_prompt_variants: int
    variant_effect_holds: bool
    n_regrade_paths: int
    regraders_agree: bool
    anchors_per_class: dict[str, int]
    anchor_consistent: bool


@dataclass
class Verdict:
    outcome: str  # 'holds' | 'artefact-suspected' | 'inconclusive'
    reasons: list[str]


def verdict(br: BatteryResult, cfg: BatteryConfig | None = None) -> Verdict:
    """Apply the conservative-v0 rule: meet coverage, then survive every
    component unanimously. A class-correlated anchor outlier (the F-017 pattern)
    fails `anchor_consistent` -> artefact-suspected, not a model deficit."""
    cfg = cfg or BatteryConfig()
    coverage_ok = (
        br.n_seeds >= cfg.min_seeds
        and br.n_prompt_variants >= cfg.min_prompt_variants
        and br.n_regrade_paths >= cfg.min_regrade_paths
        and len(br.anchors_per_class) >= cfg.min_classes
        and all(n >= cfg.min_anchors_per_class for n in br.anchors_per_class.values())
    )
    if not coverage_ok:
        return Verdict("inconclusive", ["battery coverage below minimum; cannot certify"])
    checks = {
        "seed_effect_holds": br.seed_effect_holds,
        "variant_effect_holds": br.variant_effect_holds,
        "regraders_agree": br.regraders_agree,
        "anchor_consistent": br.anchor_consistent,
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        return Verdict("artefact-suspected", ["refuted by: " + ", ".join(failed)])
    return Verdict("holds", ["survived all battery components"])


BatteryRunner = Callable[[str, BatteryConfig], BatteryResult]


def _live_battery_runner(run_id: str, cfg: BatteryConfig) -> BatteryResult:
    raise NotImplementedError(
        "live refutation battery (re-running seeds/variants/anchors) is GPU/cloud-heavy "
        "and not yet wired; inject a battery_runner or use the forthcoming sweep integration"
    )


def verify_candidate(
    run_id: str,
    *,
    battery_runner: BatteryRunner = _live_battery_runner,
    cfg: BatteryConfig | None = None,
    actor: str = "system:verifier",
) -> Verdict:
    """Run the battery on a candidate and record the transition: holds ->
    verified; otherwise -> verification_attempted carrying the verdict."""
    cfg = cfg or BatteryConfig()
    br = battery_runner(run_id, cfg)
    v = verdict(br, cfg)
    to_level = "verified" if v.outcome == "holds" else "verification_attempted"
    record_transition(
        run_id,
        to_level,
        actor=actor,
        evidence={"verdict": v.outcome, "reasons": v.reasons, "battery": asdict(br)},
    )
    return v
