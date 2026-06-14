"""Stage 0b #9 — adversarial verifier verdict logic (ADR-008)."""

from lab.core.verifier import BatteryResult, verdict


def _full(**over: object) -> BatteryResult:
    base: dict[str, object] = {
        "n_seeds": 16,
        "seed_effect_holds": True,
        "n_prompt_variants": 5,
        "variant_effect_holds": True,
        "n_regrade_paths": 2,
        "regraders_agree": True,
        "anchors_per_class": {"reasoning": 2, "non_reasoning": 2},
        "anchor_consistent": True,
    }
    base.update(over)
    return BatteryResult(**base)  # type: ignore[arg-type]


def test_holds_when_all_survive():
    assert verdict(_full()).outcome == "holds"


def test_artefact_when_anchor_inconsistent():
    v = verdict(_full(anchor_consistent=False))
    assert v.outcome == "artefact-suspected"
    assert "anchor_consistent" in v.reasons[0]


def test_artefact_when_regraders_disagree():
    assert verdict(_full(regraders_agree=False)).outcome == "artefact-suspected"


def test_inconclusive_when_seed_coverage_low():
    assert verdict(_full(n_seeds=4)).outcome == "inconclusive"


def test_inconclusive_when_single_model_class():
    assert verdict(_full(anchors_per_class={"reasoning": 2})).outcome == "inconclusive"
