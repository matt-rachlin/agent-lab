"""Stage 0 acceptance self-test (#11) — the substrate catches a KNOWN and a BLIND artefact.

research-agent-stage0 acceptance criterion 3: re-plant the F-017 tool_choice
artefact (designed-against) AND a blind artefact class the gates were not built
for, and assert the substrate refuses to certify both. A gate that only catches
the bug it was built from is Goodhart; this proves defense-in-depth (validity
gate + adversarial verifier).
"""

from lab.platform.trust import bfcl_validity
from lab.platform.verifier import BatteryResult, verdict


def _battery(**over: object) -> BatteryResult:
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


# --- Artefact 1: F-017 (designed-against) — caught at the validity gate ---


def test_f017_artefact_refused_by_validity_gate():
    # tool_choice=auto let a reasoning model answer in prose: no tool call, scored 0.
    rep = bfcl_validity(
        request_tools=None,  # tools not effectively sent
        tool_choice="auto",
        bfcl_error_type="model_output:no_tool_call",
        passed=False,
    )
    assert not rep.passed  # never reaches validity_passed
    assert any("expects tools" in v for v in rep.violations)
    assert rep.emitted is False  # a non-emission, NOT a wrong answer (decomposed)


def test_f017_class_pattern_flagged_by_verifier():
    # F-017 was class-correlated (reasoning models only) -> anchor inconsistent.
    assert verdict(_battery(anchor_consistent=False)).outcome == "artefact-suspected"


# --- Artefact 2: BLIND (prompt-sensitivity) — validity gate passes, verifier refutes ---


def test_blind_prompt_sensitivity_passes_validity_but_caught_by_verifier():
    # Request is faithful and output present, so the validity gate has no basis to
    # refuse — it PASSES (the gate was not designed for prompt-sensitivity).
    rep = bfcl_validity(
        request_tools=[{"type": "function", "function": {"name": "f"}}],
        tool_choice="required",
        bfcl_error_type=None,
        passed=True,
    )
    assert rep.passed
    # But the effect does not survive prompt perturbation -> the verifier refutes
    # it. The artefact is caught by a layer that never checked for it specifically.
    assert verdict(_battery(variant_effect_holds=False)).outcome == "artefact-suspected"


def test_clean_result_certifies():
    # Control: a genuinely sound result passes validity AND survives the battery.
    rep = bfcl_validity(
        request_tools=[{"type": "function", "function": {"name": "f"}}],
        tool_choice="required",
        bfcl_error_type=None,
        passed=True,
    )
    assert rep.passed
    assert verdict(_battery()).outcome == "holds"
