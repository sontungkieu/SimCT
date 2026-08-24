from vdt_span.canary import run_reference_canary


def test_reference_canary_is_explicit_and_dynamic() -> None:
    result = run_reference_canary()
    assert result["schema_version"] == 1
    assert result["atomic_unit_count"] == 4
    assert result["dynamic_early"] != result["dynamic_late"]
    assert result["mass_preserving_kl_gap"] > 0.0
    assert "no model" in result["claim_boundary"]
