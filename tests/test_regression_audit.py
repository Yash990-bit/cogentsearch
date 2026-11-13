"""
Regression and Audit Verification Test for CogentSearch Phase 1.
Runs the 50-element citation audit and verifies 100% pass rate.
"""

from scripts.run_citation_audit import run_audit


def test_citation_integrity_audit_50_elements():
    """Validates that 50 randomly sampled visual assets pass all 7 provenance checks."""
    audit_summary = run_audit(sample_size=50, seed=42)

    assert audit_summary["total_audited"] == 50
    assert audit_summary["failed"] == 0, f"Detected {audit_summary['failed']} citation audit mismatches: {audit_summary['mismatches']}"
    assert audit_summary["passed"] == 50
    assert audit_summary["pass_rate_percent"] == 100.0

