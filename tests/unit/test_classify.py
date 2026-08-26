# tests/unit/test_classify.py
"""Tests for repeat unit classification."""

from __future__ import annotations

import pytest

from open_pacmuci.classify import (
    _apply_bidirectional_fallback,
    _compute_classification_summary,
    _forward_classify,
    _qual_to_confidence,
    characterize_differences,
    classify_repeat,
    classify_sequence,
    edit_distance,
    reconcile_shared_frameshifts,
    validate_mutations_against_vcf,
)
from open_pacmuci.config import load_repeat_dictionary


@pytest.fixture
def repeat_dict():
    """Load bundled repeat dictionary."""
    return load_repeat_dictionary()


class TestEditDistance:
    """Tests for edit distance calculation."""

    def test_identical_sequences(self):
        """Edit distance of identical sequences is 0."""
        assert edit_distance("ACGT", "ACGT") == 0

    def test_single_substitution(self):
        """Single base substitution gives distance 1."""
        assert edit_distance("ACGT", "ACGA") == 1

    def test_single_insertion(self):
        """Single insertion gives distance 1."""
        assert edit_distance("ACGT", "ACGAT") == 1

    def test_single_deletion(self):
        """Single deletion gives distance 1."""
        assert edit_distance("ACGT", "ACT") == 1

    def test_large_indel(self):
        """Multi-base indel is counted correctly."""
        # 59dupC equivalent: inserting one C into a 60bp sequence
        seq_a = "G" * 60
        seq_b = "G" * 59 + "CG"  # insert C before last G
        assert edit_distance(seq_a, seq_b) == 1


class TestCharacterizeDifferences:
    """Tests for characterizing differences between sequences."""

    def test_single_insertion(self):
        """Detects a single insertion."""
        ref = "ACGT"
        query = "ACGCT"  # C inserted at position 3
        diffs = characterize_differences(ref, query)
        has_insertion = any(d["type"] == "insertion" for d in diffs)
        assert has_insertion

    def test_single_substitution(self):
        """Detects a single substitution."""
        ref = "ACGT"
        query = "ACAT"  # G->A at position 2
        diffs = characterize_differences(ref, query)
        has_sub = any(d["type"] == "substitution" for d in diffs)
        assert has_sub

    def test_identifies_frameshift(self):
        """Insertion of 1bp is a frameshift (1 % 3 != 0)."""
        ref = "ACGT"
        query = "ACGAT"
        diffs = characterize_differences(ref, query)
        total_indel = sum(1 for d in diffs if d["type"] in ("insertion", "deletion"))
        assert total_indel % 3 != 0  # frameshift

    def test_no_differences(self):
        """Identical sequences produce empty diff list."""
        diffs = characterize_differences("ACGT", "ACGT")
        assert diffs == []


class TestClassifyRepeat:
    """Tests for classifying a single 60bp repeat unit."""

    def test_exact_match(self, repeat_dict):
        """Known repeat X matches exactly."""
        x_seq = repeat_dict.repeats["X"]
        result = classify_repeat(x_seq, repeat_dict)
        assert result["type"] == "X"
        assert result["match"] == "exact"

    def test_known_pre_repeat(self, repeat_dict):
        """Pre-repeat 1 matches exactly."""
        seq = repeat_dict.repeats["1"]
        result = classify_repeat(seq, repeat_dict)
        assert result["type"] == "1"
        assert result["match"] == "exact"

    def test_unknown_with_insertion(self, repeat_dict):
        """Repeat X with novel 1bp insertion is classified as mutation."""
        x_seq = repeat_dict.repeats["X"]
        # Novel insertion at position 30 (not in any mutation template)
        mutated = x_seq[:30] + "T" + x_seq[30:]  # 61bp, novel
        result = classify_repeat(mutated, repeat_dict)
        assert result["match"] != "exact"
        assert result["closest_match"] == "X"
        assert result["edit_distance"] == 1
        assert any(d["type"] == "insertion" for d in result["differences"])

    def test_unknown_with_large_deletion(self, repeat_dict):
        """Repeat X with 14bp deletion matches del18_31 template exactly."""
        x_seq = repeat_dict.repeats["X"]
        # Simulate del18_31: delete positions 18-31 (1-based)
        mutated = x_seq[:17] + x_seq[31:]
        result = classify_repeat(mutated, repeat_dict)
        # Now matches the del18_31 mutation template exactly
        assert result["match"] == "exact"
        assert result["type"] == "X:del18_31"

    def test_identity_pct_reported(self, repeat_dict):
        """Identity percentage is included in result for novel mutations."""
        x_seq = repeat_dict.repeats["X"]
        # Novel insertion (not in template catalog)
        mutated = x_seq[:30] + "T" + x_seq[30:]
        result = classify_repeat(mutated, repeat_dict)
        assert "identity_pct" in result
        assert result["identity_pct"] > 90.0


class TestClassifySequence:
    """Tests for classifying a full consensus sequence."""

    def test_all_canonical_x(self, repeat_dict):
        """Sequence of 3 canonical X repeats classified correctly."""
        x_seq = repeat_dict.repeats["X"]
        full_seq = x_seq * 3
        result = classify_sequence(full_seq, repeat_dict)
        assert len(result["repeats"]) == 3
        assert all(r["type"] == "X" for r in result["repeats"])
        assert result["structure"] == "X X X"

    def test_mutation_detected(self, repeat_dict):
        """Novel mutation in one repeat is detected and reported."""
        x_seq = repeat_dict.repeats["X"]
        # Novel 2bp insertion (not in template catalog) -> lands in mutations_detected
        mutated = x_seq[:40] + "TT" + x_seq[40:]  # 62bp
        full_seq = x_seq + mutated + x_seq
        result = classify_sequence(full_seq, repeat_dict)
        assert len(result["mutations_detected"]) >= 1


class TestConfidenceScoring:
    """Tests for per-repeat confidence scores."""

    def test_exact_match_confidence_is_1(self, repeat_dict):
        """Exact match to known repeat type has confidence 1.0."""
        x_seq = repeat_dict.repeats["X"]
        result = classify_repeat(x_seq, repeat_dict)
        assert result["confidence"] == 1.0

    def test_close_match_confidence_uses_identity(self, repeat_dict):
        """Near-match (ed=1) confidence equals identity_pct / 100."""
        x_seq = repeat_dict.repeats["X"]
        # Novel insertion not in template catalog
        mutated = x_seq[:30] + "T" + x_seq[30:]  # 61bp, ed=1
        result = classify_repeat(mutated, repeat_dict)
        assert 0.9 < result["confidence"] < 1.0
        assert result["confidence"] == result["identity_pct"] / 100

    def test_distant_match_has_lower_confidence(self, repeat_dict):
        """High edit distance produces lower confidence."""
        seq = "ACGT" * 15
        result = classify_repeat(seq, repeat_dict)
        assert result["confidence"] < 0.9


class TestSequenceConfidenceSummary:
    """Tests for per-allele confidence summary in classify_sequence."""

    def test_all_exact_gives_confidence_1(self, repeat_dict):
        """All exact matches produce allele_confidence=1.0."""
        x_seq = repeat_dict.repeats["X"]
        result = classify_sequence(x_seq * 3, repeat_dict)
        assert result["allele_confidence"] == 1.0
        assert result["exact_match_pct"] == 100.0

    def test_mixed_match_reduces_confidence(self, repeat_dict):
        """One novel mutation among exact matches reduces allele_confidence below 1.0."""
        x_seq = repeat_dict.repeats["X"]
        # Novel 2bp insertion (not in template catalog)
        mutated = x_seq[:40] + "TT" + x_seq[40:]  # 62bp
        result = classify_sequence(x_seq + mutated + x_seq, repeat_dict)
        assert result["allele_confidence"] < 1.0
        assert result["exact_match_pct"] < 100.0


class TestMutationTemplateMatching:
    """Tests for exact matching against known mutation templates."""

    def test_dupc_exact_template_match(self, repeat_dict):
        """dupC on X matches the pre-computed 61bp template exactly."""
        from open_pacmuci.config import _apply_mutation

        x_seq = repeat_dict.repeats["X"]
        dupc_seq = _apply_mutation(x_seq, repeat_dict.mutations["dupC"]["changes"])
        assert len(dupc_seq) == 61
        result = classify_repeat(dupc_seq, repeat_dict)
        assert result["match"] == "exact"
        assert result["type"] == "X:dupC"
        assert result["confidence"] == 1.0

    def test_ins16bp_exact_template_match(self, repeat_dict):
        """16bp insertion on C matches the 76bp template exactly."""
        from open_pacmuci.config import _apply_mutation

        c_seq = repeat_dict.repeats["C"]
        ins_seq = _apply_mutation(c_seq, repeat_dict.mutations["ins16bp"]["changes"])
        assert len(ins_seq) == 76
        result = classify_repeat(ins_seq, repeat_dict)
        assert result["match"] == "exact"
        assert result["type"] == "C:ins16bp"
        assert result["confidence"] == 1.0

    def test_del18_31_exact_template_match(self, repeat_dict):
        """14bp deletion on X matches the 46bp template exactly."""
        from open_pacmuci.config import _apply_mutation

        x_seq = repeat_dict.repeats["X"]
        del_seq = _apply_mutation(x_seq, repeat_dict.mutations["del18_31"]["changes"])
        assert len(del_seq) == 46
        result = classify_repeat(del_seq, repeat_dict)
        assert result["match"] == "exact"
        assert result["type"] == "X:del18_31"

    def test_sequence_with_dupc_classifies_correctly(self, repeat_dict):
        """classify_sequence finds dupC at correct window size."""
        from open_pacmuci.config import _apply_mutation

        x_seq = repeat_dict.repeats["X"]
        dupc_seq = _apply_mutation(x_seq, repeat_dict.mutations["dupC"]["changes"])
        full = x_seq + dupc_seq + x_seq
        result = classify_sequence(full, repeat_dict)
        assert len(result["repeats"]) == 3
        assert result["repeats"][1]["type"] == "X:dupC"
        assert result["repeats"][1]["match"] == "exact"

    def test_sequence_with_16bp_ins_classifies_correctly(self, repeat_dict):
        """classify_sequence handles 76bp mutated repeat with template match."""
        from open_pacmuci.config import _apply_mutation

        x_seq = repeat_dict.repeats["X"]
        c_seq = repeat_dict.repeats["C"]
        ins_seq = _apply_mutation(c_seq, repeat_dict.mutations["ins16bp"]["changes"])
        full = x_seq + ins_seq + x_seq
        result = classify_sequence(full, repeat_dict)
        assert len(result["repeats"]) == 3
        assert result["repeats"][1]["type"] == "C:ins16bp"


class TestVcfMutationValidation:
    """Tests for VCF-backed mutation validation."""

    def test_confirmed_mutation_keeps_high_confidence(self, repeat_dict):
        """Mutation with VCF support keeps confidence unchanged."""
        x_seq = repeat_dict.repeats["X"]
        # Novel 2bp insertion (not in template catalog) -> lands in mutations_detected
        mutated = x_seq[:40] + "TT" + x_seq[40:]  # 62bp
        result = classify_sequence(x_seq + mutated + x_seq, repeat_dict)

        # Simulate VCF with a variant at repeat 2 position
        vcf_variants = [{"pos": 560, "qual": 25.0}]  # flank(500) + 60bp

        validated = validate_mutations_against_vcf(
            result, vcf_variants=vcf_variants, flank_length=500, unit_length=60
        )
        mut = validated["mutations_detected"][0]
        assert mut.get("vcf_support") is True

    def test_unsupported_mutation_gets_low_confidence(self, repeat_dict):
        """Mutation without VCF support gets reduced confidence."""
        x_seq = repeat_dict.repeats["X"]
        # Novel 2bp insertion (not in template catalog)
        mutated = x_seq[:40] + "TT" + x_seq[40:]  # 62bp
        result = classify_sequence(x_seq + mutated + x_seq, repeat_dict)

        # No VCF variants at all
        validated = validate_mutations_against_vcf(
            result, vcf_variants=[], flank_length=500, unit_length=60
        )
        mut = validated["mutations_detected"][0]
        assert mut.get("vcf_support") is False

    def test_no_mutations_unchanged(self, repeat_dict):
        """Sequence with no mutations passes through unchanged."""
        x_seq = repeat_dict.repeats["X"]
        result = classify_sequence(x_seq * 3, repeat_dict)

        validated = validate_mutations_against_vcf(
            result, vcf_variants=[], flank_length=500, unit_length=60
        )
        assert validated["mutations_detected"] == []


class TestBidirectionalClassification:
    """Tests for bidirectional classification fallback."""

    def test_novel_large_insertion_classified_via_bidirectional(self, repeat_dict):
        """Novel large insertion (not in templates) uses bidirectional fallback."""
        x_seq = repeat_dict.repeats["X"]
        pre1 = repeat_dict.repeats["1"]
        after9 = repeat_dict.repeats["9"]
        # Insert 20bp of random sequence into X (novel, not in templates)
        novel_mutated = x_seq[:30] + "A" * 20 + x_seq[30:]  # 80bp
        full = pre1 + x_seq + novel_mutated + x_seq + after9
        result = classify_sequence(full, repeat_dict)
        # Should classify pre1 and after9 correctly
        labels = result["structure"].split()
        assert labels[0] == "1"
        assert labels[-1] == "9"
        # The novel mutation should be detected
        assert len(result["mutations_detected"]) >= 1 or any("m" in label for label in labels)

    def test_forward_only_when_no_large_mutation(self, repeat_dict):
        """No bidirectional needed when all repeats classify well."""
        x_seq = repeat_dict.repeats["X"]
        result = classify_sequence(x_seq * 5, repeat_dict)
        assert all(r["match"] == "exact" for r in result["repeats"])
        assert result["allele_confidence"] == 1.0


class TestQualToConfidence:
    """Tests for _qual_to_confidence continuous scoring function."""

    def test_high_qual_returns_1(self):
        """QUAL >= 20 returns confidence weight 1.0."""
        assert _qual_to_confidence(20.0) == 1.0
        assert _qual_to_confidence(25.0) == 1.0
        assert _qual_to_confidence(100.0) == 1.0

    def test_mid_qual_interpolates(self):
        """QUAL between 5 and 20 interpolates linearly between 0.5 and 1.0."""
        # QUAL=12.5 is midpoint of [5, 20] → should give 0.75
        assert _qual_to_confidence(12.5) == pytest.approx(0.75, abs=0.01)

    def test_qual_5_returns_0_5(self):
        """QUAL=5.0 returns 0.5 (lower bound of interpolation)."""
        assert _qual_to_confidence(5.0) == pytest.approx(0.5, abs=0.01)

    def test_low_qual_returns_0_3(self):
        """QUAL < 5 returns 0.3 floor."""
        assert _qual_to_confidence(4.9) == 0.3
        assert _qual_to_confidence(1.0) == 0.3
        assert _qual_to_confidence(0.0) == 0.3

    def test_pair_5004_qual_gets_moderate_score(self):
        """QUAL=11.23 (pair_5004 case) gets ~0.71 — not filtered, but penalized."""
        score = _qual_to_confidence(11.23)
        assert 0.6 < score < 0.8


class TestContinuousQualScoring:
    """Tests for continuous QUAL scoring in validate_mutations_against_vcf."""

    def test_high_qual_variant_gives_full_confidence(self, repeat_dict):
        """QUAL=25 variant gives confidence weight 1.0."""
        x_seq = repeat_dict.repeats["X"]
        mutated = x_seq[:40] + "TT" + x_seq[40:]
        result = classify_sequence(x_seq + mutated + x_seq, repeat_dict)
        vcf_variants = [{"pos": 560, "qual": 25.0}]
        validated = validate_mutations_against_vcf(
            result, vcf_variants=vcf_variants, flank_length=500, unit_length=60
        )
        mut_repeat = validated["repeats"][1]
        # base_confidence * 1.0 (QUAL>=20)
        assert mut_repeat["confidence"] > 0.9

    def test_moderate_qual_variant_penalizes_confidence(self, repeat_dict):
        """QUAL=11 variant gets intermediate confidence (not 1.0, not 0.3)."""
        x_seq = repeat_dict.repeats["X"]
        mutated = x_seq[:40] + "TT" + x_seq[40:]
        result = classify_sequence(x_seq + mutated + x_seq, repeat_dict)
        vcf_variants = [{"pos": 560, "qual": 11.0}]
        validated = validate_mutations_against_vcf(
            result, vcf_variants=vcf_variants, flank_length=500, unit_length=60
        )
        mut_repeat = validated["repeats"][1]
        # Should be between 0.3 and 1.0 (moderate penalty)
        assert 0.3 < mut_repeat["confidence"] < 0.9

    def test_no_vcf_support_gives_low_confidence(self, repeat_dict):
        """No VCF variant at mutation position gives 0.3 weight."""
        x_seq = repeat_dict.repeats["X"]
        mutated = x_seq[:40] + "TT" + x_seq[40:]
        result = classify_sequence(x_seq + mutated + x_seq, repeat_dict)
        validated = validate_mutations_against_vcf(
            result, vcf_variants=[], flank_length=500, unit_length=60
        )
        mut_repeat = validated["repeats"][1]
        base = result["repeats"][1].get("confidence", 1.0)
        assert mut_repeat["confidence"] == pytest.approx(base * 0.3, abs=0.01)


class TestBoundaryPenalty:
    """Tests for boundary repeat penalty in validate_mutations_against_vcf."""

    def test_boundary_mutation_penalized(self, repeat_dict):
        """Mutation in last 3 repeats gets boundary penalty."""
        x_seq = repeat_dict.repeats["X"]
        # 10 repeats: mutation at repeat 9 → within boundary_repeats=3 of end
        mutated = x_seq[:40] + "TT" + x_seq[40:]
        result = classify_sequence(x_seq * 8 + mutated + x_seq, repeat_dict)

        # VCF at repeat 9 position: flank(500) + 8*60 = 980
        vcf_variants = [{"pos": 980, "qual": 25.0}]
        validated = validate_mutations_against_vcf(
            result,
            vcf_variants=vcf_variants,
            flank_length=500,
            unit_length=60,
            boundary_repeats=3,
            boundary_penalty=0.5,
        )
        mut = validated["mutations_detected"][0]
        assert mut["boundary"] is True
        # Confidence = base * qual_score(1.0) * boundary_penalty(0.5)
        mut_repeat = validated["repeats"][8]
        assert mut_repeat["confidence"] < 0.6

    def test_non_boundary_mutation_not_penalized(self, repeat_dict):
        """Mutation at repeat 2 of 10 repeats is not penalized."""
        x_seq = repeat_dict.repeats["X"]
        mutated = x_seq[:40] + "TT" + x_seq[40:]
        # 10 repeats: mutation at repeat 2 (well within interior)
        result = classify_sequence(x_seq + mutated + x_seq * 8, repeat_dict)

        vcf_variants = [{"pos": 560, "qual": 25.0}]
        validated = validate_mutations_against_vcf(
            result,
            vcf_variants=vcf_variants,
            flank_length=500,
            unit_length=60,
            boundary_repeats=3,
            boundary_penalty=0.5,
        )
        mut = validated["mutations_detected"][0]
        assert mut["boundary"] is False

    def test_boundary_flag_in_mutation_dict(self, repeat_dict):
        """Mutations have 'boundary' key indicating position status."""
        x_seq = repeat_dict.repeats["X"]
        mutated = x_seq[:40] + "TT" + x_seq[40:]
        result = classify_sequence(x_seq + mutated + x_seq, repeat_dict)
        vcf_variants = [{"pos": 560, "qual": 25.0}]
        validated = validate_mutations_against_vcf(
            result,
            vcf_variants=vcf_variants,
            flank_length=500,
            unit_length=60,
        )
        for mut in validated["mutations_detected"]:
            assert "boundary" in mut

    def test_pair_5003_scenario_penalized(self, repeat_dict):
        """Simulates pair_5003: FP at repeat 116 of 118 gets boundary penalty."""
        x_seq = repeat_dict.repeats["X"]
        # Build 118-repeat sequence with a "mutation" at repeat 116
        repeats_before = x_seq * 115
        mutated = x_seq[:40] + "TT" + x_seq[40:]  # repeat 116
        repeats_after = x_seq * 2  # repeats 117-118
        seq = repeats_before + mutated + repeats_after
        result = classify_sequence(seq, repeat_dict)

        # VCF variant at repeat 116: flank(500) + 115*60
        vcf_variants = [{"pos": 7400, "qual": 20.0}]
        validated = validate_mutations_against_vcf(
            result,
            vcf_variants=vcf_variants,
            flank_length=500,
            unit_length=60,
            boundary_repeats=3,
            boundary_penalty=0.5,
        )
        # Repeat 116 of 118 → within last 3 → boundary=True
        if validated["mutations_detected"]:
            mut = validated["mutations_detected"][0]
            assert mut["boundary"] is True


def test_classify_sequence_raises_on_no_match(mocker):
    """classify_sequence raises RuntimeError when no repeat match is found.

    We mock classify_repeat to return None so that best_result stays None
    after both the exact-match and fuzzy-fallback phases, triggering the
    guard at the bottom of the inner loop.
    """
    import open_pacmuci.classify as classify_mod
    from open_pacmuci.classify import classify_sequence
    from open_pacmuci.config import RepeatDictionary

    # Create a minimal but structurally valid repeat dict.
    rd = RepeatDictionary(
        repeats={},
        repeat_length_bp=60,
        pre_repeat_ids=["1", "2", "3", "4", "5"],
        after_repeat_ids=["6", "7", "8", "9"],
        canonical_repeat="X",
        flanking_left="",
        flanking_right="",
        vntr_region="",
        source="test",
        mutations={},
        mutated_sequences={},
        seq_to_id={},
    )

    # Patch classify_repeat inside the classify module to return None,
    # which leaves best_result as None and triggers the guard.
    mocker.patch.object(classify_mod, "classify_repeat", return_value=None)

    # A full-length sequence so remaining >= unit_length and the fallback
    # path runs and calls the (mocked) classify_repeat.
    sequence = "A" * 60

    with pytest.raises(RuntimeError, match="Classification failed"):
        classify_sequence(sequence, rd)


class TestForwardClassify:
    """Tests for the _forward_classify helper."""

    def test_single_exact_repeat(self, repeat_dict):
        """Single known repeat classifies with exact match, 0 mutations, offset=0."""
        x_seq = repeat_dict.repeats["X"]
        unit_length = repeat_dict.repeat_length_bp
        repeats, mutations, _labels, _pos, cumulative_offset = _forward_classify(
            x_seq, repeat_dict, unit_length, max_indel_probe=30
        )
        assert len(repeats) == 1
        assert repeats[0]["match"] == "exact"
        assert repeats[0]["type"] == "X"
        assert len(mutations) == 0
        assert cumulative_offset == 0

    def test_two_exact_repeats(self, repeat_dict):
        """Two concatenated known repeats both classify correctly."""
        x_seq = repeat_dict.repeats["X"]
        a_seq = repeat_dict.repeats.get("A", x_seq)  # fall back to X if A absent
        sequence = x_seq + a_seq
        unit_length = repeat_dict.repeat_length_bp
        repeats, mutations, _labels, _pos, cumulative_offset = _forward_classify(
            sequence, repeat_dict, unit_length, max_indel_probe=30
        )
        assert len(repeats) == 2
        assert all(r["match"] == "exact" for r in repeats)
        assert len(mutations) == 0
        assert cumulative_offset == 0


class TestApplyBidirectionalFallback:
    """Tests for the _apply_bidirectional_fallback helper."""

    def test_no_fallback_when_fully_consumed(self, repeat_dict):
        """When forward_pos >= len(sequence) - unit_length//2, no extra repeats added."""
        x_seq = repeat_dict.repeats["X"]
        unit_length = repeat_dict.repeat_length_bp
        sequence = x_seq * 3

        # Simulate a forward pass that consumed the entire sequence.
        repeats, mutations, labels, forward_pos, _ = _forward_classify(
            sequence, repeat_dict, unit_length, max_indel_probe=30
        )
        initial_count = len(repeats)

        # forward_pos should already be at or past the threshold
        result_repeats, _result_mutations, _result_labels = _apply_bidirectional_fallback(
            sequence, repeat_dict, repeats, mutations, labels, forward_pos
        )
        # No additional repeats should have been appended
        assert len(result_repeats) == initial_count


class TestComputeClassificationSummary:
    """Tests for the _compute_classification_summary helper."""

    def test_empty_repeats(self):
        """Empty inputs produce empty structure, confidence=0.0, exact_match_pct=0.0."""
        result = _compute_classification_summary(
            repeats=[], mutations=[], labels=[], cumulative_offset=0
        )
        assert result["structure"] == ""
        assert result["allele_confidence"] == 0.0
        assert result["exact_match_pct"] == 0.0
        assert result["repeats"] == []
        assert result["mutations_detected"] == []

    def test_all_exact_matches(self, repeat_dict):
        """Two exact-match repeats produce confidence=1.0, exact_match_pct=100.0."""
        x_seq = repeat_dict.repeats["X"]
        unit_length = repeat_dict.repeat_length_bp
        sequence = x_seq * 2
        repeats, mutations, labels, _, cumulative_offset = _forward_classify(
            sequence, repeat_dict, unit_length, max_indel_probe=30
        )
        result = _compute_classification_summary(repeats, mutations, labels, cumulative_offset)
        assert result["allele_confidence"] == 1.0
        assert result["exact_match_pct"] == 100.0
        # Structure should be two space-separated labels
        assert result["structure"] == "X X"


class TestReconcileSharedFrameshifts:
    """Tests for cross-allele frameshift demotion (Issue 5)."""

    def _allele(self, qual: float, frameshift: bool = True) -> dict:
        mut = {
            "repeat_index": 6,
            "closest_type": "C",
            "frameshift": frameshift,
            "differences": [{"pos": 1, "ref": "", "alt": "A", "type": "insertion"}],
            "vcf_qual": qual,
            "vcf_support": True,
        }
        return {
            "structure": "1 2 3 4 5 Cm X",
            "mutations_detected": [mut],
            "repeats": [],
            "allele_confidence": 1.0,
        }

    def test_demotes_weak_duplicate(self):
        """Weaker QUAL shared frameshift is demoted; stronger kept private."""
        results = {
            "allele_1": self._allele(8.59),
            "allele_2": self._allele(37.78),
        }
        out = reconcile_shared_frameshifts(results)
        assert len(out["allele_2"]["mutations_detected"]) == 1
        assert out["allele_2"]["mutations_detected"][0]["allele_private"] is True
        assert len(out["allele_1"]["mutations_detected"]) == 0
        assert len(out["allele_1"]["mutations_demoted"]) == 1
        assert out["allele_1"]["mutations_demoted"][0]["shared_duplicate"] is True

    def test_keeps_both_when_both_strong(self):
        """Both alleles keep the call when QUAL is strong on each."""
        results = {
            "allele_1": self._allele(30.0),
            "allele_2": self._allele(32.0),
        }
        out = reconcile_shared_frameshifts(results)
        assert len(out["allele_1"]["mutations_detected"]) == 1
        assert len(out["allele_2"]["mutations_detected"]) == 1
        assert out["allele_1"]["mutations_detected"][0]["shared"] is True
        assert out["allele_2"]["mutations_detected"][0]["shared"] is True

    def test_unique_frameshift_marked_private(self):
        """Frameshift on only one allele is allele_private."""
        results = {
            "allele_1": self._allele(20.0),
            "allele_2": {
                "structure": "1 2 3 4 5 C X",
                "mutations_detected": [],
                "repeats": [],
                "allele_confidence": 1.0,
            },
        }
        out = reconcile_shared_frameshifts(results)
        assert out["allele_1"]["mutations_detected"][0]["allele_private"] is True
        assert out["allele_1"]["mutations_detected"][0]["shared"] is False
