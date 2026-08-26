"""Tests for allele length detection from idxstats output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest

from open_pacmuci.alleles import (
    PRE_AFTER_REPEAT_COUNT,
    _build_allele_info,
    _find_peaks_by_prominence,
    _split_cluster_by_indel,
    detect_alleles,
    parse_idxstats,
    refine_peak_contig,
)

# Example idxstats output: contig_name\tcontig_length\tmapped_reads\tunmapped_reads
# contig_N means N canonical X repeats; total allele length = N + 9 (pre+after repeats)
IDXSTATS_TWO_PEAKS = """\
contig_58\t4000\t5\t0
contig_59\t4060\t12\t0
contig_60\t4120\t245\t0
contig_61\t4180\t18\t0
contig_62\t4240\t3\t0
contig_78\t5200\t8\t0
contig_79\t5260\t15\t0
contig_80\t5320\t189\t0
contig_81\t5380\t10\t0
contig_82\t5440\t2\t0
*\t0\t0\t50
"""

IDXSTATS_HOMOZYGOUS = """\
contig_59\t4060\t20\t0
contig_60\t4120\t310\t0
contig_61\t4180\t25\t0
*\t0\t0\t10
"""

IDXSTATS_LOW_COVERAGE = """\
contig_60\t4120\t5\t0
contig_80\t5320\t3\t0
*\t0\t0\t100
"""

# Plateau distribution (like real HiFi data -- reads spread across adjacent contigs)
IDXSTATS_PLATEAU = """\
contig_48\t3000\t30\t0
contig_49\t3060\t115\t0
contig_50\t3120\t115\t0
contig_51\t3180\t115\t0
contig_52\t3240\t115\t0
contig_53\t3300\t113\t0
contig_54\t3360\t85\t0
contig_68\t4200\t15\t0
contig_69\t4260\t53\t0
contig_70\t4320\t53\t0
contig_71\t4380\t53\t0
contig_72\t4440\t53\t0
contig_73\t4500\t49\t0
contig_74\t4560\t38\t0
*\t0\t0\t20
"""


class TestParseIdxstats:
    """Tests for parsing samtools idxstats output."""

    def test_parses_contig_counts(self):
        """Extracts repeat count -> read count mapping."""
        counts = parse_idxstats(IDXSTATS_TWO_PEAKS)
        assert counts[60] == 245
        assert counts[80] == 189

    def test_ignores_unmapped_line(self):
        """The '*' unmapped line is excluded."""
        counts = parse_idxstats(IDXSTATS_TWO_PEAKS)
        # All keys should be integers (no '*' string key)
        assert all(isinstance(k, int) for k in counts)

    def test_extracts_repeat_count_from_name(self):
        """Contig name 'contig_60' maps to repeat count 60."""
        counts = parse_idxstats(IDXSTATS_TWO_PEAKS)
        assert 60 in counts
        assert counts[60] == 245


class TestDetectAlleles:
    """Tests for allele detection via peak finding."""

    def test_two_alleles_detected(self):
        """Detects two distinct allele lengths from two-peak data."""
        counts = parse_idxstats(IDXSTATS_TWO_PEAKS)
        result = detect_alleles(counts, min_coverage=10)
        lengths = sorted([result["allele_1"]["length"], result["allele_2"]["length"]])
        # contig_60 + 9 = 69, contig_80 + 9 = 89
        assert lengths == [60 + PRE_AFTER_REPEAT_COUNT, 80 + PRE_AFTER_REPEAT_COUNT]

    def test_homozygous_detection(self):
        """Single dominant peak is reported as same_length (disambiguation pending)."""
        counts = parse_idxstats(IDXSTATS_HOMOZYGOUS)
        result = detect_alleles(counts, min_coverage=10)
        # Now same_length=True, homozygous=False (disambiguation happens later)
        assert result["same_length"] is True
        assert result["homozygous"] is False
        assert result["allele_1"]["length"] == 60 + PRE_AFTER_REPEAT_COUNT

    def test_low_coverage_raises(self):
        """Raises ValueError when no contig meets minimum coverage."""
        counts = parse_idxstats(IDXSTATS_LOW_COVERAGE)
        with pytest.raises(ValueError, match="coverage"):
            detect_alleles(counts, min_coverage=10)

    def test_allele_reads_reported(self):
        """Reports read counts for each allele."""
        counts = parse_idxstats(IDXSTATS_TWO_PEAKS)
        result = detect_alleles(counts, min_coverage=10)
        # allele_1 should be the one with more reads
        assert result["allele_1"]["reads"] >= result["allele_2"]["reads"]

    def test_output_is_json_serializable(self):
        """Result can be serialized to JSON."""
        counts = parse_idxstats(IDXSTATS_TWO_PEAKS)
        result = detect_alleles(counts, min_coverage=10)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_plateau_cluster_detection(self):
        """Detects peaks from plateau distributions (real HiFi-like data)."""
        counts = parse_idxstats(IDXSTATS_PLATEAU)
        result = detect_alleles(counts, min_coverage=10)
        lengths = sorted([result["allele_1"]["length"], result["allele_2"]["length"]])
        # Cluster 1 centers around contig_51 -> 51+9=60
        # Cluster 2 centers around contig_71 -> 71+9=80
        # Allow ±2 tolerance for weighted center rounding
        assert abs(lengths[0] - 60) <= 2, f"Expected ~60, got {lengths[0]}"
        assert abs(lengths[1] - 80) <= 2, f"Expected ~80, got {lengths[1]}"
        assert result["homozygous"] is False

    def test_contig_name_and_cluster_fields(self):
        """Allele result includes contig_name and cluster_contigs."""
        counts = parse_idxstats(IDXSTATS_TWO_PEAKS)
        result = detect_alleles(counts, min_coverage=10)
        a1 = result["allele_1"]
        assert "contig_name" in a1
        assert a1["contig_name"].startswith("contig_")
        assert "cluster_contigs" in a1
        assert isinstance(a1["cluster_contigs"], list)
        assert "canonical_repeats" in a1
        assert a1["length"] == a1["canonical_repeats"] + PRE_AFTER_REPEAT_COUNT


class TestProminencePeaks:
    """Tests for ONT-style prominence peak finding."""

    def test_ont_smear_does_not_invent_length_150_allele(self):
        """Wide ladder smear with one real peak → same_length, not ~150.

        Simulates ONT amplicon idxstats: every contig has residual coverage,
        one strong short peak, and a flat long-tail floor.  The old
        indel-valley path invented allele_2 near contig_150.
        """
        counts = {n: 80 for n in range(1, 151)}
        # Strong short-allele peak
        for n, r in [(32, 5000), (33, 9000), (34, 10000), (35, 9500), (36, 6000)]:
            counts[n] = r
        # Flat long-tail (must not become allele_2)
        for n in range(130, 151):
            counts[n] = 90

        result = detect_alleles(counts, min_coverage=10)
        assert result["same_length"] is True
        assert result["allele_1"]["canonical_repeats"] == 34
        assert result["allele_1"]["contig_name"] == "contig_34"
        assert result["allele_2"]["canonical_repeats"] == 34

    def test_two_well_separated_peaks_both_kept(self):
        """Two strong peaks with a deep valley → heterozygous call."""
        counts = {n: 20 for n in range(1, 121)}
        for n, r in [(39, 800), (40, 1200), (41, 900)]:
            counts[n] = r
        for n, r in [(79, 500), (80, 700), (81, 450)]:
            counts[n] = r
        # Deep valley between peaks
        for n in range(50, 70):
            counts[n] = 15

        result = detect_alleles(counts, min_coverage=10)
        assert result["same_length"] is False
        lengths = sorted(
            [
                result["allele_1"]["canonical_repeats"],
                result["allele_2"]["canonical_repeats"],
            ]
        )
        assert lengths == [40, 80]

    def test_find_peaks_prefers_longer_secondary(self):
        """Short-contig noise ridge is not chosen over a longer real peak."""
        counts = {n: 50 for n in range(1, 101)}
        # Primary
        counts[60] = 2000
        counts[59] = 1500
        counts[61] = 1500
        # Short noise ridge (must lose to longer peak)
        counts[10] = 900
        counts[9] = 800
        counts[11] = 800
        # Longer secondary with a clear valley
        counts[85] = 400
        counts[84] = 300
        counts[86] = 300
        for n in range(70, 80):
            counts[n] = 40

        clusters = _find_peaks_by_prominence(counts, min_coverage=10)
        assert len(clusters) == 2
        centers = sorted(c["center"] for c in clusters)
        assert centers == [60, 85]

    def test_short_noise_ridge_alone_yields_same_length(self):
        """Primary + short noise only → same_length (no longer secondary)."""
        counts = {n: 40 for n in range(1, 101)}
        counts[72] = 2000
        counts[71] = 1800
        counts[73] = 1800
        counts[6] = 1200
        counts[5] = 1100
        counts[7] = 1100
        result = detect_alleles(counts, min_coverage=10)
        assert result["same_length"] is True
        assert result["allele_1"]["canonical_repeats"] == 72


# ---------------------------------------------------------------------------
# SAM output helpers for refine_peak_contig tests
# ---------------------------------------------------------------------------


def _make_sam_line(
    read_name: str,
    contig: str,
    cigar: str = "3600M",
    as_score: int = 3109,
) -> str:
    """Build a minimal SAM alignment line with an AS tag."""
    seq = "A" * 10
    qual = "I" * 10
    return f"{read_name}\t0\t{contig}\t1\t60\t{cigar}\t*\t0\t0\t{seq}\t{qual}\tAS:i:{as_score}"


SAM_ONE_CONTIG = "\n".join(
    [_make_sam_line(f"read{i}", "contig_51", as_score=3100 + i) for i in range(5)]
)

SAM_TWO_CONTIGS = "\n".join(
    [
        _make_sam_line("r1", "contig_50", as_score=2800),
        _make_sam_line("r2", "contig_50", as_score=2850),
        _make_sam_line("r3", "contig_51", as_score=3100),
        _make_sam_line("r4", "contig_51", as_score=3200),
        _make_sam_line("r5", "contig_51", as_score=3150),
    ]
)

SAM_WITH_INDELS = "\n".join(
    [
        _make_sam_line("r1", "contig_50", cigar="100M50I100M", as_score=200),
        _make_sam_line("r2", "contig_51", cigar="3600M", as_score=3100),
    ]
)


class TestRefinePeakContig:
    """Tests for refine_peak_contig."""

    @patch("open_pacmuci.alleles.run_tool_iter")
    def test_selects_contig_with_highest_mean_as(self, mock_run_tool_iter, tmp_path):
        """Picks the contig whose reads have the highest mean alignment score."""
        mock_run_tool_iter.return_value = iter(SAM_TWO_CONTIGS.splitlines(keepends=True))
        bam = tmp_path / "mapping.bam"

        result = refine_peak_contig(bam, ["contig_50", "contig_51"])

        assert result["best_contig"] == "contig_51"

    @patch("open_pacmuci.alleles.run_tool_iter")
    def test_calls_samtools_view_with_cluster_contigs(self, mock_run_tool_iter, tmp_path):
        """samtools view is called with all cluster contig names."""
        mock_run_tool_iter.return_value = iter(SAM_ONE_CONTIG.splitlines(keepends=True))
        bam = tmp_path / "mapping.bam"
        contigs = ["contig_50", "contig_51", "contig_52"]

        refine_peak_contig(bam, contigs)

        cmd = mock_run_tool_iter.call_args[0][0]
        assert cmd[:2] == ["samtools", "view"]
        for c in contigs:
            assert c in cmd

    @patch("open_pacmuci.alleles.run_tool_iter")
    def test_metrics_reported_per_contig(self, mock_run_tool_iter, tmp_path):
        """Returns per-contig mean_as, mean_indel_bp, and reads counts."""
        mock_run_tool_iter.return_value = iter(SAM_TWO_CONTIGS.splitlines(keepends=True))
        bam = tmp_path / "mapping.bam"

        result = refine_peak_contig(bam, ["contig_50", "contig_51"])

        assert "contig_50" in result["metrics"]
        assert "contig_51" in result["metrics"]
        m = result["metrics"]["contig_51"]
        assert "mean_as" in m
        assert "mean_indel_bp" in m
        assert "reads" in m
        assert m["reads"] == 3

    @patch("open_pacmuci.alleles.run_tool_iter")
    def test_single_contig_is_selected_as_best(self, mock_run_tool_iter, tmp_path):
        """When only one contig has reads, it is always selected."""
        mock_run_tool_iter.return_value = iter(SAM_ONE_CONTIG.splitlines(keepends=True))
        bam = tmp_path / "mapping.bam"

        result = refine_peak_contig(bam, ["contig_51"])

        assert result["best_contig"] == "contig_51"

    @patch("open_pacmuci.alleles.run_tool_iter")
    def test_empty_sam_output_falls_back_to_first_contig(self, mock_run_tool_iter, tmp_path):
        """With no aligned reads, best_contig defaults to the first in the list."""
        mock_run_tool_iter.return_value = iter([])
        bam = tmp_path / "mapping.bam"
        contigs = ["contig_48", "contig_49", "contig_50"]

        result = refine_peak_contig(bam, contigs)

        assert result["best_contig"] == "contig_48"
        assert result["metrics"] == {}

    @patch("open_pacmuci.alleles.run_tool_iter")
    def test_indel_length_computed_from_cigar(self, mock_run_tool_iter, tmp_path):
        """mean_indel_bp is derived from I/D operations in the CIGAR string."""
        mock_run_tool_iter.return_value = iter(SAM_WITH_INDELS.splitlines(keepends=True))
        bam = tmp_path / "mapping.bam"

        result = refine_peak_contig(bam, ["contig_50", "contig_51"])

        # contig_50 read has 50I in CIGAR → mean_indel_bp == 50
        assert result["metrics"]["contig_50"]["mean_indel_bp"] == 50.0

    @patch("open_pacmuci.alleles.run_tool_iter")
    def test_header_lines_are_skipped(self, mock_run_tool_iter, tmp_path):
        """SAM header lines starting with @ are ignored."""
        sam_with_header = "@HD\tVN:1.6\n@SQ\tSN:contig_51\tLN:3660\n" + SAM_ONE_CONTIG
        mock_run_tool_iter.return_value = iter(sam_with_header.splitlines(keepends=True))
        bam = tmp_path / "mapping.bam"

        # Should not raise or count header lines as reads
        result = refine_peak_contig(bam, ["contig_51"])
        assert result["metrics"]["contig_51"]["reads"] == 5


class TestBuildAlleleInfo:
    """Tests for _build_allele_info."""

    def _cluster(self, center: int, total_reads: int, contigs=None):
        if contigs is None:
            contigs = [(center, total_reads)]
        return {"center": center, "total_reads": total_reads, "contigs": contigs}

    def test_length_includes_pre_after_repeats(self):
        """Total length = canonical + PRE_AFTER_REPEAT_COUNT."""
        cluster = self._cluster(51, 200)
        info = _build_allele_info(cluster)
        assert info["length"] == 51 + PRE_AFTER_REPEAT_COUNT

    def test_canonical_repeats_matches_center(self):
        """canonical_repeats field equals the cluster center."""
        cluster = self._cluster(71, 150)
        info = _build_allele_info(cluster)
        assert info["canonical_repeats"] == 71

    def test_reads_from_cluster_total(self):
        """reads field comes from cluster total_reads."""
        cluster = self._cluster(60, 300)
        info = _build_allele_info(cluster)
        assert info["reads"] == 300

    def test_best_contig_used_when_provided(self):
        """contig_name is set to best_contig when explicitly given."""
        cluster = self._cluster(51, 200)
        info = _build_allele_info(cluster, best_contig="contig_51")
        assert info["contig_name"] == "contig_51"

    def test_fallback_contig_name_when_no_best_contig(self):
        """contig_name defaults to contig_<center> when best_contig is None."""
        cluster = self._cluster(51, 200)
        info = _build_allele_info(cluster, best_contig=None)
        assert info["contig_name"] == "contig_51"

    def test_cluster_contigs_list_built_correctly(self):
        """cluster_contigs contains contig_N names for each contig in the cluster."""
        cluster = self._cluster(51, 400, contigs=[(50, 100), (51, 200), (52, 100)])
        info = _build_allele_info(cluster)
        assert info["cluster_contigs"] == ["contig_50", "contig_51", "contig_52"]

    def test_canonical_from_best_contig_overrides_center(self):
        """When best_contig differs from cluster center, canonical_repeats follows best_contig.

        ONT reads produce a wider distribution that can shift the weighted
        center by 1 vs the alignment-score-refined best contig.  The
        canonical_repeats (and therefore allele length) must come from the
        refined contig, not the noisy cluster center.
        """
        # Simulate an ONT-like cluster: center rounds to 52, but
        # refine_peak_contig correctly identified contig_51.
        cluster = self._cluster(
            52,  # weighted center rounds to 52 due to rightward skew
            1880,
            contigs=[
                (48, 35),
                (49, 275),
                (50, 310),
                (51, 313),
                (52, 317),
                (53, 317),
                (54, 279),
                (55, 34),
            ],
        )
        info = _build_allele_info(cluster, best_contig="contig_51")
        assert info["contig_name"] == "contig_51"
        assert info["canonical_repeats"] == 51  # from best_contig, not center
        assert info["length"] == 51 + PRE_AFTER_REPEAT_COUNT  # 60, not 61

    def test_best_contig_far_from_center_falls_back(self):
        """When AS-refined contig differs from center by >1, fall back to center.

        For long ONT alleles (>80 repeats), the AS metric can peak at a
        contig 2+ positions away from the true length.  In that case the
        cluster center (off by at most 1) is more reliable.  contig_name
        must follow the chosen length so remapping does not use the
        discarded AS peak.
        """
        # ONT 80/100 case: center=72 (true=71), AS picks contig_69 (off by -2)
        cluster = self._cluster(
            72,
            1794,
            contigs=[
                (68, 19),
                (69, 235),
                (70, 283),
                (71, 303),
                (72, 309),
                (73, 309),
                (74, 283),
                (75, 53),
            ],
        )
        info = _build_allele_info(cluster, best_contig="contig_69")
        assert info["canonical_repeats"] == 72
        assert info["length"] == 72 + PRE_AFTER_REPEAT_COUNT
        assert info["contig_name"] == "contig_72"

    def test_ont_as_peak_far_from_center_keeps_contig_aligned(self):
        """ONT smear: AS picks contig_73 while center is ~34; remap contig_34.

        Reproduces barcode18/20 allele_1 failure mode where reported length
        and Clair3 reference disagreed.
        """
        cluster = self._cluster(
            34,
            100000,
            contigs=[(n, 100) for n in range(1, 123)],
        )
        info = _build_allele_info(cluster, best_contig="contig_73")
        assert info["canonical_repeats"] == 34
        assert info["length"] == 34 + PRE_AFTER_REPEAT_COUNT
        assert info["contig_name"] == "contig_34"
        assert info["contig_name"] == f"contig_{info['canonical_repeats']}"


class TestSameLengthDetection:
    """Tests for same_length allele detection."""

    def test_single_cluster_sets_same_length(self):
        """Single cluster sets same_length=True."""
        counts = parse_idxstats(IDXSTATS_HOMOZYGOUS)
        result = detect_alleles(counts, min_coverage=10)
        assert result["same_length"] is True

    def test_same_length_keeps_allele2_data(self):
        """same_length allele_2 has contig_name (not None)."""
        counts = parse_idxstats(IDXSTATS_HOMOZYGOUS)
        result = detect_alleles(counts, min_coverage=10)
        assert result["allele_2"]["contig_name"] is not None

    def test_different_lengths_no_same_length(self):
        """Different-length alleles have same_length=False."""
        counts = parse_idxstats(IDXSTATS_TWO_PEAKS)
        result = detect_alleles(counts, min_coverage=10)
        assert result.get("same_length", False) is False


class TestBuildAlleleInfoExtra:
    """Extra tests for _build_allele_info (continuation)."""

    def test_detect_alleles_with_bam_calls_refine(self):
        """detect_alleles calls refine_peak_contig when bam_path is provided."""
        counts = parse_idxstats(IDXSTATS_TWO_PEAKS)
        bam = Path("/fake/mapping.bam")
        with patch("open_pacmuci.alleles.refine_peak_contig") as mock_refine:
            mock_refine.return_value = {
                "best_contig": "contig_60",
                "metrics": {"contig_60": {"mean_as": 3100.0, "mean_indel_bp": 0.0, "reads": 10}},
            }
            result = detect_alleles(counts, min_coverage=10, bam_path=bam)

        # refine_peak_contig should have been called (once per cluster = 2 times)
        assert mock_refine.call_count == 2
        # The contig_name should come from the mock
        assert result["allele_1"]["contig_name"] == "contig_60"


# ---------------------------------------------------------------------------
# Helper: build SAM lines for _split_cluster_by_indel tests
# ---------------------------------------------------------------------------


def _make_indel_sam_lines(contig: str, cigar: str, n: int = 5) -> list[str]:
    """Return *n* minimal SAM lines all mapping to *contig* with *cigar*."""
    seq = "A" * 10
    qual = "I" * 10
    return [
        f"read_{contig}_{i}\t0\t{contig}\t1\t60\t{cigar}\t*\t0\t0\t{seq}\t{qual}" for i in range(n)
    ]


class TestSplitClusterByIndel:
    """Tests for _split_cluster_by_indel valley splitting."""

    # ------------------------------------------------------------------
    # Cluster fixture spanning contigs 40..50
    # contigs list: [(40,50),(42,50),(45,50),(48,50),(50,50)]
    # ------------------------------------------------------------------
    _CLUSTER: ClassVar[dict] = {
        "center": 45,
        "total_reads": 250,
        "contigs": [(40, 50), (42, 50), (45, 50), (48, 50), (50, 50)],
    }

    @patch("open_pacmuci.alleles.run_tool_iter")
    def test_returns_none_when_no_reads(self, mock_run_tool_iter, tmp_path):
        """Returns None when samtools view produces no alignment lines."""
        mock_run_tool_iter.return_value = iter([])
        bam = tmp_path / "mapping.bam"

        result = _split_cluster_by_indel(bam, self._CLUSTER)

        assert result is None

    @patch("open_pacmuci.alleles.run_tool_iter")
    def test_returns_none_when_single_group(self, mock_run_tool_iter, tmp_path):
        """Returns None when all reads have similar indel lengths (single valley).

        All reads use 3600M CIGAR → 0 bp indels on every contig.  The
        indel series is flat, so only one valley exists and the function
        must return None.
        """
        # Put reads on contigs 40, 42, 45, 48, 50 — all with 0-indel CIGAR
        sam_lines: list[str] = []
        for c, _ in self._CLUSTER["contigs"]:
            sam_lines.extend(_make_indel_sam_lines(f"contig_{c}", "3600M"))
        mock_run_tool_iter.return_value = iter(line + "\n" for line in sam_lines)

        bam = tmp_path / "mapping.bam"
        result = _split_cluster_by_indel(bam, self._CLUSTER)

        assert result is None

    @patch("open_pacmuci.alleles.run_tool_iter")
    def test_splits_with_distinct_indel_groups(self, mock_run_tool_iter, tmp_path):
        """Returns 2 sub-clusters when reads fall into two distinct indel groups.

        Layout (5 contigs, all with reads):
          contig_40 → 0 bp indels   (valley A)
          contig_42 → 180 bp indels (hill)
          contig_45 → 360 bp indels (peak of hill)
          contig_48 → 180 bp indels (hill)
          contig_50 → 0 bp indels   (valley B)

        Valleys are at positions 40 and 50 (10 apart ≥ 3 threshold).
        The split point midpoint = (40+50)//2 = 45, so sub1 = contigs ≤ 45
        and sub2 = contigs > 45.
        """
        sam_lines: list[str] = []
        # Valley A: contig_40 — no indels
        sam_lines.extend(_make_indel_sam_lines("contig_40", "3600M"))
        # Hill: contig_42 and contig_45 — 180bp and 360bp insertions
        sam_lines.extend(_make_indel_sam_lines("contig_42", "3420M180I"))
        sam_lines.extend(_make_indel_sam_lines("contig_45", "3240M360I"))
        # Hill descending: contig_48 — 180bp insertion
        sam_lines.extend(_make_indel_sam_lines("contig_48", "3420M180I"))
        # Valley B: contig_50 — no indels
        sam_lines.extend(_make_indel_sam_lines("contig_50", "3600M"))
        mock_run_tool_iter.return_value = iter(line + "\n" for line in sam_lines)

        bam = tmp_path / "mapping.bam"
        result = _split_cluster_by_indel(bam, self._CLUSTER)

        assert result is not None, "Expected two sub-clusters but got None"
        assert len(result) == 2, f"Expected 2 sub-clusters, got {len(result)}"

        centers = sorted(c["center"] for c in result)
        # sub1 contains contigs 40, 42, 45  → weighted center = 40+42+45)/3 = ~42
        # sub2 contains contigs 48, 50       → weighted center = (48+50)/2 = 49
        assert centers[0] < centers[1], "Sub-cluster centers should be distinct"
