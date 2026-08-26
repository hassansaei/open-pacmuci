"""Mocked unit tests for consensus module (bcftools wrappers)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from open_pacmuci.consensus import (
    build_consensus,
    build_consensus_per_allele,
    check_consensus_length,
    trim_flanking,
)


class TestBuildConsensus:
    """Tests for build_consensus."""

    @patch("open_pacmuci.consensus.run_tool")
    def test_calls_bcftools_consensus(self, mock_run_tool, tmp_path):
        """build_consensus calls bcftools consensus with correct arguments."""
        fasta_content = ">contig_51\nACGTACGT\n"
        mock_run_tool.return_value = fasta_content
        ref = tmp_path / "ref.fa"
        ref.touch()
        vcf = tmp_path / "variants.vcf.gz"
        vcf.touch()
        output = tmp_path / "consensus.fa"

        build_consensus(ref, vcf, output)

        cmd = mock_run_tool.call_args[0][0]
        assert cmd[:2] == ["bcftools", "consensus"]
        assert str(ref) in cmd
        assert str(vcf) in cmd

    @patch("open_pacmuci.consensus.run_tool")
    def test_writes_stdout_to_output_path(self, mock_run_tool, tmp_path):
        """build_consensus writes the bcftools stdout to the output file."""
        fasta_content = ">contig_51\nACGTACGT\n"
        mock_run_tool.return_value = fasta_content
        ref = tmp_path / "ref.fa"
        ref.touch()
        vcf = tmp_path / "variants.vcf.gz"
        vcf.touch()
        output = tmp_path / "consensus.fa"

        result = build_consensus(ref, vcf, output)

        assert output.read_text() == fasta_content
        assert result == output

    @patch("open_pacmuci.consensus.run_tool")
    def test_creates_parent_directory(self, mock_run_tool, tmp_path):
        """build_consensus creates the parent directory of output_path if needed."""
        mock_run_tool.return_value = ">x\nACGT\n"
        ref = tmp_path / "ref.fa"
        ref.touch()
        vcf = tmp_path / "variants.vcf.gz"
        vcf.touch()
        output = tmp_path / "nested" / "dir" / "consensus.fa"

        assert not output.parent.exists()
        build_consensus(ref, vcf, output)
        assert output.parent.exists()

    @patch("open_pacmuci.consensus.run_tool")
    def test_reference_passed_with_f_flag(self, mock_run_tool, tmp_path):
        """bcftools consensus uses -f flag for the reference."""
        mock_run_tool.return_value = ">x\nACGT\n"
        ref = tmp_path / "ref.fa"
        ref.touch()
        vcf = tmp_path / "variants.vcf.gz"
        vcf.touch()
        output = tmp_path / "out.fa"

        build_consensus(ref, vcf, output)

        cmd = mock_run_tool.call_args[0][0]
        f_idx = cmd.index("-f")
        assert cmd[f_idx + 1] == str(ref)


class TestTrimFlanking:
    """Tests for trim_flanking (pure Python, no mocking needed)."""

    def test_removes_flanks_from_both_ends(self, tmp_path):
        """Correct flanking bp are trimmed from both ends of the sequence."""
        sequence = "A" * 100 + "VNTR_SEQ" + "A" * 100
        fasta = tmp_path / "full.fa"
        fasta.write_text(f">contig_51\n{sequence}\n")
        output = tmp_path / "trimmed.fa"

        trim_flanking(fasta, 100, output)

        lines = output.read_text().strip().splitlines()
        trimmed_seq = "".join(ln for ln in lines if not ln.startswith(">"))
        assert trimmed_seq == "VNTR_SEQ"

    def test_output_header_has_vntr_suffix(self, tmp_path):
        """Output FASTA header gets a _vntr suffix."""
        fasta = tmp_path / "full.fa"
        fasta.write_text(">contig_51\n" + "A" * 10 + "CORE" + "A" * 10 + "\n")
        output = tmp_path / "trimmed.fa"

        trim_flanking(fasta, 10, output)

        header = output.read_text().strip().splitlines()[0]
        assert header.endswith("_vntr")

    def test_zero_flank_length_keeps_full_sequence(self, tmp_path):
        """With flank_length=0, the entire sequence is kept."""
        sequence = "ACGTACGT"
        fasta = tmp_path / "full.fa"
        fasta.write_text(f">contig_51\n{sequence}\n")
        output = tmp_path / "trimmed.fa"

        trim_flanking(fasta, 0, output)

        lines = output.read_text().strip().splitlines()
        trimmed_seq = "".join(ln for ln in lines if not ln.startswith(">"))
        assert trimmed_seq == sequence

    def test_returns_output_path(self, tmp_path):
        """trim_flanking returns the output path."""
        fasta = tmp_path / "full.fa"
        fasta.write_text(">c\n" + "A" * 20 + "\n")
        output = tmp_path / "out.fa"

        result = trim_flanking(fasta, 5, output)

        assert result == output

    def test_handles_fasta_without_header(self, tmp_path):
        """Falls back to >consensus when no header line is present."""
        fasta = tmp_path / "full.fa"
        fasta.write_text("ACGT")
        output = tmp_path / "out.fa"

        trim_flanking(fasta, 0, output)

        header = output.read_text().strip().splitlines()[0]
        assert header == ">consensus_vntr"


class TestAnchorBasedTrim:
    """Tests for anchor-based flanking trim."""

    def test_anchor_trim_handles_indel_in_flank(self, tmp_path):
        """Anchor trim finds correct boundary despite indel in flanking."""
        from open_pacmuci.config import load_repeat_dictionary

        rd = load_repeat_dictionary()
        # Build a sequence: left_flank (with 1bp insertion) + pre-repeat 1 + X + after-repeat 9 + right_flank
        left_flank = rd.flanking_left[-500:]
        right_flank = rd.flanking_right[:500]
        vntr = rd.repeats["1"] + rd.repeats["X"] + rd.repeats["9"]
        # Insert a fake base in the left flank (simulating Clair3 false positive)
        corrupted_left = left_flank[:250] + "A" + left_flank[250:]
        sequence = corrupted_left + vntr + right_flank  # 501 + vntr + 500

        fasta = tmp_path / "full.fa"
        fasta.write_text(f">contig\n{sequence}\n")
        output = tmp_path / "trimmed.fa"

        trim_flanking(fasta, 500, output, repeat_dict=rd)

        trimmed = output.read_text().strip().splitlines()
        trimmed_seq = "".join(ln for ln in trimmed if not ln.startswith(">"))
        # Should contain the start of repeat "1"
        assert rd.repeats["1"][:30] in trimmed_seq

    def test_anchor_trim_fallback_without_repeat_dict(self, tmp_path):
        """Without repeat_dict, falls back to fixed-position trim."""
        sequence = "A" * 100 + "VNTR_SEQ" + "A" * 100
        fasta = tmp_path / "full.fa"
        fasta.write_text(f">contig\n{sequence}\n")
        output = tmp_path / "trimmed.fa"

        trim_flanking(fasta, 100, output, repeat_dict=None)

        trimmed = output.read_text().strip().splitlines()
        trimmed_seq = "".join(ln for ln in trimmed if not ln.startswith(">"))
        assert trimmed_seq == "VNTR_SEQ"


class TestBuildConsensusPerAllele:
    """Tests for build_consensus_per_allele."""

    def _alleles(self):
        return {
            "homozygous": False,
            "allele_1": {
                "length": 60,
                "reads": 200,
                "canonical_repeats": 51,
                "contig_name": "contig_51",
                "cluster_contigs": ["contig_51"],
            },
            "allele_2": {
                "length": 80,
                "reads": 150,
                "canonical_repeats": 71,
                "contig_name": "contig_71",
                "cluster_contigs": ["contig_71"],
            },
        }

    @patch("open_pacmuci.consensus.run_tool")
    def test_processes_all_allele_keys_in_vcf_paths(self, mock_run_tool, tmp_path):
        """build_consensus_per_allele handles every key in vcf_paths."""
        flanks = 50
        # Fake consensus output: 2*flanks + some core
        core = "GCGCGCGC"
        fake_fasta = f">contig_51\n{'A' * flanks}{core}{'A' * flanks}\n"
        mock_run_tool.return_value = fake_fasta

        vcf1 = tmp_path / "variants1.vcf.gz"
        vcf1.touch()
        vcf2 = tmp_path / "variants2.vcf.gz"
        vcf2.touch()
        vcf_paths = {"allele_1": vcf1, "allele_2": vcf2}
        alleles = self._alleles()
        out_dir = tmp_path / "consensus"

        result = build_consensus_per_allele(
            tmp_path / "ref.fa", vcf_paths, alleles, out_dir, flank_length=flanks
        )

        assert "allele_1" in result
        assert "allele_2" in result

    @patch("open_pacmuci.consensus.run_tool")
    def test_calls_samtools_faidx_for_each_allele(self, mock_run_tool, tmp_path):
        """samtools faidx is called to extract each allele's contig."""
        flanks = 10
        mock_run_tool.return_value = f">contig_51\n{'A' * (flanks * 2 + 8)}\n"

        vcf1 = tmp_path / "v.vcf.gz"
        vcf1.touch()
        vcf_paths = {"allele_1": vcf1}
        alleles = {
            "allele_1": {
                "length": 60,
                "reads": 200,
                "canonical_repeats": 51,
                "contig_name": "contig_51",
                "cluster_contigs": ["contig_51"],
            }
        }

        build_consensus_per_allele(
            tmp_path / "ref.fa", vcf_paths, alleles, tmp_path / "out", flank_length=flanks
        )

        faidx_calls = [
            c[0][0] for c in mock_run_tool.call_args_list if c[0][0][:2] == ["samtools", "faidx"]
        ]
        # At least one faidx extracting the contig
        assert any("contig_51" in c for c in faidx_calls)

    @patch("open_pacmuci.consensus.run_tool")
    def test_returns_paths_to_trimmed_fastas(self, mock_run_tool, tmp_path):
        """Returns paths to the trimmed (VNTR-only) FASTA files."""
        flanks = 10
        mock_run_tool.return_value = f">c\n{'A' * (flanks * 2 + 4)}\n"

        vcf1 = tmp_path / "v.vcf.gz"
        vcf1.touch()
        alleles = {
            "allele_1": {
                "length": 60,
                "reads": 200,
                "canonical_repeats": 51,
                "contig_name": "contig_51",
                "cluster_contigs": ["contig_51"],
            }
        }

        result = build_consensus_per_allele(
            tmp_path / "ref.fa",
            {"allele_1": vcf1},
            alleles,
            tmp_path / "out",
            flank_length=flanks,
        )

        assert isinstance(result["allele_1"], Path)
        # Should be the trimmed file, not the full consensus
        assert "full" not in result["allele_1"].name

    @patch("open_pacmuci.consensus.run_tool")
    def test_fallback_contig_name_from_length(self, mock_run_tool, tmp_path):
        """Falls back to contig_<length> when contig_name key is absent."""
        flanks = 5
        mock_run_tool.return_value = f">c\n{'A' * (flanks * 2 + 4)}\n"

        vcf1 = tmp_path / "v.vcf.gz"
        vcf1.touch()
        alleles = {
            "allele_1": {
                "length": 60,
                "reads": 200,
                "canonical_repeats": 51,
                # no "contig_name" key
                "cluster_contigs": ["contig_51"],
            }
        }

        # Should not raise
        result = build_consensus_per_allele(
            tmp_path / "ref.fa",
            {"allele_1": vcf1},
            alleles,
            tmp_path / "out",
            flank_length=flanks,
        )
        assert "allele_1" in result


class TestCheckConsensusLength:
    """Tests for check_consensus_length QC."""

    def test_passes_when_within_tolerance(self, tmp_path):
        """Exact and ±1 unit counts pass."""
        fasta = tmp_path / "c.fa"
        # 60 units * 60 bp = 3600 bp
        fasta.write_text(">allele\n" + ("A" * 3600) + "\n")
        result = check_consensus_length(fasta, expected_units=60)
        assert result["passed"] is True
        assert result["observed_units"] == 60
        assert result["delta"] == 0

        # ±1 tolerance for frameshift
        fasta.write_text(">allele\n" + ("A" * 3660) + "\n")
        result = check_consensus_length(fasta, expected_units=60)
        assert result["passed"] is True
        assert result["delta"] == 1

    def test_fails_when_outside_tolerance(self, tmp_path):
        """Large length mismatch fails QC."""
        fasta = tmp_path / "c.fa"
        # 81 units when 43 expected (barcode20-style mismatch)
        fasta.write_text(">allele\n" + ("A" * (81 * 60)) + "\n")
        result = check_consensus_length(fasta, expected_units=43)
        assert result["passed"] is False
        assert result["observed_units"] == 81
        assert result["expected_units"] == 43
        assert result["delta"] == 38

    def test_empty_sequence(self, tmp_path):
        """Empty consensus reports zero observed units."""
        fasta = tmp_path / "c.fa"
        fasta.write_text(">allele\n\n")
        result = check_consensus_length(fasta, expected_units=50)
        assert result["observed_units"] == 0
        assert result["passed"] is False
