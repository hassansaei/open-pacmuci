"""Unit tests for vcf module (filter_vcf, parse_vcf_genotypes, parse_vcf_variants)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from open_pacmuci.vcf import filter_vcf, parse_vcf_genotypes, parse_vcf_variants


class TestFilterVcf:
    """Tests for filter_vcf."""

    @patch("open_pacmuci.vcf.run_tool")
    def test_calls_bcftools_norm_then_view_then_index(self, mock_run_tool, tmp_path):
        """filter_vcf runs bcftools norm, bcftools view, then bcftools index."""
        mock_run_tool.return_value = ""
        vcf = tmp_path / "raw.vcf.gz"
        ref = tmp_path / "ref.fa"
        out_dir = tmp_path / "filtered"

        filter_vcf(vcf, ref, out_dir)

        calls = mock_run_tool.call_args_list
        assert calls[0][0][0][:2] == ["bcftools", "norm"]
        assert calls[1][0][0][:2] == ["bcftools", "view"]
        assert calls[2][0][0][:2] == ["bcftools", "index"]

    @patch("open_pacmuci.vcf.run_tool")
    def test_norm_uses_reference(self, mock_run_tool, tmp_path):
        """bcftools norm -f flag receives the reference path."""
        mock_run_tool.return_value = ""
        vcf = tmp_path / "raw.vcf.gz"
        ref = tmp_path / "ref.fa"
        out_dir = tmp_path / "filtered"

        filter_vcf(vcf, ref, out_dir)

        norm_cmd = mock_run_tool.call_args_list[0][0][0]
        assert "-f" in norm_cmd
        ref_idx = norm_cmd.index("-f")
        assert norm_cmd[ref_idx + 1] == str(ref)

    @patch("open_pacmuci.vcf.run_tool")
    def test_view_filters_pass(self, mock_run_tool, tmp_path):
        """bcftools view uses -f PASS to keep only passing variants (HiFi)."""
        mock_run_tool.return_value = ""
        vcf = tmp_path / "raw.vcf.gz"
        ref = tmp_path / "ref.fa"
        out_dir = tmp_path / "filtered"
        # Non-empty normalized VCF so QUAL filter is applied
        norm = out_dir / "normalized.vcf.gz"
        out_dir.mkdir(parents=True, exist_ok=True)

        def _side_effect(cmd):
            if cmd[:2] == ["bcftools", "norm"]:
                norm.write_bytes(b"x" * 100)
            return ""

        mock_run_tool.side_effect = _side_effect

        filter_vcf(vcf, ref, out_dir, min_qual=5.0, platform="hifi")

        view_cmd = mock_run_tool.call_args_list[1][0][0]
        assert "-f" in view_cmd
        f_idx = view_cmd.index("-f")
        assert view_cmd[f_idx + 1] == "PASS"

    @patch("open_pacmuci.vcf.run_tool")
    def test_ont_rescues_lowqual_one_bp_indels(self, mock_run_tool, tmp_path):
        """ONT filter keeps 1 bp indels at QUAL>=1 even if LowQual."""
        mock_run_tool.return_value = ""
        vcf = tmp_path / "raw.vcf.gz"
        ref = tmp_path / "ref.fa"
        out_dir = tmp_path / "filtered"
        out_dir.mkdir(parents=True, exist_ok=True)

        def _side_effect(cmd):
            if cmd[:2] == ["bcftools", "norm"]:
                (out_dir / "normalized.vcf.gz").write_bytes(b"x" * 100)
            return ""

        mock_run_tool.side_effect = _side_effect

        filter_vcf(vcf, ref, out_dir, min_qual=5.0, platform="ont")

        view_cmd = mock_run_tool.call_args_list[1][0][0]
        assert "-f" not in view_cmd  # expression replaces PASS-only
        assert "-i" in view_cmd
        expr = view_cmd[view_cmd.index("-i") + 1]
        assert "abs(strlen(REF)-strlen(ALT))=1" in expr
        assert "QUAL>=1" in expr or "QUAL>=1.0" in expr

    @patch("open_pacmuci.vcf.run_tool")
    def test_returns_variants_vcf_path(self, mock_run_tool, tmp_path):
        """filter_vcf returns path to variants.vcf.gz."""
        mock_run_tool.return_value = ""
        vcf = tmp_path / "raw.vcf.gz"
        ref = tmp_path / "ref.fa"
        out_dir = tmp_path / "filtered"

        result = filter_vcf(vcf, ref, out_dir)

        assert result == out_dir / "variants.vcf.gz"


class TestFilterVcfEmptyVcf:
    """Tests for filter_vcf handling empty VCFs (issue #8)."""

    @patch("open_pacmuci.vcf.run_tool")
    def test_empty_vcf_skips_quality_filter(self, mock_run_tool, tmp_path):
        """Empty VCF (no records) skips -i filter to avoid INFO/DP crash."""
        # bcftools norm returns empty output, then bcftools view should NOT
        # include -i filter since there are no records to filter
        norm_vcf = tmp_path / "normalized.vcf.gz"

        def side_effect(cmd):
            # After norm runs, create the normalized file with header only
            if cmd[:2] == ["bcftools", "norm"]:
                norm_vcf.write_bytes(b"")  # empty file
            return ""

        mock_run_tool.side_effect = side_effect
        vcf = tmp_path / "input.vcf.gz"
        vcf.touch()
        ref = tmp_path / "ref.fa"
        ref.touch()

        filter_vcf(vcf, ref, tmp_path, min_qual=15.0, min_dp=5)

        # bcftools view should NOT have -i flag (empty VCF)
        view_calls = [
            c[0][0] for c in mock_run_tool.call_args_list if c[0][0][:2] == ["bcftools", "view"]
        ]
        assert len(view_calls) == 1
        assert "-i" not in view_calls[0]

    @patch("open_pacmuci.vcf.run_tool")
    def test_nonempty_vcf_uses_qual_only_filter(self, mock_run_tool, tmp_path):
        """Non-empty VCF uses QUAL filter only (no INFO/DP which may not exist)."""
        norm_vcf = tmp_path / "normalized.vcf.gz"

        def side_effect(cmd):
            if cmd[:2] == ["bcftools", "norm"]:
                # Write a non-empty file to simulate records
                norm_vcf.write_bytes(b"\x00" * 100)
            return ""

        mock_run_tool.side_effect = side_effect
        vcf = tmp_path / "input.vcf.gz"
        vcf.touch()
        ref = tmp_path / "ref.fa"
        ref.touch()

        filter_vcf(vcf, ref, tmp_path, min_qual=15.0, min_dp=5)

        view_calls = [
            c[0][0] for c in mock_run_tool.call_args_list if c[0][0][:2] == ["bcftools", "view"]
        ]
        assert len(view_calls) == 1
        view_cmd = view_calls[0]
        assert "-i" in view_cmd
        i_idx = view_cmd.index("-i")
        expr = view_cmd[i_idx + 1]
        assert "QUAL" in expr
        # Should NOT reference INFO/DP (may not exist in Clair3 output)
        assert "INFO/DP" not in expr


class TestFilterVcfQuality:
    """Tests for VCF quality filter parameters."""

    @patch("open_pacmuci.vcf.run_tool")
    def test_filter_vcf_includes_quality_expression(self, mock_run_tool, tmp_path):
        """filter_vcf passes QUAL filter to bcftools view for non-empty VCFs."""
        norm_vcf = tmp_path / "normalized.vcf.gz"

        def side_effect(cmd):
            if cmd[:2] == ["bcftools", "norm"]:
                norm_vcf.write_bytes(b"\x00" * 100)  # non-empty
            return ""

        mock_run_tool.side_effect = side_effect
        vcf = tmp_path / "input.vcf.gz"
        vcf.touch()
        ref = tmp_path / "ref.fa"
        ref.touch()

        filter_vcf(vcf, ref, tmp_path, min_qual=15.0, min_dp=5)

        # Find the bcftools view call
        view_calls = [
            c[0][0] for c in mock_run_tool.call_args_list if c[0][0][:2] == ["bcftools", "view"]
        ]
        assert len(view_calls) == 1
        view_cmd = view_calls[0]
        assert "-i" in view_cmd
        i_idx = view_cmd.index("-i")
        expr = view_cmd[i_idx + 1]
        assert "QUAL" in expr

    @patch("open_pacmuci.vcf.run_tool", return_value="")
    def test_filter_vcf_default_params(self, mock_run_tool, tmp_path):
        """filter_vcf works with default parameters (backward compatible)."""
        vcf = tmp_path / "input.vcf.gz"
        vcf.touch()
        ref = tmp_path / "ref.fa"
        ref.touch()

        # Should not raise with no extra args
        filter_vcf(vcf, ref, tmp_path)


class TestParseVcfGenotypes:
    """Tests for parse_vcf_genotypes."""

    @patch("open_pacmuci.vcf.run_tool")
    def test_parses_genotype_fields(self, mock_run_tool):
        mock_run_tool.return_value = "contig_51\t100\tA\tT\t0/1\n"
        result = parse_vcf_genotypes(Path("/fake.vcf"))
        assert len(result) == 1
        assert result[0]["pos"] == 100
        assert result[0]["genotype"] == "0/1"

    @patch("open_pacmuci.vcf.run_tool")
    def test_handles_runtime_error(self, mock_run_tool):
        mock_run_tool.side_effect = RuntimeError("fail")
        assert parse_vcf_genotypes(Path("/fake.vcf")) == []


class TestParseVcfVariants:
    """Tests for parse_vcf_variants."""

    @patch("open_pacmuci.vcf.run_tool")
    def test_parses_pos_and_qual(self, mock_run_tool):
        """Parses position and quality from bcftools query output."""
        mock_run_tool.return_value = "100\t23.4\n200\t15.7\n"
        result = parse_vcf_variants(Path("/fake.vcf"))
        assert len(result) == 2
        assert result[0] == {"pos": 100, "qual": 23.4}
        assert result[1] == {"pos": 200, "qual": 15.7}

    @patch("open_pacmuci.vcf.run_tool")
    def test_handles_runtime_error(self, mock_run_tool):
        """Returns empty list on RuntimeError."""
        mock_run_tool.side_effect = RuntimeError("fail")
        assert parse_vcf_variants(Path("/fake.vcf")) == []

    @patch("open_pacmuci.vcf.run_tool")
    def test_skips_malformed_lines(self, mock_run_tool):
        """Skips lines with non-numeric values."""
        mock_run_tool.return_value = "100\t23.4\nbad\tline\n300\t10.0\n"
        result = parse_vcf_variants(Path("/fake.vcf"))
        assert len(result) == 2
        assert result[0]["pos"] == 100
        assert result[1]["pos"] == 300

    @patch("open_pacmuci.vcf.run_tool")
    def test_empty_output(self, mock_run_tool):
        """Returns empty list for empty output."""
        mock_run_tool.return_value = ""
        assert parse_vcf_variants(Path("/fake.vcf")) == []
