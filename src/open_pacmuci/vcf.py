"""VCF parsing and filtering utilities for open-pacmuci."""

from __future__ import annotations

from pathlib import Path

from open_pacmuci.tools import run_tool


def filter_vcf(
    vcf_path: Path,
    reference_path: Path,
    output_dir: Path,
    min_qual: float = 0.0,
    min_dp: int = 0,
    platform: str = "hifi",
    min_qual_indel: float | None = None,
) -> Path:
    """Normalize and filter a VCF file with bcftools.

    Runs ``bcftools norm -f <reference>`` followed by
    ``bcftools view`` with quality filters and indexes the result.

    Args:
        vcf_path: Path to input VCF (may be gzipped).
        reference_path: Path to reference FASTA for left-normalisation.
        output_dir: Directory for output files.
        min_qual: Minimum QUAL score to keep a PASS variant (0 = no filter).
        min_dp: Accepted for API compatibility but **not applied**.
            Clair3 HiFi places depth in FORMAT/DP (per-sample), not
            INFO/DP.  Filtering on INFO/DP would crash on Clair3 output.
            QUAL-only filtering is used instead since QUAL already
            integrates depth information.
        platform: Sequencing platform (``hifi`` or ``ont``).
        min_qual_indel: Optional lower QUAL floor for 1 bp indels.  When
            ``platform=="ont"`` and this is None, defaults to ``1.0`` so
            LowQual frameshift candidates are not silently dropped
            (Issue 6).  HiFi keeps PASS-only filtering unless overridden.

    Returns:
        Path to the filtered, indexed VCF (``variants.vcf.gz``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if min_qual_indel is None and platform == "ont":
        min_qual_indel = 1.0

    norm_vcf = output_dir / "normalized.vcf.gz"
    filtered = output_dir / "variants.vcf.gz"

    # Left-normalise indels against the reference
    run_tool(
        [
            "bcftools",
            "norm",
            "-f",
            str(reference_path),
            "-o",
            str(norm_vcf),
            "-O",
            "z",
            str(vcf_path),
        ]
    )

    # Skip quality filtering on empty VCFs (no records = no fields to filter).
    # Clair3 produces header-only VCFs for wild-type samples with no variants.
    # Note: a bgzipped header-only VCF may still have non-zero size; this
    # heuristic is conservative (may apply QUAL filter to header-only files,
    # which bcftools handles gracefully). A precise check would use
    # `bcftools view -H | wc -l` but adds an extra subprocess call.
    is_empty = not norm_vcf.exists() or norm_vcf.stat().st_size == 0

    view_cmd = [
        "bcftools",
        "view",
    ]

    if not is_empty and (min_qual > 0 or min_qual_indel is not None):
        # ONT: rescue 1 bp indels at a lower QUAL (incl. LowQual FILTER).
        # HiFi / default: PASS + QUAL floor only.
        if min_qual_indel is not None:
            expr = (
                f'(FILTER="PASS" && QUAL>={min_qual}) || '
                f'(abs(strlen(REF)-strlen(ALT))=1 && QUAL>={min_qual_indel})'
            )
            view_cmd.extend(["-i", expr])
        else:
            view_cmd.extend(["-f", "PASS"])
            if min_qual > 0:
                view_cmd.extend(["-i", f"QUAL>={min_qual}"])
    else:
        view_cmd.extend(["-f", "PASS"])

    view_cmd.extend(
        [
            "-o",
            str(filtered),
            "-O",
            "z",
            str(norm_vcf),
        ]
    )
    run_tool(view_cmd)
    run_tool(["bcftools", "index", str(filtered)])

    # Remove intermediate normalized VCF
    norm_vcf.unlink(missing_ok=True)

    return filtered


def parse_vcf_genotypes(vcf_path: Path) -> list[dict]:
    """Parse VCF to extract variant positions and genotypes.

    Returns list of dicts with keys: chrom, pos, ref, alt, genotype.
    """
    try:
        output = run_tool(
            [
                "bcftools",
                "query",
                "-f",
                "%CHROM\\t%POS\\t%REF\\t%ALT\\t[%GT]\\n",
                str(vcf_path),
            ]
        )
    except RuntimeError:
        return []

    variants: list[dict] = []
    for line in output.strip().splitlines():
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) < 5:
            continue
        variants.append(
            {
                "chrom": fields[0],
                "pos": int(fields[1]),
                "ref": fields[2],
                "alt": fields[3],
                "genotype": fields[4],
            }
        )
    return variants


def parse_vcf_variants(vcf_path: Path) -> list[dict]:
    """Parse VCF to extract variant positions and quality scores."""
    try:
        output = run_tool(["bcftools", "query", "-f", "%POS\\t%QUAL\\n", str(vcf_path)])
    except RuntimeError:
        return []
    variants = []
    for line in output.strip().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                variants.append({"pos": int(parts[0]), "qual": float(parts[1])})
            except ValueError:
                continue
    return variants
