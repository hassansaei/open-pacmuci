"""Per-allele consensus sequence generation with bcftools."""

from __future__ import annotations

import logging
from pathlib import Path

from open_pacmuci.config import RepeatDictionary
from open_pacmuci.tools import run_tool

logger = logging.getLogger(__name__)


def build_consensus(
    reference_path: Path,
    vcf_path: Path,
    output_path: Path,
) -> Path:
    """Generate a consensus FASTA by applying VCF variants to a reference.

    Runs ``bcftools consensus -f <reference> <vcf>`` and writes the stdout
    to *output_path*.

    Args:
        reference_path: Path to the reference FASTA (typically the matching
            ladder contig extracted with ``samtools faidx``).
        vcf_path: Path to the filtered VCF (may be gzipped).
        output_path: Destination path for the consensus FASTA.

    Returns:
        Path to the written consensus FASTA.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stdout = run_tool(
        [
            "bcftools",
            "consensus",
            "-f",
            str(reference_path),
            str(vcf_path),
        ]
    )

    output_path.write_text(stdout)
    return output_path


def _find_anchor(
    sequence: str,
    anchor: str,
    expected_pos: int,
    tolerance: int = 50,
) -> int | None:
    """Find anchor sequence near expected position.

    Searches for an exact substring match within ``expected_pos +/- tolerance``.
    Returns the position where the anchor ENDS (i.e., the start of the region
    after the anchor).

    Returns None if not found.
    """
    search_start = max(0, expected_pos - tolerance)
    search_end = min(len(sequence), expected_pos + tolerance + len(anchor))
    region = sequence[search_start:search_end]
    idx = region.find(anchor)
    if idx >= 0:
        return search_start + idx + len(anchor)
    return None


def trim_flanking(
    consensus_fasta: Path,
    flank_length: int,
    output_path: Path,
    repeat_dict: RepeatDictionary | None = None,
) -> Path:
    """Remove flanking sequences from a consensus FASTA, keeping only the VNTR.

    The ladder contigs have structure:
    ``[left_flank] [pre-repeats] [N * X] [after-repeats] [right_flank]``

    When *repeat_dict* is provided, uses anchor-based boundary detection
    that is resilient to indels in the flanking region (e.g. from Clair3
    false positives).  Falls back to fixed-position trim without it.

    Args:
        consensus_fasta: Path to the full-contig consensus FASTA.
        flank_length: Number of flanking bp on each side (must match the
            value used during ladder generation, typically 500).
        output_path: Destination path for the trimmed FASTA.
        repeat_dict: Optional repeat dictionary for anchor-based trimming.

    Returns:
        Path to the trimmed FASTA containing only the VNTR region.
    """
    lines = consensus_fasta.read_text().strip().splitlines()
    header = lines[0] if lines and lines[0].startswith(">") else ">consensus"
    sequence = "".join(line for line in lines if not line.startswith(">"))

    # Validate: if sequence is too short to trim, return it untrimmed
    if len(sequence) < 2 * flank_length:
        logger.warning(
            "Sequence length %d is shorter than 2 * flank_length (%d); "
            "returning full sequence untrimmed.",
            len(sequence),
            2 * flank_length,
        )
        output_path.write_text(f"{header}_vntr\n{sequence}\n")
        return output_path

    # Default: fixed-position trim
    left_trim = flank_length
    right_trim = len(sequence) - flank_length if flank_length > 0 else len(sequence)

    # Try anchor-based trimming if repeat_dict provided
    if repeat_dict is not None and flank_length > 0:
        # Left anchor: last 20bp of left flank + first 20bp of pre-repeat "1"
        if "1" in repeat_dict.repeats:
            left_anchor = repeat_dict.flanking_left[-20:] + repeat_dict.repeats["1"][:20]
            anchor_pos = _find_anchor(sequence, left_anchor, flank_length)
            if anchor_pos is not None:
                # anchor_pos points to end of anchor (= 20bp into repeat "1")
                # We want to start at the beginning of repeat "1"
                left_trim = anchor_pos - 20

        # Right anchor: last 20bp of after-repeat "9" + first 20bp of right flank
        if "9" in repeat_dict.repeats:
            right_anchor = repeat_dict.repeats["9"][-20:] + repeat_dict.flanking_right[:20]
            right_anchor_pos = _find_anchor(sequence, right_anchor, len(sequence) - flank_length)
            if right_anchor_pos is not None:
                right_trim = right_anchor_pos - 20  # end after repeat "9"

    vntr = sequence[left_trim:right_trim]

    output_path.write_text(f"{header}_vntr\n{vntr}\n")
    return output_path


def check_consensus_length(
    consensus_fasta: Path,
    expected_units: int,
    unit_length: int = 60,
    tolerance: int = 1,
) -> dict:
    """Compare trimmed consensus unit count to the detected allele length.

    Args:
        consensus_fasta: VNTR-only consensus FASTA.
        expected_units: Expected total repeat units (``allele.length``).
        unit_length: Repeat unit size in bp (default 60).
        tolerance: Allowed absolute difference in units (default ±1 for
            frameshifted alleles).

    Returns:
        Dict with ``expected_units``, ``observed_units``, ``observed_bp``,
        ``delta``, and ``passed`` (True if within tolerance).
    """
    lines = consensus_fasta.read_text().strip().splitlines()
    sequence = "".join(line for line in lines if not line.startswith(">"))
    observed_bp = len(sequence)
    observed_units = round(observed_bp / unit_length) if unit_length else 0
    delta = observed_units - expected_units
    return {
        "expected_units": expected_units,
        "observed_units": observed_units,
        "observed_bp": observed_bp,
        "delta": delta,
        "passed": abs(delta) <= tolerance,
    }


def build_consensus_per_allele(
    reference_path: Path,
    vcf_paths: dict[str, Path],
    alleles: dict,
    output_dir: Path,
    flank_length: int = 500,
    repeat_dict: RepeatDictionary | None = None,
) -> dict[str, Path]:
    """Build a consensus FASTA for each detected allele and trim flanking.

    For each allele, the function:

    1. Extracts the matching single contig from the ladder reference with
       ``samtools faidx`` (using the peak contig name from allele detection).
    2. Indexes the single-contig FASTA.
    3. Calls :func:`build_consensus` to apply the allele's VCF variants.
    4. Trims flanking sequences to isolate the VNTR region.

    Args:
        reference_path: Path to the full ladder reference FASTA.
        vcf_paths: Mapping from allele key (e.g. ``"allele_1"``) to VCF path.
        alleles: Allele detection result from
            :func:`~open_pacmuci.alleles.detect_alleles`.
        output_dir: Base output directory for consensus files.
        flank_length: Flanking sequence length used in ladder generation.

    Returns:
        Dictionary mapping allele key to the trimmed consensus FASTA path
        (containing only the VNTR region, ready for repeat classification).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    for allele_key, vcf_path in vcf_paths.items():
        allele_info = alleles[allele_key]
        # Use contig_name from allele detection (canonical repeat count),
        # not total length.  Fall back for backwards compatibility.
        contig_name = allele_info.get("contig_name", f"contig_{allele_info['length']}")

        # Extract the single matching contig from the full ladder reference
        contig_fa = output_dir / f"ref_{contig_name}.fa"
        stdout = run_tool(
            [
                "samtools",
                "faidx",
                str(reference_path),
                contig_name,
            ]
        )
        contig_fa.write_text(stdout)

        # Index the single-contig reference so bcftools consensus can use it
        run_tool(["samtools", "faidx", str(contig_fa)])

        # Build full consensus (with flanking)
        full_consensus = output_dir / f"consensus_{allele_key}_full.fa"
        build_consensus(contig_fa, vcf_path, full_consensus)

        # Trim flanking to get VNTR-only sequence for classification
        trimmed = output_dir / f"consensus_{allele_key}.fa"
        trim_flanking(full_consensus, flank_length, trimmed, repeat_dict=repeat_dict)
        results[allele_key] = trimmed

    return results
