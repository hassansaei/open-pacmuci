"""Variant calling with Clair3 and VCF processing with bcftools."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from open_pacmuci.tools import run_tool
from open_pacmuci.vcf import filter_vcf, parse_vcf_genotypes, parse_vcf_variants

__all__ = [
    "call_variants_per_allele",
    "disambiguate_same_length_alleles",
    "extract_allele_reads",
    "filter_vcf",
    "parse_vcf_genotypes",
    "parse_vcf_variants",
    "run_clair3",
]

logger = logging.getLogger(__name__)


def extract_allele_reads(
    bam_path: Path,
    contig_names: str | list[str],
    output_dir: Path,
) -> Path:
    """Extract reads mapped to one or more contigs from a BAM file.

    Uses ``samtools view -b`` to subset the BAM to the specified contig(s)
    and indexes the result.

    Args:
        bam_path: Path to the full mapping BAM.
        contig_names: Single contig name or list of contig names to extract.
        output_dir: Directory for output files.

    Returns:
        Path to the extracted, indexed BAM file.
    """
    if isinstance(contig_names, str):
        contig_names = [contig_names]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_bam = output_dir / "allele_reads.bam"

    run_tool(
        [
            "samtools",
            "view",
            "-b",
            "-o",
            str(out_bam),
            str(bam_path),
            *contig_names,
        ]
    )
    run_tool(["samtools", "index", str(out_bam)])

    return out_bam


def _extract_and_remap_reads(
    bam_path: Path,
    cluster_contigs: list[str],
    peak_contig: str,
    reference_path: Path,
    output_dir: Path,
    threads: int = 4,
    preset: str = "map-hifi",
) -> Path:
    """Extract reads from cluster contigs, convert to FASTQ, remap to peak contig.

    Reads from the ladder mapping are spread across multiple contigs in a
    cluster (e.g. contig_48 through contig_54).  Clair3 needs all reads
    aligned to a *single* reference contig to call variants.  This function
    extracts the cluster reads, converts to FASTQ, extracts the single peak
    contig as a mini-reference, and remaps with minimap2.

    Args:
        bam_path: Full ladder mapping BAM.
        cluster_contigs: All contig names in the allele's cluster.
        peak_contig: The single peak contig name to remap against.
        reference_path: Full ladder reference FASTA (for extracting the contig).
        output_dir: Working directory for intermediate files.
        threads: Thread count for minimap2/samtools.
        preset: minimap2 preset (default ``"map-hifi"``). Use ``"lr:hq"`` for ONT.

    Returns:
        Path to the remapped, sorted, indexed BAM.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract reads from all cluster contigs
    cluster_bam = extract_allele_reads(bam_path, cluster_contigs, output_dir)

    # 2. Convert to FASTQ
    fastq_path = output_dir / "cluster_reads.fq"
    stdout = run_tool(["samtools", "fastq", str(cluster_bam)])
    fastq_path.write_text(stdout)

    # 3. Extract peak contig as mini-reference
    contig_ref = output_dir / f"{peak_contig}.fa"
    stdout = run_tool(
        [
            "samtools",
            "faidx",
            str(reference_path),
            peak_contig,
        ]
    )
    contig_ref.write_text(stdout)
    run_tool(["samtools", "faidx", str(contig_ref)])

    # 4. Remap to peak contig
    sam_path = output_dir / "remapped.sam"
    sam_output = run_tool(
        [
            "minimap2",
            "-a",
            "-x",
            preset,
            "-t",
            str(threads),
            str(contig_ref),
            str(fastq_path),
        ]
    )
    sam_path.write_text(sam_output)

    # 5. Sort and index
    remapped_bam = output_dir / "allele_reads.bam"
    run_tool(
        [
            "samtools",
            "sort",
            "-@",
            str(threads),
            "-o",
            str(remapped_bam),
            str(sam_path),
        ]
    )
    run_tool(["samtools", "index", str(remapped_bam)])

    # Clean up intermediates
    sam_path.unlink(missing_ok=True)
    fastq_path.unlink(missing_ok=True)

    return remapped_bam


def run_clair3(
    bam_path: Path,
    reference_path: Path,
    output_dir: Path,
    model_path: str = "",
    platform: str = "hifi",
    threads: int = 4,
) -> Path:
    """Run the Clair3 variant caller on a BAM file.

    Args:
        bam_path: Path to input BAM (reads for one allele).
        reference_path: Path to the reference FASTA.
        output_dir: Directory for Clair3 output.
        model_path: Path to Clair3 model directory (optional).
        platform: Sequencing platform (default ``"hifi"``).
        threads: Number of threads (default 4).

    Returns:
        Path to the Clair3 output VCF (``merge_output.vcf.gz``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running Clair3 on %s", bam_path.name)

    cmd = [
        "run_clair3.sh",
        f"--bam_fn={bam_path}",
        f"--ref_fn={reference_path}",
        f"--output={output_dir}",
        f"--threads={threads}",
        f"--platform={platform}",
        "--sample_name=sample",
        # Our contigs are named contig_N, not chr1..22/X/Y, so Clair3
        # must be told to process all contigs.
        "--include_all_ctgs",
    ]
    if model_path:
        cmd.append(f"--model_path={model_path}")

    run_tool(cmd)

    return output_dir / "merge_output.vcf.gz"


def disambiguate_same_length_alleles(
    bam_path: Path,
    reference_path: Path,
    alleles: dict,
    output_dir: Path,
    clair3_model: str = "",
    threads: int = 4,
    min_qual: float = 5.0,
    min_dp: int = 5,
    platform: str = "hifi",
    preset: str = "map-hifi",
) -> dict:
    """Disambiguate same-length alleles using Clair3 genotype calls.

    Runs Clair3 on all reads, checks for heterozygous (0/1) variants.
    If found, creates two VCFs: one with only hom-alt variants (WT allele),
    one with all variants (mutant allele).

    Returns dict mapping allele key to filtered VCF path, plus a
    ``"homozygous"`` boolean key indicating whether the alleles are identical.
    """
    allele_info = alleles["allele_1"]
    contig_name = allele_info["contig_name"]
    cluster_contigs = allele_info["cluster_contigs"]
    merged_dir = output_dir / "merged"

    # Remap all cluster reads to peak contig
    merged_bam = _extract_and_remap_reads(
        bam_path,
        cluster_contigs,
        contig_name,
        reference_path,
        merged_dir,
        threads,
        preset=preset,
    )

    # Run Clair3
    contig_ref = merged_dir / f"{contig_name}.fa"
    clair3_dir = merged_dir / "clair3"
    raw_vcf = run_clair3(
        merged_bam,
        contig_ref,
        clair3_dir,
        model_path=clair3_model,
        platform=platform,
        threads=threads,
    )
    filtered_vcf = filter_vcf(
        raw_vcf,
        contig_ref,
        merged_dir,
        min_qual=min_qual,
        min_dp=min_dp,
        platform=platform,
    )

    # Check genotypes
    variants = parse_vcf_genotypes(filtered_vcf)
    het_variants = [
        v
        for v in variants
        if any(sep in v["genotype"] for sep in ("/", "|"))
        and v["genotype"] not in ("0/0", "1/1", "./.", "0|0", "1|1", ".|.")
    ]

    results: dict = {}

    if not het_variants:
        # Truly homozygous -- no het variants found
        results["allele_1"] = filtered_vcf
        results["homozygous"] = True
        return results

    # Compound heterozygous -- split into two VCFs

    # allele_1 (WT): exclude het variants, keep only hom-alt
    wt_vcf = output_dir / "allele_1" / "variants.vcf.gz"
    wt_vcf.parent.mkdir(parents=True, exist_ok=True)
    run_tool(
        [
            "bcftools",
            "view",
            "-i",
            'GT="1/1" || GT="1|1"',
            "-o",
            str(wt_vcf),
            "-O",
            "z",
            str(filtered_vcf),
        ]
    )
    run_tool(["bcftools", "index", str(wt_vcf)])
    results["allele_1"] = wt_vcf

    # allele_2 (mutant): include all variants
    mut_vcf = output_dir / "allele_2" / "variants.vcf.gz"
    mut_vcf.parent.mkdir(parents=True, exist_ok=True)
    run_tool(
        [
            "bcftools",
            "view",
            "-o",
            str(mut_vcf),
            "-O",
            "z",
            str(filtered_vcf),
        ]
    )
    run_tool(["bcftools", "index", str(mut_vcf)])
    results["allele_2"] = mut_vcf

    # Note: bcftools consensus applies the ALT allele for het (0/1) calls
    # by default, so the allele_2 VCF with het genotypes will produce the
    # correct mutant consensus without needing to force hom-alt genotypes.
    results["homozygous"] = False
    return results


def call_variants_per_allele(
    bam_path: Path,
    reference_path: Path,
    alleles: dict,
    output_dir: Path,
    clair3_model: str = "",
    threads: int = 4,
    min_qual: float = 5.0,
    min_dp: int = 5,
    platform: str = "hifi",
    preset: str = "map-hifi",
) -> dict[str, Path]:
    """Run variant calling for each detected allele.

    For each allele key in *alleles* (``"allele_1"`` and optionally
    ``"allele_2"``), this function:

    1. Extracts the reads mapped to the allele's best-matching contig.
    2. Calls variants with Clair3.
    3. Filters the VCF with :func:`filter_vcf`.

    For same-length alleles, delegates to :func:`disambiguate_same_length_alleles`.

    Args:
        bam_path: Path to the full mapping BAM.
        reference_path: Path to the ladder reference FASTA.
        alleles: Allele detection result from :func:`~open_pacmuci.alleles.detect_alleles`.
        output_dir: Base output directory.
        clair3_model: Path to Clair3 model directory (optional).
        threads: Number of threads (default 4).
        min_qual: Minimum QUAL score for VCF filtering (default 5.0).
            Low-confidence variants are retained but penalized in the
            confidence scoring system rather than hard-filtered.
        min_dp: Minimum INFO/DP for VCF filtering (default 5).
        platform: Sequencing platform for Clair3 (default ``"hifi"``).
        preset: minimap2 preset for remapping (default ``"map-hifi"``).

    Returns:
        Dictionary mapping allele key (``"allele_1"`` / ``"allele_2"``) to
        the filtered VCF path.
    """
    if alleles.get("same_length"):
        logger.info("Same-length alleles detected, using disambiguation")
        disambig = disambiguate_same_length_alleles(
            bam_path,
            reference_path,
            alleles,
            output_dir,
            clair3_model,
            threads,
            min_qual,
            min_dp,
            platform=platform,
            preset=preset,
        )
        alleles["homozygous"] = disambig.pop("homozygous")
        return disambig

    # Collect allele keys to process (skip allele_2 when homozygous)
    allele_keys = [
        k
        for k in ("allele_1", "allele_2")
        if k in alleles and not (alleles.get("homozygous") and k == "allele_2")
    ]

    def _process_allele(allele_key: str) -> tuple[str, Path]:
        allele_info = alleles[allele_key]
        # Use the peak contig name from allele detection (contig_N where N is
        # canonical X repeat count, NOT total length).  Fall back to computing
        # from length for backwards compatibility with older alleles.json.
        contig_name = allele_info.get("contig_name", f"contig_{allele_info['length']}")
        cluster_contigs = allele_info.get("cluster_contigs", [contig_name])
        allele_dir = output_dir / allele_key

        # Extract reads from ALL contigs in the cluster, then remap to
        # the peak contig.  Reads spread across multiple ladder contigs
        # due to length variation; Clair3 needs them all aligned to a
        # single reference contig to call variants effectively.
        allele_bam = _extract_and_remap_reads(
            bam_path,
            cluster_contigs,
            contig_name,
            reference_path,
            allele_dir,
            threads,
            preset=preset,
        )

        # Run Clair3 against the single-contig reference (created during
        # remapping).  Using the single contig rather than the full ladder
        # avoids Clair3 scanning 150 empty contigs.
        contig_ref = allele_dir / f"{contig_name}.fa"
        clair3_dir = allele_dir / "clair3"
        # Split thread budget across parallel alleles to avoid CPU oversubscription
        per_allele_threads = max(1, threads // len(allele_keys))
        vcf = run_clair3(
            allele_bam,
            contig_ref,
            clair3_dir,
            model_path=clair3_model,
            platform=platform,
            threads=per_allele_threads,
        )

        # Filter VCF
        filtered = filter_vcf(
            vcf,
            contig_ref,
            allele_dir,
            min_qual=min_qual,
            min_dp=min_dp,
            platform=platform,
        )
        return allele_key, filtered

    # Process both alleles in parallel when they are independent
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_process_allele, k): k for k in allele_keys}
        results: dict[str, Path] = {}
        for future in futures:
            key, vcf_path = future.result()
            results[key] = vcf_path

    return results
