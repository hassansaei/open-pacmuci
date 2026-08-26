"""Click CLI for open-pacmuci pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from open_pacmuci.version import __version__

PLATFORM_PRESETS: dict[str, str] = {"hifi": "map-hifi", "ont": "lr:hq"}


@click.group()
@click.version_option(version=__version__, prog_name="open-pacmuci")
@click.option(
    "-v", "--verbose", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)."
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output.")
@click.pass_context
def main(ctx: click.Context, verbose: int, quiet: bool) -> None:
    """open-pacmuci: MUC1 VNTR analysis pipeline for PacBio HiFi and ONT amplicon data."""
    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(name)s %(levelname)s: %(message)s",
    )


@main.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="reference_ladder.fa",
    help="Output FASTA path.",
)
@click.option("--min-units", type=int, default=1, help="Minimum repeat units.")
@click.option("--max-units", type=int, default=150, help="Maximum repeat units.")
@click.option("--flank-length", type=int, default=500, help="Flanking sequence length (bp).")
@click.option(
    "--repeats-db",
    type=click.Path(exists=True),
    default=None,
    help="Custom repeat dictionary JSON.",
)
def ladder(
    output: str,
    min_units: int,
    max_units: int,
    flank_length: int,
    repeats_db: str | None,
) -> None:
    """Generate or regenerate the reference ladder FASTA."""
    from open_pacmuci.config import load_repeat_dictionary
    from open_pacmuci.ladder import generate_ladder_fasta

    rd = load_repeat_dictionary(Path(repeats_db) if repeats_db else None)
    out_path = generate_ladder_fasta(rd, Path(output), min_units, max_units, flank_length)
    click.echo(f"Ladder written to {out_path} ({max_units - min_units + 1} contigs)")


@main.command(name="map")
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input FASTQ or BAM file.",
)
@click.option(
    "--reference",
    "-r",
    type=click.Path(exists=True),
    default=None,
    help="Reference FASTA (defaults to bundled ladder).",
)
@click.option("--output-dir", "-o", type=click.Path(), default=".", help="Output directory.")
@click.option("--threads", "-t", type=int, default=4, help="Number of threads.")
@click.option(
    "--platform",
    type=click.Choice(["hifi", "ont"], case_sensitive=False),
    default="hifi",
    help="Sequencing platform (default: hifi).",
)
@click.option(
    "--minimap2-preset",
    type=str,
    default=None,
    help="minimap2 -x preset (auto-selected from --platform if not set).",
)
def map_cmd(
    input_path: str,
    reference: str | None,
    output_dir: str,
    threads: int,
    platform: str,
    minimap2_preset: str | None,
) -> None:
    """Map reads to the ladder reference with minimap2."""
    from open_pacmuci.mapping import map_reads
    from open_pacmuci.tools import check_tools

    check_tools(["minimap2", "samtools"])

    preset = minimap2_preset or PLATFORM_PRESETS[platform]
    ref = Path(reference) if reference else _bundled_reference()
    bam = map_reads(Path(input_path), ref, Path(output_dir), threads, preset=preset)
    click.echo(f"Mapping written to {bam}")


@main.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input BAM file (mapped to ladder).",
)
@click.option("--min-coverage", type=int, default=10, help="Minimum read coverage.")
@click.option("--output-dir", "-o", type=click.Path(), default=".", help="Output directory.")
def alleles(input_path: str, min_coverage: int, output_dir: str) -> None:
    """Determine allele lengths from mapping."""
    from open_pacmuci.alleles import detect_alleles, parse_idxstats
    from open_pacmuci.mapping import get_idxstats

    bam = Path(input_path)
    idxstats_output = get_idxstats(bam)
    counts = parse_idxstats(idxstats_output)
    result = detect_alleles(counts, min_coverage, bam_path=bam)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "alleles.json"
    out_file.write_text(json.dumps(result, indent=2) + "\n")
    click.echo(f"Alleles: {result}")


@main.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input BAM file.",
)
@click.option(
    "--reference",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Reference FASTA.",
)
@click.option(
    "--alleles-json",
    "-a",
    required=True,
    type=click.Path(exists=True),
    help="Alleles JSON from 'alleles' command.",
)
@click.option("--output-dir", "-o", type=click.Path(), default=".", help="Output directory.")
@click.option("--clair3-model", type=str, default="", help="Path to Clair3 model.")
@click.option("--threads", "-t", type=int, default=4, help="Number of threads.")
@click.option(
    "--min-qual",
    type=float,
    default=5.0,
    help="Minimum QUAL score for VCF filtering (default 5.0).",
)
@click.option(
    "--platform",
    type=click.Choice(["hifi", "ont"], case_sensitive=False),
    default="hifi",
    help="Sequencing platform (default: hifi).",
)
@click.option(
    "--minimap2-preset",
    type=str,
    default=None,
    help="minimap2 -x preset (auto-selected from --platform if not set).",
)
def call(
    input_path: str,
    reference: str,
    alleles_json: str,
    output_dir: str,
    clair3_model: str,
    threads: int,
    min_qual: float,
    platform: str,
    minimap2_preset: str | None,
) -> None:
    """Call variants with Clair3."""
    from open_pacmuci.calling import call_variants_per_allele
    from open_pacmuci.tools import check_tools

    check_tools(["minimap2", "samtools", "bcftools", "run_clair3.sh"])

    try:
        alleles_data = json.loads(Path(alleles_json).read_text())
    except (json.JSONDecodeError, ValueError) as exc:
        click.echo(f"Error: failed to parse alleles JSON: {exc}", err=True)
        sys.exit(1)
    preset = minimap2_preset or PLATFORM_PRESETS[platform]
    vcfs = call_variants_per_allele(
        Path(input_path),
        Path(reference),
        alleles_data,
        Path(output_dir),
        clair3_model,
        threads,
        min_qual=min_qual,
        platform=platform,
        preset=preset,
    )
    for key, vcf in vcfs.items():
        click.echo(f"{key}: {vcf}")


@main.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input BAM file.",
)
@click.option(
    "--reference",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Reference FASTA.",
)
@click.option(
    "--alleles-json",
    "-a",
    required=True,
    type=click.Path(exists=True),
    help="Alleles JSON.",
)
@click.option("--output-dir", "-o", type=click.Path(), default=".", help="Output directory.")
def consensus(
    input_path: str,
    reference: str,
    alleles_json: str,
    output_dir: str,
) -> None:
    """Build per-allele consensus sequences."""
    from open_pacmuci.consensus import build_consensus_per_allele
    from open_pacmuci.tools import check_tools

    check_tools(["samtools", "bcftools"])

    try:
        alleles_data = json.loads(Path(alleles_json).read_text())
    except (json.JSONDecodeError, ValueError) as exc:
        click.echo(f"Error: failed to parse alleles JSON: {exc}", err=True)
        sys.exit(1)
    out = Path(output_dir)
    vcf_paths: dict[str, Path] = {}
    for key in ["allele_1", "allele_2"]:
        vcf = out / key / "variants.vcf.gz"
        if vcf.exists():
            vcf_paths[key] = vcf

    if not vcf_paths:
        click.echo(
            "Warning: no VCF files found. Expected allele_1/variants.vcf.gz "
            "or allele_2/variants.vcf.gz in the output directory.",
            err=True,
        )

    fastas = build_consensus_per_allele(
        Path(reference),
        vcf_paths,
        alleles_data,
        out,
    )
    for key, fa in fastas.items():
        click.echo(f"{key}: {fa}")


@main.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input consensus FASTA.",
)
@click.option(
    "--repeats-db",
    type=click.Path(exists=True),
    default=None,
    help="Custom repeat dictionary JSON.",
)
@click.option("--output-dir", "-o", type=click.Path(), default=".", help="Output directory.")
def classify(
    input_path: str,
    repeats_db: str | None,
    output_dir: str,
) -> None:
    """Classify repeat units in a consensus sequence."""
    from open_pacmuci.classify import classify_sequence
    from open_pacmuci.config import load_repeat_dictionary

    rd = load_repeat_dictionary(Path(repeats_db) if repeats_db else None)

    lines = Path(input_path).read_text().strip().splitlines()
    sequence = "".join(line for line in lines if not line.startswith(">"))

    result = classify_sequence(sequence, rd)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "repeats.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "repeats.txt").write_text(result["structure"] + "\n")

    click.echo(f"Structure: {result['structure']}")
    if result["mutations_detected"]:
        click.echo(f"Mutations: {len(result['mutations_detected'])} detected")


@main.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input FASTQ or BAM file.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="results",
    help="Output directory.",
)
@click.option(
    "--reference",
    "-r",
    type=click.Path(exists=True),
    default=None,
    help="Reference FASTA (defaults to bundled ladder).",
)
# TODO: --config support for YAML configuration is planned for a future release
@click.option("--clair3-model", type=str, default="", help="Path to Clair3 model.")
@click.option("--threads", "-t", type=int, default=4, help="Number of threads.")
@click.option("--min-coverage", type=int, default=10, help="Minimum read coverage.")
@click.option(
    "--min-qual",
    type=float,
    default=5.0,
    help="Minimum QUAL score for VCF filtering (default 5.0).",
)
@click.option(
    "--report/--no-report",
    default=False,
    help="Generate HTML report (requires jinja2: pip install open-pacmuci[report]).",
)
@click.option(
    "--platform",
    type=click.Choice(["hifi", "ont"], case_sensitive=False),
    default="hifi",
    help="Sequencing platform (default: hifi).",
)
@click.option(
    "--minimap2-preset",
    type=str,
    default=None,
    help="minimap2 -x preset (auto-selected from --platform if not set).",
)
def run(
    input_path: str,
    output_dir: str,
    reference: str | None,
    clair3_model: str,
    threads: int,
    min_coverage: int,
    min_qual: float,
    report: bool,
    platform: str,
    minimap2_preset: str | None,
) -> None:
    """Run the full open-pacmuci pipeline."""
    from open_pacmuci.alleles import detect_alleles, parse_idxstats
    from open_pacmuci.calling import call_variants_per_allele
    from open_pacmuci.classify import classify_sequence
    from open_pacmuci.config import load_repeat_dictionary
    from open_pacmuci.consensus import build_consensus_per_allele
    from open_pacmuci.mapping import get_idxstats, map_reads
    from open_pacmuci.tools import check_tools, get_tool_versions
    from open_pacmuci.vcf import parse_vcf_variants

    preset = minimap2_preset or PLATFORM_PRESETS[platform]

    check_tools(["minimap2", "samtools", "bcftools", "run_clair3.sh"])
    tool_versions = get_tool_versions(["minimap2", "samtools", "bcftools", "run_clair3.sh"])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = Path(reference) if reference else _bundled_reference()
    rd = load_repeat_dictionary()

    # Step 1: Map reads
    click.echo("Step 1/5: Mapping reads...")
    bam = map_reads(Path(input_path), ref, out, threads, preset=preset)

    # Step 2: Detect alleles
    click.echo("Step 2/5: Detecting alleles...")
    idxstats = get_idxstats(bam)
    counts = parse_idxstats(idxstats)
    alleles_result = detect_alleles(counts, min_coverage, bam_path=bam)
    (out / "alleles.json").write_text(json.dumps(alleles_result, indent=2) + "\n")
    click.echo(f"  Alleles: {alleles_result}")

    # Step 3: Call variants
    click.echo("Step 3/5: Calling variants...")
    vcf_paths = call_variants_per_allele(
        bam,
        ref,
        alleles_result,
        out,
        clair3_model,
        threads,
        min_qual=min_qual,
        platform=platform,
        preset=preset,
    )

    # Step 4: Build consensus
    click.echo("Step 4/5: Building consensus...")
    consensus_paths = build_consensus_per_allele(
        ref, vcf_paths, alleles_result, out, repeat_dict=rd
    )

    # Step 5: Classify repeats
    click.echo("Step 5/5: Classifying repeats...")
    from open_pacmuci.classify import reconcile_shared_frameshifts, validate_mutations_against_vcf

    all_results: dict[str, dict] = {}
    for allele_key, fa_path in consensus_paths.items():
        fa_lines = fa_path.read_text().strip().splitlines()
        sequence = "".join(line for line in fa_lines if not line.startswith(">"))
        result = classify_sequence(sequence, rd)

        # VCF-backed validation if VCF available
        if allele_key in vcf_paths:
            vcf_variants = parse_vcf_variants(vcf_paths[allele_key])
            result = validate_mutations_against_vcf(result, vcf_variants=vcf_variants)

        all_results[allele_key] = result

    # Demote weak duplicate frameshifts shared across alleles (ONT bleed)
    if len(all_results) >= 2 and not alleles_result.get("same_length"):
        all_results = reconcile_shared_frameshifts(all_results)

    for allele_key, result in all_results.items():
        click.echo(f"  {allele_key}: {result['structure']}")
        if result.get("allele_confidence") is not None:
            click.echo(f"    confidence: {result['allele_confidence']:.2f}")
        demoted = result.get("mutations_demoted") or []
        if demoted:
            click.echo(f"    demoted shared frameshifts: {len(demoted)}")

    # Write combined outputs
    (out / "repeats.json").write_text(json.dumps(all_results, indent=2) + "\n")
    structures = {k: v["structure"] for k, v in all_results.items()}
    (out / "repeats.txt").write_text("\n".join(f"{k}: {v}" for k, v in structures.items()) + "\n")

    # Summary
    summary = {
        "alleles": alleles_result,
        "classifications": {
            k: {
                "structure": v["structure"],
                "mutations": v["mutations_detected"],
                "mutations_demoted": v.get("mutations_demoted", []),
            }
            for k, v in all_results.items()
        },
        "tool_versions": tool_versions,
        "pipeline_version": __version__,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if report:
        try:
            from open_pacmuci.report import generate_report

            report_path = out / "report.html"
            generate_report(
                summary,
                report_path,
                sample_name=Path(input_path).stem,
                detailed_repeats=all_results,
            )
            click.echo(f"Report: {report_path}")
        except ImportError as e:
            click.echo(f"Warning: {e}", err=True)

    click.echo("Pipeline complete.")


@main.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to summary.json from a previous run.",
)
@click.option("--output", "-o", type=click.Path(), default="report.html", help="Output HTML path.")
@click.option("--sample-name", "-s", default=None, help="Sample name for report header.")
@click.option(
    "--repeats",
    type=click.Path(exists=True),
    default=None,
    help="Path to repeats.json for detailed repeat table.",
)
def report(input_path: str, output: str, sample_name: str | None, repeats: str | None) -> None:
    """Generate an HTML report from pipeline results."""
    import json

    try:
        from open_pacmuci.report import generate_report
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e

    try:
        summary = json.loads(Path(input_path).read_text())
    except (json.JSONDecodeError, ValueError) as e:
        click.echo(f"Error: Failed to parse JSON from {input_path}: {e}", err=True)
        raise SystemExit(1) from e

    name = sample_name or Path(input_path).parent.name

    detailed = None
    if repeats:
        try:
            detailed = json.loads(Path(repeats).read_text())
        except (json.JSONDecodeError, ValueError) as e:
            click.echo(f"Error: Failed to parse JSON from {repeats}: {e}", err=True)
            raise SystemExit(1) from e

    out_path = generate_report(summary, Path(output), sample_name=name, detailed_repeats=detailed)
    click.echo(f"Report written to {out_path}")


def _bundled_reference() -> Path:
    """Get path to the bundled reference ladder.

    Returns:
        Path to the bundled reference FASTA.

    Raises:
        SystemExit: If the bundled reference file does not exist.
    """
    import importlib.resources

    res = importlib.resources.files("open_pacmuci.data.reference").joinpath("reference_ladder.fa")
    ref = Path(str(res))
    if not ref.exists():
        click.echo(
            f"Bundled reference not found at {ref}. Run 'open-pacmuci ladder' to generate it.",
            err=True,
        )
        sys.exit(1)
    return ref
