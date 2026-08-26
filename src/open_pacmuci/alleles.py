"""Allele length detection from samtools idxstats output.

Peak contig selection uses two alignment quality metrics extracted from the
BAM file:

- **Alignment score (AS):** Higher AS means the read aligns better to the
  contig.  Reads from a 60-repeat allele produce the highest AS when
  aligned to the contig whose length matches the allele (contig_51 for
  51 canonical X repeats + 9 fixed = 60 total).

- **Indel length:** Lower mean indel length indicates a better length
  match between read and reference.  Reads aligned to a contig that is
  too short or too long accumulate large insertions or deletions in the
  CIGAR string.

Both metrics independently identify the correct contig in testing.  We
use AS as the primary selector because it integrates all alignment
factors (matches, mismatches, gaps) into a single score.  Mean indel
length is reported alongside for transparency.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TypedDict

from open_pacmuci.tools import run_tool_iter

# TypedDicts below document the expected structure of return values.
# Functions return plain dicts for mypy compatibility; these types are
# available for callers who want to annotate their own code.


class AlleleInfo(TypedDict):
    """Information about a single detected allele."""

    length: int
    reads: int
    canonical_repeats: int
    contig_name: str
    cluster_contigs: list[str]


class AlleleResult(TypedDict):
    """Result of allele detection for a sample."""

    allele_1: AlleleInfo
    allele_2: AlleleInfo
    homozygous: bool
    same_length: bool


logger = logging.getLogger(__name__)

# Number of fixed repeat units in the ladder reference (pre-repeats 1-5 + after-repeats 6-9).
# Each contig_N has N canonical X repeats plus these fixed repeats,
# so total allele length = N + PRE_AFTER_REPEAT_COUNT.
PRE_AFTER_REPEAT_COUNT = 9

# ONT amplicon ladder mappings often put >= min_coverage reads on every
# contig, merging the whole ladder into one gap-cluster.  Spans wider than
# this switch from indel-valley splitting to read-count peak finding.
MAX_CLUSTER_SPAN = 40

# Minimum distance (canonical repeats) between two allele peaks.
MIN_PEAK_SEPARATION = 10

# Secondary peak must reach this fraction of the primary peak height.
MIN_SECONDARY_HEIGHT_FRACTION = 0.05

# Indel-valley contigs must have at least this fraction of the cluster's
# max per-contig read count (ignores sparse long-tail false valleys).
MIN_VALLEY_COVERAGE_FRACTION = 0.1

# Half-width of the contig window retained around each prominence peak.
PEAK_CLUSTER_HALF_WIDTH = 5


def parse_idxstats(idxstats_output: str) -> dict[int, int]:
    """Parse samtools idxstats output into repeat_count -> read_count mapping.

    Expects contig names like 'contig_60' where 60 is the number of
    canonical X repeats in that contig.

    Args:
        idxstats_output: Raw text output from ``samtools idxstats``.

    Returns:
        Dictionary mapping canonical repeat count (int) to mapped read
        count (int).  The '*' unmapped line is excluded.
    """
    counts: dict[int, int] = {}

    for line in idxstats_output.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue

        contig_name = parts[0]
        if contig_name == "*":
            continue

        mapped_reads = int(parts[2])

        match = re.search(r"_(\d+)$", contig_name)
        if match:
            repeat_count = int(match.group(1))
            counts[repeat_count] = mapped_reads

    return counts


def _parse_cigar_indel_bp(cigar: str) -> int:
    """Sum of insertion and deletion bases from a CIGAR string."""
    total = 0
    for m in re.finditer(r"(\d+)([ID])", cigar):
        total += int(m.group(1))
    return total


def refine_peak_contig(
    bam_path: Path,
    cluster_contigs: list[str],
) -> dict:
    """Select the best contig from a cluster using alignment quality metrics.

    Scans all reads mapped to the cluster contigs and computes per-contig
    mean indel length (from CIGAR), mean alignment score (AS), and
    length-normalized AS (AS / query length).

    Selection priority (Issue 2 / ONT):

    1. Lowest mean indel bp (best length match to the allele)
    2. Tie-break with highest length-normalized AS
    3. Tie-break with highest absolute mean AS

    Absolute AS alone favors longer ladder contigs because AS scales with
    alignment length, which systematically mis-picks peaks on ONT amplicons.

    Args:
        bam_path: Path to the ladder mapping BAM (indexed).
        cluster_contigs: List of contig names in the cluster
            (e.g. ``["contig_48", ..., "contig_54"]``).

    Returns:
        Dictionary with:

        - ``best_contig`` (str): name of the best-matching contig
        - ``metrics`` (dict): per-contig ``{mean_as, mean_as_norm,
          mean_indel_bp, reads}``
    """
    # Accumulate per-contig stats
    contig_stats: dict[str, dict] = {
        c: {"as_sum": 0, "indel_sum": 0, "query_len_sum": 0, "count": 0}
        for c in cluster_contigs
    }

    for line in run_tool_iter(["samtools", "view", str(bam_path), *cluster_contigs]):
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        fields = line.split("\t")
        if len(fields) < 11:
            continue

        contig = fields[2]
        if contig not in contig_stats:
            continue

        cigar = fields[5]
        contig_stats[contig]["indel_sum"] += _parse_cigar_indel_bp(cigar)
        contig_stats[contig]["count"] += 1
        contig_stats[contig]["query_len_sum"] += len(fields[9])

        # Parse AS tag
        for tag in fields[11:]:
            if tag.startswith("AS:i:"):
                contig_stats[contig]["as_sum"] += int(tag[5:])
                break

    # Compute means and pick best: min indel, then max normalized AS, then AS
    metrics: dict[str, dict] = {}
    best_contig = cluster_contigs[0]
    best_key: tuple[float, float, float] | None = None

    for contig, stats in contig_stats.items():
        n = stats["count"]
        if n == 0:
            continue
        mean_as = stats["as_sum"] / n
        mean_indel = stats["indel_sum"] / n
        mean_qlen = stats["query_len_sum"] / n
        mean_as_norm = mean_as / mean_qlen if mean_qlen > 0 else 0.0
        metrics[contig] = {
            "mean_as": round(mean_as, 1),
            "mean_as_norm": round(mean_as_norm, 4),
            "mean_indel_bp": round(mean_indel, 1),
            "reads": n,
        }
        # Sort key: lower indel better; higher norm AS better; higher AS better
        key = (mean_indel, -mean_as_norm, -mean_as)
        if best_key is None or key < best_key:
            best_key = key
            best_contig = contig

    logger.debug(
        "Refined peak contig: %s (indel=%.1f)",
        best_contig,
        metrics.get(best_contig, {}).get("mean_indel_bp", -1),
    )
    return {"best_contig": best_contig, "metrics": metrics}


def _find_clusters(
    counts: dict[int, int],
    min_coverage: int,
    min_gap: int = 5,
) -> list[dict]:
    """Identify read-count clusters in the contig distribution.

    Groups contigs that are within ``min_gap`` of each other into clusters,
    then computes the weighted center and total reads for each.

    Args:
        counts: Canonical repeat count -> mapped reads mapping.
        min_coverage: Minimum reads for a contig to be included.
        min_gap: Minimum gap between contigs to start a new cluster.

    Returns:
        List of cluster dicts sorted by total_reads descending.
        Each dict has keys: center (int), total_reads (int),
        contigs (list of (repeat_count, reads) tuples).
    """
    passing = sorted(
        [(k, v) for k, v in counts.items() if v >= min_coverage],
        key=lambda x: x[0],
    )

    if not passing:
        return []

    # Group into clusters by proximity
    clusters: list[list[tuple[int, int]]] = []
    current_cluster: list[tuple[int, int]] = [passing[0]]

    for i in range(1, len(passing)):
        if passing[i][0] - passing[i - 1][0] >= min_gap:
            clusters.append(current_cluster)
            current_cluster = [passing[i]]
        else:
            current_cluster.append(passing[i])
    clusters.append(current_cluster)

    # Compute weighted center and total reads for each cluster
    result: list[dict] = []
    for cluster in clusters:
        total_reads = sum(reads for _, reads in cluster)
        weighted_center = sum(pos * reads for pos, reads in cluster) / total_reads
        result.append(
            {
                "center": round(weighted_center),
                "total_reads": total_reads,
                "contigs": cluster,
            }
        )

    result.sort(key=lambda x: x["total_reads"], reverse=True)
    return result


def _cluster_span(cluster: dict) -> int:
    """Return the contig-index span of a cluster (max - min)."""
    contigs = [c for c, _ in cluster["contigs"]]
    if not contigs:
        return 0
    return contigs[-1] - contigs[0]


def _make_cluster_from_contigs(contigs: list[tuple[int, int]], center: int | None = None) -> dict:
    """Build a cluster dict from (repeat_count, reads) pairs."""
    total = sum(r for _, r in contigs)
    if center is None:
        center = round(sum(c * r for c, r in contigs) / total) if total else 0
    return {"center": center, "total_reads": total, "contigs": contigs}


def _compute_prominence(items: list[tuple[int, int]], peak_idx: int) -> float:
    """Topographic prominence of ``items[peak_idx]`` in a sorted series."""
    peak_reads = items[peak_idx][1]

    def _side_prominence(indices: range) -> float:
        min_along = peak_reads
        for j in indices:
            min_along = min(min_along, items[j][1])
            if items[j][1] > peak_reads:
                return float(peak_reads - min_along)
        side_min = min(items[j][1] for j in indices) if indices else peak_reads
        return float(peak_reads - side_min)

    left = _side_prominence(range(peak_idx - 1, -1, -1))
    right = _side_prominence(range(peak_idx + 1, len(items)))
    return min(left, right)


def _valley_reads_between(counts: dict[int, int], pos_a: int, pos_b: int) -> int:
    """Minimum read count strictly between two contig positions."""
    lo, hi = sorted((pos_a, pos_b))
    between = [counts[c] for c in counts if lo < c < hi]
    return min(between) if between else 0


def _find_peaks_by_prominence(
    counts: dict[int, int],
    min_coverage: int,
    min_separation: int = MIN_PEAK_SEPARATION,
    min_height_fraction: float = MIN_SECONDARY_HEIGHT_FRACTION,
) -> list[dict]:
    """Find up to two allele clusters from read-count local maxima.

    Primary peak = highest prominence local maximum.  Secondary peak must be
    at least ``min_separation`` contigs away, reach ``min_height_fraction`` of
    the primary height, and sit behind a clear valley (valley below half the
    secondary height).  Otherwise a single cluster is returned (same-length).

    Each cluster is a tight window around the peak contig, not a ladder-wide
    smear.
    """
    items = sorted((k, v) for k, v in counts.items() if v >= min_coverage)
    if not items:
        return []

    peaks: list[dict] = []
    for i, (pos, reads) in enumerate(items):
        left = items[i - 1][1] if i > 0 else -1
        right = items[i + 1][1] if i + 1 < len(items) else -1
        if reads >= left and reads >= right:
            peaks.append(
                {
                    "pos": pos,
                    "reads": reads,
                    "prominence": _compute_prominence(items, i),
                }
            )

    if not peaks:
        # Flat series: fall back to global maximum
        pos, reads = max(items, key=lambda x: x[1])
        peaks = [{"pos": pos, "reads": reads, "prominence": float(reads)}]

    peaks.sort(key=lambda p: (-p["prominence"], -p["reads"]))
    primary = peaks[0]

    def _accept(cand: dict, height_fraction: float) -> bool:
        if abs(cand["pos"] - primary["pos"]) < min_separation:
            return False
        if cand["reads"] < height_fraction * primary["reads"]:
            return False
        valley = _valley_reads_between(counts, primary["pos"], cand["pos"])
        # Reject shoulders: valley must drop below half the secondary height
        return valley <= 0.5 * cand["reads"]

    # Only accept a longer secondary allele.  Short-contig noise ridges are
    # ubiquitous on ONT amplicons (PCR / partial products) and must not become
    # allele_2.  If no longer peak qualifies, report same-length.
    secondary: dict | None = None
    for cand in peaks[1:]:
        if cand["pos"] < primary["pos"] + min_separation:
            continue
        if _accept(cand, min_height_fraction):
            secondary = cand
            break

    def _window_cluster(peak_pos: int) -> dict:
        half = PEAK_CLUSTER_HALF_WIDTH
        if secondary is not None:
            half = max(2, min(half, abs(secondary["pos"] - primary["pos"]) // 2))
        contigs = sorted(
            (c, counts[c])
            for c in counts
            if abs(c - peak_pos) <= half and counts[c] >= min_coverage
        )
        if not contigs:
            contigs = [(peak_pos, counts.get(peak_pos, 0))]
        return _make_cluster_from_contigs(contigs, center=peak_pos)

    clusters = [_window_cluster(primary["pos"])]
    if secondary is not None:
        clusters.append(_window_cluster(secondary["pos"]))
        clusters.sort(key=lambda x: x["total_reads"], reverse=True)
        logger.info(
            "Prominence peaks: contig_%d (%d reads) and contig_%d (%d reads)",
            primary["pos"],
            primary["reads"],
            secondary["pos"],
            secondary["reads"],
        )
    else:
        logger.info(
            "Prominence peaks: single peak contig_%d (%d reads); treating as same-length",
            primary["pos"],
            primary["reads"],
        )

    return clusters


def _split_cluster_by_indel(
    bam_path: Path,
    cluster: dict,
    min_valley_coverage_fraction: float = MIN_VALLEY_COVERAGE_FRACTION,
) -> list[dict] | None:
    """Attempt to split a single cluster into two alleles using indel valleys.

    Reads from a short allele mapped to a long contig (or vice versa)
    accumulate large indels in CIGAR.  Reads mapped to the correct-length
    contig have near-zero indels.  By finding the two local minima in
    per-contig mean indel length, we can resolve close alleles that
    gap-based clustering merges into one cluster.

    Valleys are ignored when their contig has fewer than
    ``min_valley_coverage_fraction`` of the cluster's busiest contig.  This
    blocks ONT long-tail false valleys that otherwise invent a ~150-repeat
    second allele.

    Returns two sub-clusters if a clear split is found, or None if the
    cluster is genuinely homozygous (single indel valley).
    """
    contig_names = [f"contig_{c}" for c, _ in cluster["contigs"]]

    # Compute per-contig mean indel bp
    contig_stats: dict[int, dict] = {}
    for c, _ in cluster["contigs"]:
        contig_stats[c] = {"indel_sum": 0, "count": 0}

    for line in run_tool_iter(["samtools", "view", str(bam_path), *contig_names]):
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        fields = line.split("\t")
        if len(fields) < 6:
            continue
        contig_name = fields[2]
        m = re.search(r"_(\d+)$", contig_name)
        if not m:
            continue
        c = int(m.group(1))
        if c not in contig_stats:
            continue
        contig_stats[c]["indel_sum"] += _parse_cigar_indel_bp(fields[5])
        contig_stats[c]["count"] += 1

    # Build mean-indel series (only contigs with reads)
    indel_series: list[tuple[int, float, int]] = []
    for c in sorted(contig_stats):
        n = contig_stats[c]["count"]
        if n == 0:
            continue
        indel_series.append((c, contig_stats[c]["indel_sum"] / n, n))

    if len(indel_series) < 3:
        return None

    max_reads = max(n for _, _, n in indel_series)
    min_valley_reads = max(1, int(min_valley_coverage_fraction * max_reads))

    # Find local minima (valleys) in mean indel, coverage-filtered
    valleys: list[tuple[int, float, int]] = []
    for i in range(len(indel_series)):
        c, val, n = indel_series[i]
        if n < min_valley_reads:
            continue
        left = indel_series[i - 1][1] if i > 0 else float("inf")
        right = indel_series[i + 1][1] if i < len(indel_series) - 1 else float("inf")
        if val <= left and val <= right:
            valleys.append((c, val, n))

    logger.debug(
        "Indel valley splitting: %d coverage-qualified valleys (min reads %d)",
        len(valleys),
        min_valley_reads,
    )
    if len(valleys) < 2:
        return None

    # Take the two deepest valleys (lowest mean indel)
    valleys.sort(key=lambda x: x[1])
    best_two = sorted(valleys[:2], key=lambda x: x[0])
    v1, v2 = best_two[0][0], best_two[1][0]

    # The split point is the midpoint between the two valleys
    split = (v1 + v2) // 2

    # Verify the valleys are meaningfully separated (at least 3 contigs apart)
    if abs(v2 - v1) < 3:
        return None

    # Split cluster contigs into two sub-clusters
    contigs_dict = dict(cluster["contigs"])
    sub1 = [(c, contigs_dict[c]) for c in sorted(contigs_dict) if c <= split]
    sub2 = [(c, contigs_dict[c]) for c in sorted(contigs_dict) if c > split]

    if not sub1 or not sub2:
        return None

    return [_make_cluster_from_contigs(sub1), _make_cluster_from_contigs(sub2)]


def _build_allele_info(cluster: dict, best_contig: str | None = None) -> dict:
    """Build allele info dict from a cluster.

    ``contig_name`` always matches ``canonical_repeats`` so remapping and
    Clair3 use the same ladder length that the report advertises.

    Args:
        cluster: Cluster dict from _find_clusters.
        best_contig: Contig name selected by refine_peak_contig.
            If None, falls back to the weighted center.
            Trusted only when within ±1 of the cluster center; otherwise
            discarded (ONT AS can peak far from the true length).
    """
    canonical = cluster["center"]

    # When refine_peak_contig has identified a specific best contig,
    # adopt it only if it agrees with the cluster center within ±1.
    # ONT AS can prefer a much longer contig; remapping to that contig
    # while reporting the center length produces wrong-length consensus.
    if best_contig is not None:
        match = re.search(r"_(\d+)$", best_contig)
        if match:
            refined_canonical = int(match.group(1))
            if abs(refined_canonical - canonical) <= 1:
                canonical = refined_canonical
            else:
                logger.warning(
                    "Discarding AS-refined %s (differs from cluster center "
                    "%d by >1); remapping will use contig_%d",
                    best_contig,
                    canonical,
                    canonical,
                )

    contig_name = f"contig_{canonical}"

    return {
        "length": canonical + PRE_AFTER_REPEAT_COUNT,
        "reads": cluster["total_reads"],
        "canonical_repeats": canonical,
        "contig_name": contig_name,
        "cluster_contigs": [f"contig_{c}" for c, _ in cluster["contigs"]],
    }


def detect_alleles(
    counts: dict[int, int],
    min_coverage: int = 10,
    bam_path: Path | None = None,
) -> dict:
    """Detect allele lengths from read count distribution across ladder contigs.

    Strategy (in order):

    1. Gap-cluster contigs with ``>= min_coverage`` reads.
    2. If clusters are narrow and separated (typical HiFi), use them.
    3. If a single cluster is narrow and a BAM is available, try
       coverage-weighted indel-valley splitting (close alleles).
    4. If the ladder is a wide ONT-like smear (span > ``MAX_CLUSTER_SPAN``),
       switch to read-count prominence peaks with a secondary-height and
       valley-depth filter.  No credible second peak → same-length.

    If *bam_path* is provided, the best contig within each cluster is refined
    using alignment scores (AS), trusted only within ±1 of the cluster center.

    Args:
        counts: Canonical repeat count -> mapped reads mapping from
            :func:`parse_idxstats`.
        min_coverage: Minimum mapped reads to include a contig.
        bam_path: Optional path to the indexed ladder mapping BAM.
            When provided, enables alignment-quality-based peak refinement.

    Returns:
        Dictionary with keys ``allele_1``, ``allele_2``, and ``homozygous``.
        Each allele has:

        - ``length`` (int): total repeat units including pre/after
        - ``reads`` (int): total mapped reads across the cluster
        - ``canonical_repeats`` (int): number of canonical X repeats
        - ``contig_name`` (str): best-matching contig (e.g. ``"contig_51"``)
        - ``cluster_contigs`` (list[str]): all contig names in the cluster

    Raises:
        ValueError: If no contig meets the minimum coverage threshold.
    """
    # Handle mixed-key dicts (legacy compat): only use integer keys
    int_counts = {k: v for k, v in counts.items() if isinstance(k, int)}

    clusters = _find_clusters(int_counts, min_coverage)
    logger.info("Detected %d gap-cluster(s) from %d contigs", len(clusters), len(int_counts))

    if not clusters:
        max_observed = max(int_counts.values()) if int_counts else 0
        raise ValueError(
            f"No contig has >= {min_coverage} mapped reads (minimum coverage). "
            f"Max observed: {max_observed} reads."
        )

    wide_smear = any(_cluster_span(c) > MAX_CLUSTER_SPAN for c in clusters)

    if wide_smear:
        # ONT amplicon: residual coverage across the whole ladder.  Do not
        # trust indel-valley splits (they invent ~150-repeat alleles).
        logger.info(
            "Wide ladder smear detected (span > %d); using prominence peaks",
            MAX_CLUSTER_SPAN,
        )
        clusters = _find_peaks_by_prominence(int_counts, min_coverage)
    elif len(clusters) == 1 and bam_path is not None:
        # Narrow single cluster: try coverage-weighted indel valleys (HiFi
        # close alleles).  Fall back to prominence if valleys are weak.
        sub_clusters = _split_cluster_by_indel(bam_path, clusters[0])
        if sub_clusters is not None:
            clusters = sub_clusters
            clusters.sort(key=lambda x: x["total_reads"], reverse=True)
        elif _cluster_span(clusters[0]) >= MIN_PEAK_SEPARATION:
            peak_clusters = _find_peaks_by_prominence(int_counts, min_coverage)
            if len(peak_clusters) >= 2:
                clusters = peak_clusters

    if not clusters:
        raise ValueError("Allele detection produced no clusters.")

    def _get_best_contig(cluster: dict) -> str | None:
        if bam_path is None:
            return None
        contig_names = [f"contig_{c}" for c, _ in cluster["contigs"]]
        refined = refine_peak_contig(bam_path, contig_names)
        best: str = refined["best_contig"]
        return best

    allele_1 = _build_allele_info(clusters[0], _get_best_contig(clusters[0]))

    if len(clusters) < 2:
        return {
            "allele_1": allele_1,
            "allele_2": {**allele_1},
            "homozygous": False,
            "same_length": True,
        }

    allele_2 = _build_allele_info(clusters[1], _get_best_contig(clusters[1]))

    # Drop a secondary allele whose peak support is negligible vs allele_1
    # (applies when gap clustering left a tiny long-tail remnant).
    if allele_1["reads"] > 0 and allele_2["reads"] < MIN_SECONDARY_HEIGHT_FRACTION * allele_1["reads"]:
        logger.info(
            "Dropping allele_2 (%d reads < %.0f%% of allele_1); same-length",
            allele_2["reads"],
            100 * MIN_SECONDARY_HEIGHT_FRACTION,
        )
        return {
            "allele_1": allele_1,
            "allele_2": {**allele_1},
            "homozygous": False,
            "same_length": True,
        }

    if allele_1["length"] == allele_2["length"]:
        allele_1["reads"] += allele_2["reads"]
        return {
            "allele_1": allele_1,
            "allele_2": {**allele_1},
            "homozygous": False,
            "same_length": True,
        }

    return {
        "allele_1": allele_1,
        "allele_2": allele_2,
        "homozygous": False,
        "same_length": False,
    }
