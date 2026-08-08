"""
Fit length-aware beta-binomial family coefficients from sequence FASTA records.
"""
from __future__ import annotations

import math
import os
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln
from tqdm.auto import tqdm

from ._utils import LOG, is_progress_disabled
from .alphabets import get_transform
from .merge_search import parse_seq
from .sequence_reader import SequenceReader


BBINOM_COEFFICIENT_BOUND = 50.0
BBINOM_COEFFICIENT_BOUND_TOLERANCE = 1e-3


@numba.njit(nogil=True)
def count_selected_family_hits(codes, table_idx, table_buff, hog_to_family, target_rank, counts, touched):
    n_touched = 0
    for i in range(codes.size):
        code = codes[i]
        start = table_idx[code]
        end = table_idx[code + 1]
        for pos in range(start, end):
            family = hog_to_family[table_buff[pos]]
            rank = target_rank[family]
            if rank >= 0:
                if counts[rank] == 0:
                    touched[n_touched] = rank
                    n_touched += 1
                counts[rank] += 1
    return n_touched


def _modality_index_arrays(db, modality):
    """Return (table_index, table_buffer) for the requested modality."""
    if modality == "ss":
        if not db.has_structure():
            raise ValueError(
                "Database has no structure index (/Index/SSTableIndex); "
                "cannot fit ss beta-binomial coefficients"
            )
        return db._db_Index_SSTableIndex, db._db_Index_SSTableBuffer
    if modality == "seq":
        return db._db_Index_TableIndex, db._db_Index_TableBuffer
    raise ValueError("modality must be 'seq' or 'ss', got {!r}".format(modality))


def _modality_family_probability(db, modality):
    if modality == "ss":
        if not db.has_structure():
            raise ValueError(
                "Database has no structure index; cannot select ss families by probability"
            )
        return db._db_Index_SSFamilyProbability[:]
    return db._db_Index_FamilyProbability[:]


def _modality_kmer_max_df(db, modality):
    """Return the build-time information-filter cutoff for a modality."""
    if modality == "ss":
        return int(db.ki.ss_kmer_max_df)
    if modality == "seq":
        return int(db.ki.kmer_max_df)
    raise ValueError("modality must be 'seq' or 'ss', got {!r}".format(modality))


def _iter_sequences(paths, k, chunksize, sanitiser):
    record_i = 0
    for path in paths:
        for ids, seqs in SequenceReader.read(path, k=k, chunksize=chunksize, sanitiser=sanitiser):
            for seq_id, seq in zip(ids, seqs):
                yield record_i, seq_id, seq
                record_i += 1


def _unique_valid_codes(
    seq,
    k,
    trans,
    digits_lookup,
    x_flag,
    table_idx=None,
    kmer_filter_max_df=0,
):
    n_kmers = len(seq) - (k - 1)
    if n_kmers <= 0:
        return np.empty(0, dtype=np.uint32)
    if isinstance(seq, str):
        seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    elif isinstance(seq, bytes):
        seq_arr = np.frombuffer(seq, dtype=np.uint8)
    else:
        seq_arr = np.asarray(seq).view(np.uint8)
    codes, _, _ = parse_seq(seq_arr, digits_lookup, n_kmers, k, trans, x_flag)
    codes = codes[codes != x_flag]
    if kmer_filter_max_df > 0:
        dfs = table_idx[codes + 1] - table_idx[codes]
        codes = codes[dfs <= kmer_filter_max_df]
    return codes


def _buffer_record_bounds(sequence_buffer, expected_records, bounds=None):
    """Return sequence starts/stops for a space-delimited mkdb buffer.

    ``bounds`` short-circuits the delimiter scan when a caller already holds
    the offsets, e.g. from a training-buffer sidecar. The scan itself needs a
    transient boolean mask the size of the whole buffer, which is worth
    avoiding on LUCA-scale builds.
    """
    if bounds is not None:
        starts, stops = bounds
        starts = np.asarray(starts, dtype=np.int64)
        stops = np.asarray(stops, dtype=np.int64)
        if starts.size != int(expected_records):
            raise ValueError(
                "Supplied record bounds hold {} records, expected {}".format(
                    starts.size, expected_records
                )
            )
        return starts, stops
    byte_view = np.asarray(sequence_buffer).view(np.uint8)
    stops = np.flatnonzero(byte_view == ord(" ")).astype(np.int64)
    if stops.size != int(expected_records):
        raise ValueError(
            "Sequence buffer contains {} records, expected {} proteins".format(
                stops.size,
                expected_records,
            )
        )
    starts = np.empty(stops.size, dtype=np.int64)
    if starts.size:
        starts[0] = 0
        starts[1:] = stops[:-1] + 1
    return starts, stops


def scan_buffer_unique_counts(
    sequence_buffer,
    expected_records,
    k,
    alphabet,
    table_index,
    kmer_filter_max_df=0,
    bounds=None,
):
    """Count exact unique k-mers directly from an mkdb sequence buffer."""
    starts, stops = _buffer_record_bounds(sequence_buffer, expected_records, bounds)
    transform = get_transform(k, alphabet.DIGITS_AA)
    x_flag = table_index.size - 1
    counts = np.empty(starts.size, dtype=np.uint32)
    for record in tqdm(
        range(starts.size),
        desc="scan exact unique-kmer counts",
        disable=is_progress_disabled(),
    ):
        counts[record] = _unique_valid_codes(
            sequence_buffer[starts[record] : stops[record]],
            k,
            transform,
            alphabet.DIGITS_AA_LOOKUP,
            x_flag,
            table_index,
            kmer_filter_max_df,
        ).size
    return counts


def load_sampled_buffer_sequences(
    sequence_buffer,
    expected_records,
    sampled_by_n,
    bounds=None,
):
    """Materialize only sampled records from an mkdb sequence buffer."""
    starts, stops = _buffer_record_bounds(sequence_buffer, expected_records, bounds)
    selected = {
        int(record): int(n)
        for n, indices in sampled_by_n.items()
        for record in indices
    }
    sampled_sequences = []
    sampled_indices = []
    for record in sorted(selected):
        sampled_sequences.append(
            (
                selected[record],
                np.asarray(
                    sequence_buffer[starts[record] : stops[record]]
                ).tobytes(),
            )
        )
        sampled_indices.append(record)
    return sampled_sequences, np.asarray(sampled_indices, dtype=np.int64)


def scan_sequence_unique_counts(
    paths,
    k,
    alphabet,
    table_idx,
    chunksize,
    kmer_filter_max_df=0,
):
    trans = get_transform(k, alphabet.DIGITS_AA)
    digits_lookup = alphabet.DIGITS_AA_LOOKUP
    x_flag = table_idx.size - 1
    counts = array("I")
    for _, _, seq in tqdm(
        _iter_sequences(paths, k, chunksize, alphabet.sanitise_seq),
        desc="scan exact unique-kmer counts",
        disable=is_progress_disabled(),
    ):
        counts.append(
            int(
                _unique_valid_codes(
                    seq,
                    k,
                    trans,
                    digits_lookup,
                    x_flag,
                    table_idx,
                    kmer_filter_max_df,
                ).size
            )
        )
    return np.frombuffer(counts, dtype=np.uint32).copy()


def parse_n_values(value):
    if not value:
        return None
    return [int(x) for x in str(value).replace(",", " ").split()]


def choose_sequence_n_values(n_unique, min_records_per_n=50, n_buckets=24, requested_n_values=None):
    hist = np.bincount(np.asarray(n_unique, dtype=np.int64))
    eligible = np.flatnonzero(hist >= int(min_records_per_n))

    # N=0 has no k-mer trials; log(0) is undefined
    eligible = eligible[eligible > 0]

    if eligible.size == 0:
        raise RuntimeError("No exact N value has enough records for fitting")

    chosen = []
    if requested_n_values:
        for n in requested_n_values:
            if n < 0 or n >= hist.size:
                raise ValueError(f"Requested N={n} is outside the observed N range")
            if hist[n] < int(min_records_per_n):
                raise ValueError(f"Requested N={n} has {int(hist[n])} records; min_records_per_n={min_records_per_n}")
            chosen.append(int(n))
    elif int(n_buckets) <= 0 or eligible.size <= int(n_buckets):
        chosen = [int(n) for n in eligible]
    else:
        targets = np.linspace(int(eligible.min()), int(eligible.max()), int(n_buckets))
        for target in targets:
            remaining = np.setdiff1d(eligible, np.asarray(chosen, dtype=np.int64), assume_unique=False)
            if remaining.size == 0:
                break
            distance = np.abs(remaining - target)
            tied = remaining[distance == distance.min()]
            chosen.append(int(tied[np.argmax(hist[tied])]))

    chosen = np.asarray(sorted(set(chosen)), dtype=np.uint32)
    if chosen.size < 2:
        raise RuntimeError("Need at least two exact N values to fit a length-aware beta-binomial model")

    summary = pd.DataFrame(
        {
            "n_unique": np.arange(hist.size, dtype=np.int64),
            "available": hist,
            "eligible": hist >= int(min_records_per_n),
            "selected": np.isin(np.arange(hist.size, dtype=np.int64), chosen),
        }
    )
    return summary, chosen


def sample_record_indices_by_n(n_unique, selected_n, max_records_per_n, seed):
    rng = np.random.default_rng(seed)
    sampled = {}
    for n in selected_n:
        candidates = np.flatnonzero(n_unique == int(n))
        if int(max_records_per_n) > 0:
            take = min(int(max_records_per_n), int(candidates.size))
        else:
            take = int(candidates.size)
        if take == candidates.size:
            sampled[int(n)] = candidates.astype(np.int64, copy=False)
        else:
            sampled[int(n)] = np.sort(rng.choice(candidates, size=take, replace=False))
    return sampled


def load_sampled_sequences(paths, sampled_by_n, k, chunksize, sanitiser):
    """Load only sampled records so family batches do not rescan the FASTA."""
    selected_lookup = {
        int(record_i): int(n)
        for n, indices in sampled_by_n.items()
        for record_i in indices
    }
    sampled = []
    for record_i, _, seq in tqdm(
        _iter_sequences(paths, k, chunksize, sanitiser),
        desc="load sampled sequences",
        disable=is_progress_disabled(),
    ):
        n = selected_lookup.get(record_i)
        if n is not None:
            sampled.append((n, seq))
    if len(sampled) != len(selected_lookup):
        raise RuntimeError(
            "Loaded {} of {} sampled FASTA records".format(
                len(sampled),
                len(selected_lookup),
            )
        )
    return sampled


def read_family_offsets(path):
    values = []
    with open(path, "r") as handle:
        first = handle.readline()
        if not first:
            return np.empty(0, dtype=np.int64)
        fields = first.rstrip("\n").split("\t")
        if "family_offset" in fields:
            family_i = fields.index("family_offset")
            for line in handle:
                if line.strip():
                    values.append(int(line.rstrip("\n").split("\t")[family_i]))
        else:
            if first.strip() and not first.lstrip().startswith("#"):
                values.append(int(first.split()[0]))
            for line in handle:
                if line.strip() and not line.lstrip().startswith("#"):
                    values.append(int(line.split()[0]))
    return np.asarray(values, dtype=np.int64)


def select_family_offsets(db, family_offsets_path=None, min_family_prob=0.0, max_families=0, seed=42, modality="seq"):
    n_families = db.family_table.nrows
    if family_offsets_path:
        families = read_family_offsets(family_offsets_path)
    else:
        prob = _modality_family_probability(db, modality)
        families = np.flatnonzero(prob >= float(min_family_prob)).astype(np.int64)

    families = np.unique(families)
    if np.any((families < 0) | (families >= n_families)):
        raise ValueError("family offsets must be in DB range 0..{}".format(n_families - 1))

    if int(max_families) > 0 and families.size > int(max_families):
        rng = np.random.default_rng(seed)
        families = np.sort(rng.choice(families, size=int(max_families), replace=False))
    return families.astype(np.int64, copy=False)


def parse_family_shard(value):
    """Parse an ``i/n`` family shard selector into a zero-based (i, n)."""
    if value is None:
        return None
    text = str(value).strip()
    if "/" not in text:
        raise ValueError(
            "family shard must look like 'i/n', got {!r}".format(value)
        )
    index_text, count_text = text.split("/", 1)
    index, count = int(index_text), int(count_text)
    if count < 1:
        raise ValueError("family shard count must be >= 1")
    if not 0 <= index < count:
        raise ValueError(
            "family shard index {} outside 0..{}".format(index, count - 1)
        )
    return index, count


def apply_family_shard(families, shard):
    """Return the contiguous slice of ``families`` belonging to one shard.

    Sharding is by family, never by null query: every shard samples the same
    background records from the same seed, so a sharded run and a whole run
    produce identical coefficients. Contiguous slices keep each shard's
    histogram rows adjacent, which is what bounds its memory.
    """
    if shard is None:
        return families
    index, count = shard
    edges = np.linspace(0, families.size, count + 1).astype(np.int64)
    selected = families[edges[index] : edges[index + 1]]
    LOG.info(
        "Family shard %d/%d: %d of %d families (offsets %s..%s)",
        index + 1,
        count,
        selected.size,
        families.size,
        selected[0] if selected.size else "-",
        selected[-1] if selected.size else "-",
    )
    return selected


def load_or_scan_unique_counts(
    sequence_buffer,
    n_records,
    k,
    alphabet,
    table_index,
    kmer_filter_max_df,
    bounds=None,
    cache_path=None,
):
    """Return per-record exact unique-k-mer counts, caching the scan.

    The counts depend on the k-mer filter, so a cache written for one
    ``kmer_percentage`` must not be reused for another. The cache records the
    cutoff it was built with and refuses a mismatch rather than silently
    fitting the wrong N grid.
    """
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path) as cached:
            cached_max_df = int(cached["kmer_filter_max_df"])
            cached_records = int(cached["n_records"])
            counts = cached["counts"]
        if cached_max_df != int(kmer_filter_max_df):
            raise ValueError(
                "{} was built with kmer_filter_max_df={} but this fit uses "
                "{}; delete it or use a per-filter cache path".format(
                    cache_path, cached_max_df, kmer_filter_max_df
                )
            )
        if cached_records != int(n_records):
            raise ValueError(
                "{} holds {} records but the database has {}".format(
                    cache_path, cached_records, n_records
                )
            )
        LOG.info("Loaded %d exact-N values from %s", counts.size, cache_path)
        return counts

    counts = scan_buffer_unique_counts(
        sequence_buffer,
        n_records,
        k,
        alphabet,
        table_index,
        kmer_filter_max_df,
        bounds=bounds,
    )
    if cache_path:
        # Write through a handle: np.savez would otherwise append ".npz" and
        # the existence check above would never find its own cache. Write to a
        # unique temporary and rename, so that shards of one sharded fit racing
        # on the same path cannot leave a half-written cache behind.
        temp_path = "{}.{}.tmp".format(cache_path, os.getpid())
        with open(temp_path, "wb") as handle:
            np.savez(
                handle,
                counts=counts,
                kmer_filter_max_df=np.int64(kmer_filter_max_df),
                n_records=np.int64(n_records),
            )
        os.replace(temp_path, cache_path)
        LOG.info("Cached %d exact-N values in %s", counts.size, cache_path)
    return counts


def collect_family_hit_histograms(
    paths,
    sampled_by_n,
    selected_families,
    db,
    chunksize=10000,
    modality="seq",
    kmer_filter_max_df=0,
    table_idx=None,
):
    selected_lookup = {int(idx): int(n) for n, indices in sampled_by_n.items() for idx in indices}
    selected_indices = set(selected_lookup)
    selected_n = set(int(n) for n in sampled_by_n)
    n_query_counts = Counter()
    hist = defaultdict(int)

    k = db.ki.k
    alphabet = db.ki.alphabet
    trans = get_transform(k, alphabet.DIGITS_AA)
    digits_lookup = alphabet.DIGITS_AA_LOOKUP
    table_index_node, table_buffer_node = _modality_index_arrays(db, modality)
    if table_idx is None:
        table_idx = table_index_node[:]
    table_buff = table_buffer_node[:]
    x_flag = table_idx.size - 1
    hog_to_family = db.hog_table.col("FamOff")

    target_rank = np.full(db.family_table.nrows, -1, dtype=np.int32)
    target_rank[selected_families] = np.arange(selected_families.size, dtype=np.int32)
    counts = np.zeros(selected_families.size, dtype=np.uint32)
    touched = np.empty(selected_families.size, dtype=np.int32)

    for record_i, _, seq in tqdm(
        _iter_sequences(paths, k, chunksize, alphabet.sanitise_seq),
        desc="count family hits",
        disable=is_progress_disabled(),
    ):
        if record_i not in selected_indices:
            continue
        n = selected_lookup[record_i]
        codes = _unique_valid_codes(
            seq,
            k,
            trans,
            digits_lookup,
            x_flag,
            table_idx,
            kmer_filter_max_df,
        )
        if int(codes.size) != n or n not in selected_n:
            continue
        n_query_counts[n] += 1
        n_touched = count_selected_family_hits(
            codes,
            table_idx,
            table_buff,
            hog_to_family,
            target_rank,
            counts,
            touched,
        )
        for j in range(n_touched):
            rank = int(touched[j])
            x = int(counts[rank])
            hist[(rank, n, x)] += 1
            counts[rank] = 0

    return hist, dict(n_query_counts)


@numba.njit(nogil=True)
def update_dense_family_histogram(
    counts,
    touched,
    n_touched,
    n_offset,
    histogram,
):
    """Record one query's nonzero family counts and reset scratch counts."""
    for j in range(n_touched):
        rank = touched[j]
        x = counts[rank]
        histogram[rank, n_offset + x] += 1
        counts[rank] = 0


class DenseFamilyHitHistogram:
    """Compact per-family histograms with ragged X ranges flattened by N."""

    def __init__(self, data, n_values, offsets, excluded_query_counts=None):
        self.data = data
        self.n_values = np.asarray(n_values, dtype=np.uint32)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        if excluded_query_counts is None:
            excluded_query_counts = np.zeros(
                (self.data.shape[0], self.n_values.size),
                dtype=np.uint32,
            )
        self.excluded_query_counts = np.asarray(
            excluded_query_counts,
            dtype=np.uint32,
        )

    @property
    def n_families(self):
        return self.data.shape[0]

    def records(self, rank):
        records = []
        row = self.data[int(rank)]
        for n_i, n in enumerate(self.n_values):
            start = int(self.offsets[n_i])
            values = row[start + 1 : start + int(n) + 1]
            for x_i in np.flatnonzero(values):
                records.append(
                    (
                        int(n),
                        int(x_i) + 1,
                        int(values[x_i]),
                    )
                )
        return records

    def query_counts(self, rank, total_query_counts):
        """Return per-N null counts after excluding own-family queries."""
        counts = {}
        for n_i, n in enumerate(self.n_values):
            count = int(total_query_counts[int(n)]) - int(
                self.excluded_query_counts[int(rank), n_i]
            )
            if count > 0:
                counts[int(n)] = count
        return counts


def collect_dense_family_hit_histograms(
    sampled_sequences,
    selected_families,
    db,
    modality="seq",
    kmer_filter_max_df=0,
    table_idx=None,
    table_buff=None,
    sampled_family_offsets=None,
):
    """Collect bounded-memory histograms without Python objects per bin.

    When ``sampled_family_offsets`` is supplied, it must contain one database
    family offset per sampled sequence (or -1 when unknown). The returned
    ``own_family_hits`` array records the hit count for that family before the
    scratch counts are reset. This permits an audit, or a subsequent fit, to
    remove positive-family queries from the null distribution without treating
    them as zero-hit observations.
    """
    n_query_counts = Counter(int(n) for n, _ in sampled_sequences)
    if sampled_family_offsets is not None:
        sampled_family_offsets = np.asarray(
            sampled_family_offsets,
            dtype=np.int64,
        )
        if sampled_family_offsets.shape != (len(sampled_sequences),):
            raise ValueError(
                "sampled_family_offsets must have one value per sampled "
                "sequence"
            )
        own_family_hits = np.zeros(
            len(sampled_sequences),
            dtype=np.uint32,
        )
    else:
        own_family_hits = None
    n_values = np.asarray(sorted(n_query_counts), dtype=np.uint32)
    offsets = np.zeros(n_values.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(n_values.astype(np.int64) + 1)
    offset_lookup = np.full(int(n_values.max()) + 1, -1, dtype=np.int64)
    offset_lookup[n_values] = offsets[:-1]
    n_rank_lookup = np.full(int(n_values.max()) + 1, -1, dtype=np.int32)
    n_rank_lookup[n_values] = np.arange(n_values.size, dtype=np.int32)
    max_queries_per_n = max(n_query_counts.values())
    hist_dtype = np.uint16 if max_queries_per_n <= np.iinfo(np.uint16).max else np.uint32
    histogram = np.zeros(
        (selected_families.size, int(offsets[-1])),
        dtype=hist_dtype,
    )
    excluded_query_counts = np.zeros(
        (selected_families.size, n_values.size),
        dtype=np.uint32,
    )

    k = db.ki.k
    alphabet = db.ki.alphabet
    trans = get_transform(k, alphabet.DIGITS_AA)
    digits_lookup = alphabet.DIGITS_AA_LOOKUP
    table_index_node, table_buffer_node = _modality_index_arrays(db, modality)
    if table_idx is None:
        table_idx = table_index_node[:]
    if table_buff is None:
        table_buff = table_buffer_node[:]
    x_flag = table_idx.size - 1
    hog_to_family = db.hog_table.col("FamOff")

    target_rank = np.full(db.family_table.nrows, -1, dtype=np.int32)
    target_rank[selected_families] = np.arange(
        selected_families.size,
        dtype=np.int32,
    )
    counts = np.zeros(selected_families.size, dtype=np.uint32)
    touched = np.empty(selected_families.size, dtype=np.int32)

    for sample_i, (n, seq) in enumerate(tqdm(
        sampled_sequences,
        desc="count family hits",
        disable=is_progress_disabled(),
    )):
        codes = _unique_valid_codes(
            seq,
            k,
            trans,
            digits_lookup,
            x_flag,
            table_idx,
            kmer_filter_max_df,
        )
        if int(codes.size) != int(n):
            raise RuntimeError(
                "Sampled FASTA record changed from N={} to N={}".format(
                    n,
                    codes.size,
                )
            )
        n_touched = count_selected_family_hits(
            codes,
            table_idx,
            table_buff,
            hog_to_family,
            target_rank,
            counts,
            touched,
        )
        own_rank = -1
        own_hit_count = 0
        if own_family_hits is not None:
            own_family = sampled_family_offsets[sample_i]
            if 0 <= own_family < target_rank.size:
                own_rank = target_rank[own_family]
                if own_rank >= 0:
                    own_hit_count = int(counts[own_rank])
                    own_family_hits[sample_i] = own_hit_count
        update_dense_family_histogram(
            counts,
            touched,
            n_touched,
            offset_lookup[int(n)],
            histogram,
        )
        if own_rank >= 0:
            excluded_query_counts[own_rank, n_rank_lookup[int(n)]] += 1
            if own_hit_count > 0:
                histogram[
                    own_rank,
                    offset_lookup[int(n)] + own_hit_count,
                ] -= 1

    result = (
        DenseFamilyHitHistogram(
            histogram,
            n_values,
            offsets,
            excluded_query_counts,
        ),
        dict(n_query_counts),
    )
    if own_family_hits is not None:
        return result + (own_family_hits,)
    return result


def _weighted_center_scale(ns, weights):
    log_n = np.log(ns.astype(float))
    total = float(weights.sum())
    center = float(np.sum(weights * log_n) / total)
    var = float(np.sum(weights * (log_n - center) ** 2) / total)
    scale = math.sqrt(var)
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return center, scale


def _design(n, degree, center, scale):
    z = (np.log(n.astype(float)) - center) / scale
    return np.column_stack([z**i for i in range(degree + 1)])


def _betabinom_logpmf(x, n, alpha, beta):
    return (
        gammaln(n + 1)
        - gammaln(x + 1)
        - gammaln(n - x + 1)
        + gammaln(x + alpha)
        + gammaln(n - x + beta)
        - gammaln(n + alpha + beta)
        + gammaln(alpha + beta)
        - gammaln(alpha)
        - gammaln(beta)
    )


def _bbinom_objective_and_gradient(
    theta,
    xq,
    xk,
    ns,
    xs,
    weights,
):
    """Weighted negative log-likelihood and exact coefficient gradient."""
    q_size = xq.shape[1]
    q_coef = theta[:q_size]
    kappa_coef = theta[q_size:]

    q_raw = expit(xq @ q_coef)
    q = np.clip(q_raw, 1e-8, 1 - 1e-8)
    log_kappa_raw = xk @ kappa_coef
    log_kappa = np.clip(
        log_kappa_raw,
        math.log(1e-4),
        math.log(1e8),
    )
    kappa = np.exp(log_kappa)
    alpha = q * kappa
    beta = (1.0 - q) * kappa
    logpmf = _betabinom_logpmf(xs, ns, alpha, beta)
    if not np.all(np.isfinite(logpmf)):
        return float("inf"), np.zeros_like(theta)

    dlog_dalpha = (
        digamma(xs + alpha)
        - digamma(ns + alpha + beta)
        + digamma(alpha + beta)
        - digamma(alpha)
    )
    dlog_dbeta = (
        digamma(ns - xs + beta)
        - digamma(ns + alpha + beta)
        + digamma(alpha + beta)
        - digamma(beta)
    )
    q_derivative = q * (1.0 - q)
    q_derivative[
        (q_raw <= 1e-8) | (q_raw >= 1.0 - 1e-8)
    ] = 0.0
    log_kappa_derivative = np.ones_like(log_kappa)
    log_kappa_derivative[
        (log_kappa_raw <= math.log(1e-4))
        | (log_kappa_raw >= math.log(1e8))
    ] = 0.0

    dlog_dq_linear = (
        (dlog_dalpha - dlog_dbeta)
        * kappa
        * q_derivative
    )
    dlog_dlog_kappa = (
        dlog_dalpha * q + dlog_dbeta * (1.0 - q)
    ) * kappa * log_kappa_derivative
    gradient = -np.concatenate(
        [
            xq.T @ (weights * dlog_dq_linear),
            xk.T @ (weights * dlog_dlog_kappa),
        ]
    )
    return -float(np.sum(weights * logpmf)), gradient


def fit_family_bbinom_from_hist(
    family_offset,
    records,
    n_query_counts,
    q_degree=2,
    kappa_degree=1,
    min_nonzero_queries=20,
    modality="seq",
    kmer_percentage=100.0,
):
    n_query_counts = {
        int(n): int(count)
        for n, count in n_query_counts.items()
        if int(count) > 0
    }
    if len(n_query_counts) < 2:
        return None
    nonzero_queries = int(sum(count for _, _, count in records))
    if nonzero_queries < int(min_nonzero_queries):
        return None

    rows = []
    by_n_nonzero = Counter()
    for n, x, count in records:
        rows.append((int(n), int(x), int(count)))
        by_n_nonzero[int(n)] += int(count)
    for n, total in n_query_counts.items():
        zero_count = int(total) - int(by_n_nonzero.get(int(n), 0))
        if zero_count > 0:
            rows.append((int(n), 0, zero_count))

    if not rows:
        return None

    ns = np.asarray([r[0] for r in rows], dtype=float)
    xs = np.asarray([r[1] for r in rows], dtype=float)
    weights = np.asarray([r[2] for r in rows], dtype=float)
    if weights.sum() <= 0 or np.sum(weights * xs) <= 0:
        return None

    center, scale = _weighted_center_scale(ns, weights)
    xq = _design(ns, int(q_degree), center, scale)
    xk = _design(ns, int(kappa_degree), center, scale)

    overall_q = float(np.clip(np.sum(weights * xs) / np.sum(weights * ns), 1e-8, 1 - 1e-8))
    q_init = np.zeros(int(q_degree) + 1, dtype=float)
    q_init[0] = math.log(overall_q / (1.0 - overall_q))
    kappa_init = np.zeros(int(kappa_degree) + 1, dtype=float)
    kappa_init[0] = math.log(1000.0)
    theta0 = np.concatenate([q_init, kappa_init])

    def objective(theta):
        return _bbinom_objective_and_gradient(
            theta,
            xq,
            xk,
            ns,
            xs,
            weights,
        )[0]

    def gradient(theta):
        return _bbinom_objective_and_gradient(
            theta,
            xq,
            xk,
            ns,
            xs,
            weights,
        )[1]

    bounds = [
        (-BBINOM_COEFFICIENT_BOUND, BBINOM_COEFFICIENT_BOUND)
    ] * theta0.size

    def optimise_retry(start):
        divisor = float(weights.sum())
        return minimize(
            lambda theta: objective(theta) / divisor,
            start,
            jac=lambda theta: gradient(theta) / divisor,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxls": 100},
        )

    # Preserve the historical fit for every family where it succeeds.  A
    # larger line-search budget and mean-NLL scaling are only used to recover
    # failures, so this change cannot perturb existing successful fits.
    result = minimize(
        objective,
        theta0,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
    )
    optimizer_attempt_count = 1
    optimizer_successful_attempt = "initial" if result.success else ""
    initial_result = result
    if not result.success:
        result = optimise_retry(theta0)
        optimizer_attempt_count += 1
        if result.success:
            optimizer_successful_attempt = "retry_scaled_maxls100"

    # If the same-start retry is still insufficient, try a small, fixed set of
    # initial dispersions and retain the converged solution with the greatest
    # (unscaled) likelihood.
    if not result.success:
        successful = []
        for kappa_start in (1.0, 10.0, 100.0, 1000.0, 3000.0):
            start = theta0.copy()
            start[int(q_degree) + 1] = math.log(kappa_start)
            attempt = optimise_retry(start)
            optimizer_attempt_count += 1
            attempt_theta = np.asarray(attempt.x, dtype=np.float64)
            if (
                attempt.success
                and np.all(np.isfinite(attempt_theta))
                and np.isfinite(objective(attempt_theta))
                and np.all(
                    np.abs(attempt_theta)
                    < BBINOM_COEFFICIENT_BOUND
                    - BBINOM_COEFFICIENT_BOUND_TOLERANCE
                )
            ):
                successful.append(
                    (objective(attempt_theta), kappa_start, attempt)
                )
        if successful:
            _, kappa_start, result = min(successful, key=lambda item: item[0])
            optimizer_successful_attempt = "multistart_kappa_{}".format(
                int(kappa_start)
            )
        else:
            # Keep the original failed result as the diagnostic result and the
            # historical initial coefficients as the explicit fallback.
            result = initial_result

    optimizer_success = bool(result.success)
    theta = np.asarray(
        result.x if optimizer_success else theta0,
        dtype=np.float64,
    )
    invalid_reasons = []
    if not optimizer_success:
        invalid_reasons.append("optimizer_failed")
    if not np.all(np.isfinite(theta)) or not np.isfinite(result.fun):
        invalid_reasons.append("nonfinite_fit")
    if np.any(
        np.abs(theta)
        >= BBINOM_COEFFICIENT_BOUND - BBINOM_COEFFICIENT_BOUND_TOLERANCE
    ):
        invalid_reasons.append("coefficient_at_bound")
    loglik = -objective(theta)
    binomial_logpmf = (
        gammaln(ns + 1)
        - gammaln(xs + 1)
        - gammaln(ns - xs + 1)
        + xs * math.log(overall_q)
        + (ns - xs) * math.log1p(-overall_q)
    )
    binomial_loglik = float(np.sum(weights * binomial_logpmf))
    if loglik + 1e-5 < binomial_loglik:
        invalid_reasons.append("worse_than_binomial")
    fit_valid = not invalid_reasons
    q_coef = theta[: int(q_degree) + 1]
    kappa_coef = theta[int(q_degree) + 1 :]

    row = {
        "family_offset": int(family_offset),
        "modality": modality,
        "kmer_percentage": float(kmer_percentage),
        "model": "length_aware_beta_binomial",
        "q_degree": int(q_degree),
        "kappa_degree": int(kappa_degree),
        "log_n_center": float(center),
        "log_n_scale": float(scale),
        "n_train_min": int(min(n_query_counts)),
        "n_train_max": int(max(n_query_counts)),
        "queries": int(sum(n_query_counts.values())),
        "nonzero_queries": nonzero_queries,
        "overall_hit_probability": overall_q,
        "loglik": float(loglik),
        "binomial_mle_loglik": binomial_loglik,
        "bbinom_loglik_gain": float(loglik - binomial_loglik),
        "aic": float(2 * theta.size - 2 * loglik),
        "optimizer_success": optimizer_success,
        "optimizer_attempt_count": optimizer_attempt_count,
        "optimizer_successful_attempt": optimizer_successful_attempt,
        "optimizer_message": str(result.message),
        "optimizer_status": int(getattr(result, "status", -1)),
        "optimizer_nit": int(getattr(result, "nit", -1)),
        "optimizer_nfev": int(getattr(result, "nfev", -1)),
        "fit_valid": fit_valid,
        "fit_invalid_reason": ";".join(invalid_reasons),
    }
    row.update({f"q_coef_{i}": float(v) for i, v in enumerate(q_coef)})
    row.update({f"kappa_coef_{i}": float(v) for i, v in enumerate(kappa_coef)})
    return row


def _fit_family_worker(args):
    return fit_family_bbinom_from_hist(*args)


def fit_bbinom_rows(hist, n_query_counts, selected_families, q_degree=2, kappa_degree=1, min_nonzero_queries=20, workers=1, modality="seq", kmer_percentage=100.0):
    if isinstance(hist, DenseFamilyHitHistogram):
        ranks = range(hist.n_families)

        def make_task(rank):
            return (
                int(selected_families[rank]),
                hist.records(rank),
                hist.query_counts(rank, n_query_counts),
                int(q_degree),
                int(kappa_degree),
                int(min_nonzero_queries),
                modality,
                float(kmer_percentage),
            )
    else:
        by_rank = defaultdict(list)
        for (rank, n, x), count in hist.items():
            by_rank[int(rank)].append((int(n), int(x), int(count)))
        ranks = sorted(by_rank)

        def make_task(rank):
            return (
                int(selected_families[rank]),
                by_rank[rank],
                dict(n_query_counts),
                int(q_degree),
                int(kappa_degree),
                int(min_nonzero_queries),
                modality,
                float(kmer_percentage),
            )

    rows = []
    if int(workers) > 1 and len(ranks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            progress = tqdm(
                total=len(ranks),
                desc="fit beta-binomial families",
                disable=is_progress_disabled(),
            )
            # Keep only a small number of variable-sized histogram records in
            # the multiprocessing queue at once.
            ranks = list(ranks)
            for start in range(0, len(ranks), 32):
                tasks = [make_task(rank) for rank in ranks[start : start + 32]]
                for row in pool.map(_fit_family_worker, tasks, chunksize=1):
                    if row is not None:
                        rows.append(row)
                    progress.update(1)
            progress.close()
    else:
        for rank in tqdm(
            ranks,
            desc="fit beta-binomial families",
            disable=is_progress_disabled(),
        ):
            row = _fit_family_worker(make_task(rank))
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def _fit_sampled_bbinom_coefficients(
    db,
    families,
    sampled_sequences,
    *,
    modality,
    min_nonzero_queries,
    workers,
    max_histogram_gb,
    kmer_filter_max_df,
    table_index,
    table_buffer=None,
    sampled_family_offsets=None,
    checkpoint_output_path=None,
):
    """Fit selected families from an already materialized query sample."""
    n_query_counts = Counter(int(n) for n, _ in sampled_sequences)
    selected_n = np.asarray(sorted(n_query_counts), dtype=np.int64)
    max_queries_per_n = max(n_query_counts.values())
    itemsize = 2 if max_queries_per_n <= np.iinfo(np.uint16).max else 4
    bytes_per_family = int(np.sum(selected_n + 1)) * itemsize
    if float(max_histogram_gb) > 0:
        max_batch_families = max(
            1,
            int(float(max_histogram_gb) * (1024**3) // bytes_per_family),
        )
    else:
        max_batch_families = families.size
    max_batch_families = min(max_batch_families, families.size)
    n_family_batches = int(math.ceil(families.size / max_batch_families))
    LOG.info(
        "Using {} family histogram batch(es), up to {} families and "
        "{:.2f} GiB each".format(
            n_family_batches,
            max_batch_families,
            max_batch_families * bytes_per_family / (1024**3),
        )
    )

    fitted_batches = []
    for batch_i, start in enumerate(
        range(0, families.size, max_batch_families),
        start=1,
    ):
        family_batch = families[start : start + max_batch_families]
        LOG.info(
            "Collecting family histogram batch {}/{} ({} families)".format(
                batch_i,
                n_family_batches,
                family_batch.size,
            )
        )
        collected = collect_dense_family_hit_histograms(
            sampled_sequences,
            family_batch,
            db,
            modality=modality,
            kmer_filter_max_df=kmer_filter_max_df,
            table_idx=table_index,
            table_buff=table_buffer,
            sampled_family_offsets=sampled_family_offsets,
        )
        hist, batch_query_counts = collected[:2]
        batch_rows = fit_bbinom_rows(
            hist,
            batch_query_counts,
            family_batch,
            q_degree=2,
            kappa_degree=1,
            min_nonzero_queries=min_nonzero_queries,
            workers=workers,
            modality=modality,
            kmer_percentage=db.ki.kmer_percentage,
        )
        if checkpoint_output_path:
            batch_output_path = "{}.batch-{:04d}-of-{:04d}.tsv".format(
                checkpoint_output_path,
                batch_i,
                n_family_batches,
            )
            batch_rows.to_csv(batch_output_path, sep="\t", index=False)
            LOG.info(
                "Checkpointed {} fitted rows to {}".format(
                    len(batch_rows),
                    batch_output_path,
                )
            )
        fitted_batches.append(batch_rows)
        del hist

    if fitted_batches:
        return pd.concat(fitted_batches, ignore_index=True)
    return pd.DataFrame()


def _store_training_metadata(
    db,
    modality,
    *,
    source,
    selected_n,
    record_count,
    sampled_record_count,
):
    """Persist enough fitting provenance to audit a built database."""
    attrs = db.db.root.Index._v_attrs
    prefix = "{}_bbinom_training_".format(modality)
    attrs[prefix + "source"] = source
    attrs[prefix + "n_values"] = ",".join(
        map(str, np.asarray(selected_n).tolist())
    )
    attrs[prefix + "record_count"] = int(record_count)
    attrs[prefix + "sampled_record_count"] = int(sampled_record_count)


def fit_bbinom_coefficients_from_buffer(
    db,
    sequence_buffer,
    *,
    modality="seq",
    table_index=None,
    table_buffer=None,
    max_families=0,
    min_family_prob=0.0,
    n_values=None,
    n_buckets=24,
    min_records_per_n=50,
    max_records_per_n=500,
    seed=42,
    min_nonzero_queries=20,
    workers=1,
    max_histogram_gb=3.0,
    family_offsets_path=None,
    family_shard=None,
    bounds=None,
    n_counts_cache=None,
    store_metadata=True,
):
    """Fit beta-binomial coefficients from buffers retained during mkdb."""
    if modality not in ("seq", "ss"):
        raise ValueError("modality must be 'seq' or 'ss', got {!r}".format(modality))
    if modality == "ss" and not db.has_structure():
        raise ValueError("Database has no structure index; cannot fit ss coefficients")

    families = select_family_offsets(
        db,
        family_offsets_path=family_offsets_path,
        min_family_prob=min_family_prob,
        max_families=max_families,
        seed=seed,
        modality=modality,
    )
    if families.size == 0:
        raise RuntimeError("No families selected for fitting")
    LOG.info("Selected {} families for beta-binomial fitting".format(families.size))
    families = apply_family_shard(families, family_shard)
    if families.size == 0:
        raise RuntimeError("Family shard selected no families")

    index_node, buffer_node = _modality_index_arrays(db, modality)
    if table_index is None:
        table_index = index_node[:]
    if table_buffer is None:
        table_buffer = buffer_node[:]
    kmer_filter_max_df = _modality_kmer_max_df(db, modality)
    n_records = len(db.protein_table)
    n_unique = load_or_scan_unique_counts(
        sequence_buffer,
        n_records,
        db.ki.k,
        db.ki.alphabet,
        table_index,
        kmer_filter_max_df,
        bounds=bounds,
        cache_path=n_counts_cache,
    )
    _, selected_n = choose_sequence_n_values(
        n_unique,
        min_records_per_n=min_records_per_n,
        n_buckets=n_buckets,
        requested_n_values=parse_n_values(n_values),
    )
    LOG.info(
        "Selected exact N values: {}".format(
            ",".join(map(str, selected_n.tolist()))
        )
    )
    sampled_by_n = sample_record_indices_by_n(
        n_unique,
        selected_n,
        max_records_per_n=max_records_per_n,
        seed=seed,
    )
    sampled_sequences, sampled_indices = load_sampled_buffer_sequences(
        sequence_buffer,
        n_records,
        sampled_by_n,
        bounds=bounds,
    )
    if store_metadata:
        _store_training_metadata(
            db,
            modality,
            source="database_protein_buffer",
            selected_n=selected_n,
            record_count=n_records,
            sampled_record_count=len(sampled_indices),
        )
    protein_hogs = db.protein_table.col("HOGoff")
    hog_families = db.hog_table.col("FamOff")
    sampled_family_offsets = hog_families[protein_hogs[sampled_indices]]

    rows = _fit_sampled_bbinom_coefficients(
        db,
        families,
        sampled_sequences,
        modality=modality,
        min_nonzero_queries=min_nonzero_queries,
        workers=workers,
        max_histogram_gb=max_histogram_gb,
        kmer_filter_max_df=kmer_filter_max_df,
        table_index=table_index,
        table_buffer=table_buffer,
        sampled_family_offsets=sampled_family_offsets,
    )
    valid_count = int(rows["fit_valid"].sum()) if "fit_valid" in rows else 0
    LOG.info(
        "Fitted {} coefficient rows for '{}' ({} valid, {} invalid)".format(
            len(rows),
            modality,
            valid_count,
            len(rows) - valid_count,
        )
    )
    return rows
