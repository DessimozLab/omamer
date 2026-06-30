"""
Fit length-aware beta-binomial family coefficients from sequence FASTA records.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numba
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from tqdm.auto import tqdm

from ._utils import LOG, is_progress_disabled
from .alphabets import get_transform
from .merge_search import parse_seq
from .sequence_reader import SequenceReader


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


def _iter_sequences(paths, k, chunksize, sanitiser):
    record_i = 0
    for path in paths:
        for ids, seqs in SequenceReader.read(path, k=k, chunksize=chunksize, sanitiser=sanitiser):
            for seq_id, seq in zip(ids, seqs):
                yield record_i, seq_id, seq
                record_i += 1


def _unique_valid_codes(seq, k, trans, digits_lookup, x_flag):
    n_kmers = len(seq) - (k - 1)
    if n_kmers <= 0:
        return np.empty(0, dtype=np.uint32)
    seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    codes, _, _ = parse_seq(seq_arr, digits_lookup, n_kmers, k, trans, x_flag)
    return codes[codes != x_flag]


def scan_sequence_unique_counts(paths, k, alphabet, table_index_size, chunksize):
    trans = get_transform(k, alphabet.DIGITS_AA)
    digits_lookup = alphabet.DIGITS_AA_LOOKUP
    x_flag = table_index_size - 1
    counts = []
    for _, _, seq in tqdm(
        _iter_sequences(paths, k, chunksize, alphabet.sanitise_seq),
        desc="scan exact unique-kmer counts",
        disable=is_progress_disabled(),
    ):
        counts.append(int(_unique_valid_codes(seq, k, trans, digits_lookup, x_flag).size))
    return np.asarray(counts, dtype=np.uint32)


def parse_n_values(value):
    if not value:
        return None
    return [int(x) for x in str(value).replace(",", " ").split()]


def choose_sequence_n_values(n_unique, min_records_per_n=50, n_buckets=24, requested_n_values=None):
    hist = np.bincount(np.asarray(n_unique, dtype=np.int64))
    eligible = np.flatnonzero(hist >= int(min_records_per_n))
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


def collect_family_hit_histograms(
    paths,
    sampled_by_n,
    selected_families,
    db,
    chunksize=10000,
    modality="seq",
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
        codes = _unique_valid_codes(seq, k, trans, digits_lookup, x_flag)
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


def fit_family_bbinom_from_hist(
    family_offset,
    records,
    n_query_counts,
    q_degree=2,
    kappa_degree=1,
    min_nonzero_queries=1,
    modality="seq",
):
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

    def unpack(theta):
        q_coef = theta[: int(q_degree) + 1]
        kappa_coef = theta[int(q_degree) + 1 :]
        q = np.clip(expit(xq @ q_coef), 1e-8, 1 - 1e-8)
        log_kappa = np.clip(xk @ kappa_coef, math.log(1e-4), math.log(1e8))
        kappa = np.exp(log_kappa)
        return q, kappa

    def objective(theta):
        q, kappa = unpack(theta)
        alpha = q * kappa
        beta = (1.0 - q) * kappa
        logpmf = _betabinom_logpmf(xs, ns, alpha, beta)
        if not np.all(np.isfinite(logpmf)):
            return float("inf")
        return -float(np.sum(weights * logpmf))

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=[(-50.0, 50.0)] * theta0.size,
    )
    theta = result.x if result.success else theta0
    loglik = -objective(theta)
    q_coef = theta[: int(q_degree) + 1]
    kappa_coef = theta[int(q_degree) + 1 :]

    row = {
        "family_offset": int(family_offset),
        "modality": modality,
        "model": "length_aware_beta_binomial",
        "q_degree": int(q_degree),
        "kappa_degree": int(kappa_degree),
        "log_n_center": float(center),
        "log_n_scale": float(scale),
        "n_train_min": int(min(n_query_counts)),
        "n_train_max": int(max(n_query_counts)),
        "queries": int(sum(n_query_counts.values())),
        "nonzero_queries": nonzero_queries,
        "loglik": float(loglik),
        "aic": float(2 * theta.size - 2 * loglik),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }
    row.update({f"q_coef_{i}": float(v) for i, v in enumerate(q_coef)})
    row.update({f"kappa_coef_{i}": float(v) for i, v in enumerate(kappa_coef)})
    return row


def _fit_family_worker(args):
    return fit_family_bbinom_from_hist(*args)


def fit_bbinom_rows(hist, n_query_counts, selected_families, q_degree=2, kappa_degree=1, min_nonzero_queries=1, workers=1, modality="seq"):
    by_rank = defaultdict(list)
    for (rank, n, x), count in hist.items():
        by_rank[int(rank)].append((int(n), int(x), int(count)))

    tasks = [
        (
            int(selected_families[rank]),
            records,
            dict(n_query_counts),
            int(q_degree),
            int(kappa_degree),
            int(min_nonzero_queries),
            modality,
        )
        for rank, records in sorted(by_rank.items())
    ]

    rows = []
    if int(workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            iterator = pool.map(_fit_family_worker, tasks, chunksize=16)
            for row in tqdm(iterator, total=len(tasks), desc="fit beta-binomial families", disable=is_progress_disabled()):
                if row is not None:
                    rows.append(row)
    else:
        for task in tqdm(tasks, desc="fit beta-binomial families", disable=is_progress_disabled()):
            row = _fit_family_worker(task)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


def compute_bbinom_coefficients(
    db,
    sequence_paths,
    output_path,
    family_offsets_path=None,
    max_families=0,
    min_family_prob=0.0,
    n_values=None,
    n_buckets=24,
    min_records_per_n=50,
    max_records_per_n=500,
    chunksize=10000,
    seed=42,
    min_nonzero_queries=1,
    workers=1,
    n_summary_path=None,
    modality="seq",
):
    if not sequence_paths:
        raise ValueError("At least one sequence FASTA path is required")
    if modality not in ("seq", "ss"):
        raise ValueError("modality must be 'seq' or 'ss', got {!r}".format(modality))
    if modality == "ss" and not db.has_structure():
        raise ValueError("Database has no structure index; cannot fit ss coefficients")
    if db.ki.alphabet.n != 21:
        LOG.warning("Computing coefficients with alphabet size {}".format(db.ki.alphabet.n))
    LOG.info("Fitting beta-binomial coefficients for modality '{}'".format(modality))

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

    index_node, _ = _modality_index_arrays(db, modality)
    n_unique = scan_sequence_unique_counts(
        sequence_paths,
        db.ki.k,
        db.ki.alphabet,
        index_node.shape[0],
        chunksize,
    )
    summary, selected_n = choose_sequence_n_values(
        n_unique,
        min_records_per_n=min_records_per_n,
        n_buckets=n_buckets,
        requested_n_values=parse_n_values(n_values),
    )
    LOG.info("Selected exact N values: {}".format(",".join(map(str, selected_n.tolist()))))
    if n_summary_path:
        summary.to_csv(n_summary_path, sep="\t", index=False)

    sampled_by_n = sample_record_indices_by_n(
        n_unique,
        selected_n,
        max_records_per_n=max_records_per_n,
        seed=seed,
    )
    LOG.info("Sampled {} sequence records for fitting".format(sum(len(v) for v in sampled_by_n.values())))

    hist, n_query_counts = collect_family_hit_histograms(
        sequence_paths,
        sampled_by_n,
        families,
        db,
        chunksize=chunksize,
        modality=modality,
    )
    LOG.info("Collected {} nonzero family/N/X histogram bins".format(len(hist)))

    rows = fit_bbinom_rows(
        hist,
        n_query_counts,
        families,
        q_degree=2,
        kappa_degree=1,
        min_nonzero_queries=min_nonzero_queries,
        workers=workers,
        modality=modality,
    )
    rows.to_csv(output_path, sep="\t", index=False)
    LOG.info("Wrote {} coefficient rows to {}".format(len(rows), output_path))
    return rows
