import numpy as np
import math

def entropy_only_topk(scores, frac=0.7):
    """
    Global top-k by score (entropy-only selection).
    """
    scores = np.asarray(scores)
    K = scores.shape[0]
    n = int(math.ceil(frac * K))
    return np.argsort(-scores)[:n].tolist()

def coverage_topk(scores, frac=0.8, n_bins=12):
    """
    Coverage-constrained top-k:
      - split indices 0..K-1 into n_bins consecutive bins
      - pick top views per bin by score
      - ensures total selected ~= frac*K (exactly enforced)
    Assumes view index order roughly follows camera trajectory (common in NeRF datasets).
    """
    scores = np.asarray(scores)
    K = scores.shape[0]
    n_select = int(math.ceil(frac * K))

    all_idx = np.arange(K)
    bins = np.array_split(all_idx, n_bins)

    base = n_select // n_bins
    rem  = n_select % n_bins
    take_per_bin = [base + (1 if b < rem else 0) for b in range(n_bins)]

    selected = []
    for b, idx_bin in enumerate(bins):
        tb = take_per_bin[b]
        if tb <= 0 or len(idx_bin) == 0:
            continue
        local_sorted = idx_bin[np.argsort(-scores[idx_bin])]
        selected.extend(local_sorted[:tb].tolist())

    # enforce exact size n_select
    selected = list(dict.fromkeys(selected))  # unique while preserving order

    if len(selected) < n_select:
        remaining = np.setdiff1d(all_idx, np.array(selected), assume_unique=False)
        remaining_sorted = remaining[np.argsort(-scores[remaining])]
        selected.extend(remaining_sorted[:(n_select - len(selected))].tolist())

    if len(selected) > n_select:
        sel = np.array(selected)
        sel = sel[np.argsort(-scores[sel])][:n_select]
        selected = sel.tolist()

    return selected

def union_superset(a, b):
    """
    Superset = union of two index lists, returned as sorted unique indices.
    """
    return sorted(set(a).union(set(b)))

def gap_stats(idxs):
    s = np.sort(np.array(idxs, dtype=np.int64))
    if len(s) < 2:
        return {"min_gap": None, "mean_gap": None, "head": s.tolist(), "tail": s.tolist()}
    diffs = np.diff(s)
    return {
        "min_gap": int(diffs.min()),
        "mean_gap": float(diffs.mean()),
        "head": s[:20].tolist(),
        "tail": s[-20:].tolist(),
        "count": int(len(s)),
        "min_idx": int(s.min()),
        "max_idx": int(s.max()),
    }