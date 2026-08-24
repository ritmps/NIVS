import numpy as np


def detect_entropy_degeneracy(
    entropy_values,
    eval_iters,
    tol_d1=1e-3,
    tol_d2=1e-3,
    tol_rel=1e-3,
    min_consecutive=3
):
    """
    Detect entropy degeneracy / stabilization point.

    Args:
        entropy_values: list or np.ndarray of entropy values [H_0, H_1, ...]
        eval_iters: corresponding training iterations
        tol_d1: threshold for first difference |ΔH|
        tol_d2: threshold for second difference |Δ²H|
        tol_rel: threshold for relative entropy change
        min_consecutive: number of consecutive stable evaluations required

    Returns:
        result: dict with:
            - degeneracy_iter
            - degeneracy_index
            - diff1
            - diff2
            - rel_change
            - stable_mask
    """
    H = np.asarray(entropy_values, dtype=np.float64)
    eval_iters = np.asarray(eval_iters)

    n = len(H)
    if n == 0:
        return {
            "degeneracy_iter": None,
            "degeneracy_index": None,
            "diff1": np.array([]),
            "diff2": np.array([]),
            "rel_change": np.array([]),
            "stable_mask": np.array([], dtype=bool),
        }

    diff1 = np.full(n, np.nan, dtype=np.float64)
    diff2 = np.full(n, np.nan, dtype=np.float64)
    rel_change = np.full(n, np.nan, dtype=np.float64)

    for k in range(1, n):
        diff1[k] = H[k] - H[k - 1]
        rel_change[k] = abs(diff1[k]) / (abs(H[k - 1]) + 1e-12)

    for k in range(2, n):
        diff2[k] = diff1[k] - diff1[k - 1]

    stable_mask = (
        (np.abs(diff1) < tol_d1) &
        (np.abs(diff2) < tol_d2) &
        (rel_change < tol_rel)
    )

    degeneracy_index = None
    count = 0
    for k in range(n):
        if stable_mask[k]:
            count += 1
            if count >= min_consecutive:
                degeneracy_index = k - min_consecutive + 1
                break
        else:
            count = 0

    degeneracy_iter = None if degeneracy_index is None else int(eval_iters[degeneracy_index])

    return {
        "degeneracy_iter": degeneracy_iter,
        "degeneracy_index": degeneracy_index,
        "diff1": diff1,
        "diff2": diff2,
        "rel_change": rel_change,
        "stable_mask": stable_mask,
    }
def shannon_entropy_from_hist(values, bins=80, value_range=None, eps=1e-12):
    hist, bin_edges = np.histogram(values, bins=bins, range=value_range, density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + eps)
    H = -np.sum(p * np.log(p + eps))
    return H, p, bin_edges
