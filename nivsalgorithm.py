import math
from scipy.stats import entropy as scipy_entropy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from UtilityNeRF import get_coarse_query_points, get_fine_query_points, render_radiance_volume, run_one_iter_of_nerf


def select_top_views_at_iter(
    *,
    poses, images, init_ds, init_o,
    coarse_mlp, fine_mlp,
    num_coarse_sample, t_i_c_bin_edges, t_i_c_gap,
    num_fine_sample, t_f, chunk_size,
    top_fraction=0.7,
    n_probe=4096,
    probe_seed=42,
    bins=60,
    opacity_thresh=0.01,
    device="cuda",
):
    """
    Returns top view indices by InfoGain score based on p(T_final | view),
    computed using only foreground rays (opacity > opacity_thresh).
    """
    K = poses.shape[0]
    H = init_ds.shape[0]
    W = init_ds.shape[1]

    # fixed probe pixels shared across all views
    rng = np.random.default_rng(probe_seed)
    probe_idx = rng.choice(H * W, size=min(n_probe, H * W), replace=False)
    probe_idx = torch.from_numpy(probe_idx).long().to(device)

    pks = []
    Hk = np.zeros(K, dtype=np.float64)

    with torch.no_grad():
        for k in range(K):
            # build rays for view k (your convention)
            R = torch.tensor(poses[k, :3, :3], device=device, dtype=init_ds.dtype)
            ds = torch.einsum("ij,hwj->hwi", R, init_ds)
            os = (R @ init_o).expand(ds.shape)

            ds_flat = ds.view(H * W, 3)[probe_idx]
            os_flat = os.view(H * W, 3)[probe_idx]

            # shape to (N_probe,1,3) so your run_one_iter works
            ds_probe = ds_flat.view(-1, 1, 3)
            os_probe = os_flat.view(-1, 1, 3)

            # run nerf forward
            C_rs_c, C_rs_f, w_is_c, w_is_f, sigma_c, sigma_f, alpha_c, alpha_f, T_c, T_f = run_one_iter_of_nerf(
                ds_probe,
                num_coarse_sample,
                t_i_c_bin_edges,
                t_i_c_gap,
                os_probe,
                chunk_size,
                coarse_mlp,
                num_fine_sample,
                t_f,
                fine_mlp,
            )

            # foreground filter using opacity
            opacity = w_is_f.sum(dim=-1).view(-1)      # (N_probe,)
            T_final = T_f[..., -1].view(-1)            # (N_probe,)

            hit = opacity > opacity_thresh
            T_hit = T_final[hit]

            # fallback if too few rays survive
            if T_hit.numel() < 256:
                hit = opacity > (opacity_thresh * 0.1)
                T_hit = T_final[hit]

            # histogram -> p_k(t)
            T_np = T_hit.detach().cpu().numpy()
            hist, _ = np.histogram(T_np, bins=bins, range=(0.0, 1.0), density=False)
            pk = (hist.astype(np.float64) + 1e-12)
            pk = pk / pk.sum()

            pks.append(pk)
            Hk[k] = scipy_entropy(pk, base=2)  # bits

    pks = np.stack(pks, axis=0)          # (K,bins)
    p_marg = pks.mean(axis=0)
    H_marg = scipy_entropy(p_marg, base=2)

    scores = H_marg - Hk                # InfoGain per view
    order = np.argsort(-scores)         # descending

    n_select = max(1, int(math.ceil(top_fraction * K)))
    selected = order[:n_select].tolist()

    return selected, scores, Hk, H_marg


def compute_scores_all_views(
    images_full,
    poses_full,
    init_ds,
    init_o,
    coarse_mlp,
    fine_mlp,
    num_coarse_sample,
    t_i_c_bin_edges,
    t_i_c_gap,
    num_fine_sample,
    t_f,
    chunk_size,
    n_probe=4096,
    bins=60,
    opacity_thresh=0.01,
    device="cuda",
):
    K = poses_full.shape[0]
    H = init_ds.shape[0]
    W = init_ds.shape[1]

    # fixed probe pixel indices
    rng = np.random.default_rng(0)
    probe_idx = rng.choice(H * W, size=min(n_probe, H * W), replace=False)
    probe_idx = torch.from_numpy(probe_idx).long().to(device)

    pks = []

    with torch.no_grad():
        for k in range(K):

            target_pose = poses_full[k].to(device)
            R = target_pose[:3, :3]

            ds = torch.einsum("ij,hwj->hwi", R, init_ds)
            os = (R @ init_o).expand(ds.shape)

            ds_flat = ds.view(H * W, 3)[probe_idx]
            os_flat = os.view(H * W, 3)[probe_idx]

            ds_probe = ds_flat.view(-1, 1, 3)
            os_probe = os_flat.view(-1, 1, 3)

            _, _, _, w_is_f, _, _, _, _, _, T_f = run_one_iter_of_nerf(
                ds_probe,
                num_coarse_sample,
                t_i_c_bin_edges,
                t_i_c_gap,
                os_probe,
                chunk_size,
                coarse_mlp,
                num_fine_sample,
                t_f,
                fine_mlp,
            )

            opacity = w_is_f.sum(dim=-1).view(-1)
            T_final = T_f[..., -1].view(-1)

            hit = opacity > opacity_thresh
            T_hit = T_final[hit]

            if T_hit.numel() < 256:
                T_hit = T_final  # fallback

            T_np = T_hit.detach().cpu().numpy()

            hist, _ = np.histogram(T_np, bins=bins, range=(0.0, 1.0))
            pk = hist.astype(np.float64) + 1e-12
            pk = pk / pk.sum()

            pks.append(pk)

    pks = np.stack(pks, axis=0)

    p_marg = pks.mean(axis=0)

    H_marg = scipy_entropy(p_marg, base=2)

    scores = np.zeros(K)

    for k in range(K):
        Hk = scipy_entropy(pks[k], base=2)
        scores[k] = H_marg - Hk

    return scores