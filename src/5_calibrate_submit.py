import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import full_path, load_config, save_json


def rank01(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return (ranks / float(n - 1)).astype(np.float32)


def zscore_per_cluster(score: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    z = np.zeros(len(score), dtype=np.float32)
    uniq = np.unique(cluster_ids)
    for k in uniq:
        idx = np.where(cluster_ids == k)[0]
        if len(idx) == 0:
            continue
        s = score[idx]
        mu = float(np.mean(s))
        sigma = float(np.std(s))
        if sigma < 1e-8:
            z[idx] = 0.0
        else:
            z[idx] = ((s - mu) / sigma).astype(np.float32)
    return z


def _gauss_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-8)
    coef = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    expo = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return coef * expo


def find_gmm_intersection_or_quantile(z: np.ndarray, gmm: GaussianMixture, pos_comp: int):
    means = gmm.means_.ravel().astype(float)
    variances = gmm.covariances_.reshape(-1).astype(float)
    sigmas = np.sqrt(np.maximum(variances, 1e-8))
    weights = gmm.weights_.ravel().astype(float)

    neg_comp = 1 - int(pos_comp)
    lo = float(np.min(z) - 1.0)
    hi = float(np.max(z) + 1.0)
    grid = np.linspace(lo, hi, 4096, dtype=np.float64)

    p_pos = weights[pos_comp] * _gauss_pdf(grid, means[pos_comp], sigmas[pos_comp])
    p_neg = weights[neg_comp] * _gauss_pdf(grid, means[neg_comp], sigmas[neg_comp])
    diff = p_pos - p_neg

    sign_change = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(sign_change) > 0:
        centers = 0.5 * (grid[sign_change] + grid[sign_change + 1])
        mid = 0.5 * (means[pos_comp] + means[neg_comp])
        pick = int(np.argmin(np.abs(centers - mid)))
        return float(centers[pick]), "intersection"

    q = float(np.clip(weights[pos_comp], 0.01, 0.99))
    thresh = float(np.quantile(z, 1.0 - q))
    return thresh, "quantile_fallback"

def best_expected_f1_threshold(scores: np.ndarray, q_post: np.ndarray):
    # scores: higher => more likely positive; q_post: posterior P(y=1|score)
    order = np.argsort(-scores)
    scores_s = scores[order]
    q_s = q_post[order]

    # cumulative sums for predicted positives as threshold moves down
    cumsum_q = np.cumsum(q_s)  # expected TP for top-k
    ks = np.arange(1, len(scores_s) + 1)
    expected_TP = cumsum_q
    expected_FP = ks - expected_TP
    expected_FN = np.sum(q_s) - expected_TP

    denom = 2.0 * expected_TP + expected_FP + expected_FN
    # avoid divide by zero
    valid = denom > 0
    expF1 = np.zeros_like(denom)
    expF1[valid] = (2.0 * expected_TP[valid]) / denom[valid]

    # pick top k maximizing expected F1; threshold is score at position k-1
    best_idx = int(np.argmax(expF1))
    best_threshold = scores_s[best_idx]
    # labels: score >= best_threshold
    labels = (scores >= best_threshold).astype(np.int64)
    return labels, float(expF1[best_idx]), best_threshold


def main():
    cfg = load_config()
    print("calib: load tri-view predictions", flush=True)

    p_lex = np.load(full_path("cache/preds_lexical.npy")).astype(np.float32)
    p_struct = np.load(full_path("cache/preds_structural.npy")).astype(np.float32)
    p_sym = np.load(full_path("cache/preds_symbolic.npy")).astype(np.float32)

    x_test_inv_path = full_path(cfg["paths"].get("x_test_invariant", "cache/X_test_invariant.npy"))
    if not x_test_inv_path.exists():
        raise FileNotFoundError("Missing invariant features for BGMM clustering. Run src/1_extract_features.py first.")
    print("calib: load invariant test features", flush=True)
    x_test_inv = np.load(x_test_inv_path).astype(np.float32)

    n = len(p_lex)
    if len(p_struct) != n or len(p_sym) != n or len(x_test_inv) != n:
        raise ValueError(
            f"length mismatch: lexical={len(p_lex)} structural={len(p_struct)} symbolic={len(p_sym)} x_test_inv={len(x_test_inv)}"
        )

    test_ids = np.load(full_path(cfg["paths"]["test_ids"]))
    if len(test_ids) != n:
        test_df = pd.read_parquet(full_path(cfg["paths"]["test_parquet"]))
        test_ids = test_df[cfg["columns"]["test_id"]].to_numpy()
    if len(test_ids) != n:
        test_ids = np.arange(n, dtype=np.int64)

    # rank averaging ensemble
    print("calib: rank-ensemble scores", flush=True)
    r_lex = rank01(p_lex)
    r_struct = rank01(p_struct)
    r_sym = rank01(p_sym)
    score = ((r_lex + r_struct + r_sym) / 3.0).astype(np.float32)

    # latent-domain BGMM on structural space
    print("calib: scale features for BGMM", flush=True)
    x_scaled = StandardScaler().fit_transform(x_test_inv)
    rng = np.random.default_rng(int(cfg.get("seed", 2262)))
    fit_rows = int(min(n, 200000))
    if fit_rows < n:
        fit_idx = rng.choice(n, size=fit_rows, replace=False)
        x_fit = x_scaled[fit_idx]
    else:
        x_fit = x_scaled
    print(f"calib: fit BGMM on {len(x_fit)} rows; this can take a while", flush=True)
    bgmm = BayesianGaussianMixture(
        n_components=30,
        covariance_type="diag",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=0.01,
        reg_covar=1e-5,
        max_iter=350,
        n_init=1,
        init_params="random_from_data",
        verbose=2,
        verbose_interval=10,
        random_state=int(cfg.get("seed", 2262)),
    )
    bgmm.fit(x_fit)
    print("calib: BGMM fit done; assign clusters", flush=True)
    cluster_ids = bgmm.predict(x_scaled)
    uniq_clusters, cnt_clusters = np.unique(cluster_ids, return_counts=True)
    min_cluster_size = max(64, int(0.02 * n))
    active_clusters = int(np.sum(cnt_clusters >= min_cluster_size))

    # cluster-conditional z-score normalization
    print("calib: cluster conditional z-score", flush=True)
    z = zscore_per_cluster(score, cluster_ids)

    # global 2-GMM thresholding on normalized scores
    print("calib: fit global 2-GMM", flush=True)
    z_unit = z.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=int(cfg.get("seed", 2262)))
    gmm.fit(z_unit)
    probs = gmm.predict_proba(z_unit)
    means = gmm.means_.ravel()
    pos_comp = int(np.argmax(means))
    q_post = probs[:, pos_comp]

    labels_expf1, expf1_auto, thresh_expf1 = best_expected_f1_threshold(z, q_post)
    thresh_gmm, threshold_mode = find_gmm_intersection_or_quantile(z, gmm, pos_comp)
    labels_gmm = (z >= thresh_gmm).astype(np.int64)

    print("calib: write submission files", flush=True)
    out_files = []
    stats = {}

    out_auto = "submission_zscore_auto.csv"
    pd.DataFrame({"id": test_ids, "label": labels_gmm}).to_csv(full_path(out_auto), index=False)
    out_files.append(out_auto)

    # keep legacy sweep outputs for quick comparison in experiments
    priors = [0.35, 0.45, 0.55]
    for prior in priors:
        k = max(1, min(n - 1, int(round(prior * n))))
        labels_prior = np.zeros(n, dtype=np.int64)
        top = np.argsort(-z)[:k]
        labels_prior[top] = 1
        out_name = f"submission_zscore_{prior:.2f}.csv"
        pd.DataFrame({"id": test_ids, "label": labels_prior}).to_csv(full_path(out_name), index=False)
        out_files.append(out_name)
        stats[f"{prior:.2f}"] = {
            "positive_rate": float(labels_prior.mean()),
            "positive_count": int(labels_prior.sum()),
        }

    # write default submission using auto GMM boundary
    pd.DataFrame({"id": test_ids, "label": labels_gmm}).to_csv(full_path(cfg["paths"]["submission"]), index=False)

    stats["auto"] = {
        "positive_rate": float(labels_gmm.mean()),
        "positive_count": int(labels_gmm.sum()),
        "threshold": float(thresh_gmm),
        "threshold_mode": threshold_mode,
        "expected_f1_proxy": float(expf1_auto),
        "expf1_threshold": float(thresh_expf1),
        "positive_rate_expf1": float(labels_expf1.mean()),
        "gmm_weights": [float(x) for x in gmm.weights_.ravel().tolist()],
        "gmm_means": [float(x) for x in means.tolist()],
    }

    cluster_counts = {str(int(k)): int(v) for k, v in zip(uniq_clusters.tolist(), cnt_clusters.tolist())}

    save_json(
        {
            "strategy": "tri_view_rank_avg_plus_bgmm_cluster_zscore",
            "outputs": out_files,
            "priors": priors,
            "stats": stats,
            "active_domains": active_clusters,
            "active_domain_min_size": int(min_cluster_size),
            "bgmm_total_components": 30,
            "domain_counts": cluster_counts,
            "n_test": int(n),
        },
        full_path(cfg["paths"]["calibration_meta"]),
    )

    print("calib: finalize", flush=True)
    print("checkpoint_5")
    print("saved", out_files)
    print("default_submission", cfg["paths"]["submission"])
    print(f"BGMM used {active_clusters} active domains")
    print("Detected Global Prior", float(labels_gmm.mean()))
    print("stats", stats)


if __name__ == "__main__":
    main()
