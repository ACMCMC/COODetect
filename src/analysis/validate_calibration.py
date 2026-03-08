import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import full_path, load_config, save_json


def build_density_lut(scores: np.ndarray, labels: np.ndarray, bins: int = 300):
    eps = 1e-6
    s = np.clip(scores.astype(np.float64), eps, 1.0 - eps)
    y = labels.astype(np.int64)

    h0, edges = np.histogram(s[y == 0], bins=bins, range=(0.0, 1.0), density=True)
    h1, _ = np.histogram(s[y == 1], bins=bins, range=(0.0, 1.0), density=True)

    h0 = h0 + 1e-4
    h1 = h1 + 1e-4
    h0 = h0 / h0.sum()
    h1 = h1 / h1.sum()
    return h0.astype(np.float64), h1.astype(np.float64), edges


def lookup_likelihoods(scores: np.ndarray, h0: np.ndarray, h1: np.ndarray, edges: np.ndarray):
    idx = np.searchsorted(edges, np.clip(scores, 0.0, 1.0), side="right") - 1
    idx = np.clip(idx, 0, len(h0) - 1)
    l0 = np.clip(h0[idx], 1e-9, None)
    l1 = np.clip(h1[idx], 1e-9, None)
    return l0, l1


def em_prior(scores_test: np.ndarray, h0: np.ndarray, h1: np.ndarray, edges: np.ndarray, init_prior: float, max_iter: int = 120):
    pi = float(init_prior)
    eps = 1e-12
    l0, l1 = lookup_likelihoods(scores_test.astype(np.float64), h0, h1, edges)

    for _ in range(max_iter):
        num = pi * l1
        den = num + (1.0 - pi) * l0
        q = num / np.clip(den, eps, None)
        new_pi = float(np.mean(q))
        if abs(new_pi - pi) < 1e-8:
            pi = new_pi
            break
        pi = new_pi
    return float(pi)


def canon_lang(x: str) -> str:
    low = str(x).strip().lower()
    if "python" in low:
        return "Python"
    if "java" in low:
        return "Java"
    if "c++" in low or low == "cpp":
        return "C++"
    return "OTHER"


def distorted_sample(scores: np.ndarray, labels: np.ndarray, target_prior: float, seed: int):
    rng = np.random.default_rng(seed)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    n1_max = len(idx1)
    n0_max = len(idx0)

    n_from_1 = int(min(n1_max, (target_prior * n0_max) / max(1e-12, 1.0 - target_prior)))
    n_from_0 = int(min(n0_max, (n_from_1 * (1.0 - target_prior)) / max(1e-12, target_prior)))

    n_from_1 = max(n_from_1, 1)
    n_from_0 = max(n_from_0, 1)

    pick1 = rng.choice(idx1, size=n_from_1, replace=False)
    pick0 = rng.choice(idx0, size=n_from_0, replace=False)

    merged = np.concatenate([pick0, pick1])
    rng.shuffle(merged)

    return scores[merged], labels[merged]


def main():
    cfg = load_config()
    seed = int(cfg["seed"])

    y_train = np.load(full_path(cfg["paths"]["y_train"])).astype(np.int64)
    langs_raw = np.load(full_path(cfg["paths"]["train_languages"]), allow_pickle=True)
    langs = np.array([canon_lang(x) for x in langs_raw], dtype=object)

    fold_defs = [
        ("cpp", "C++"),
        ("java", "Java"),
        ("py", "Python"),
    ]

    target_prior = 0.20
    fold_rows = []

    for fold_tag, holdout_lang in fold_defs:
        fold_scores = np.load(full_path(f"cache/ood_preds_fold_{fold_tag}.npy")).astype(np.float32)
        fold_mask = langs == holdout_lang
        fold_labels = y_train[fold_mask].astype(np.int64)

        h0, h1, edges = build_density_lut(fold_scores, fold_labels, bins=300)
        distorted_scores, distorted_labels = distorted_sample(fold_scores, fold_labels, target_prior=target_prior, seed=seed)

        init_prior = float(fold_labels.mean())
        estimated_prior = em_prior(distorted_scores, h0, h1, edges, init_prior=init_prior, max_iter=120)
        true_prior = float(distorted_labels.mean())
        abs_err = abs(estimated_prior - true_prior)

        fold_rows.append(
            {
                "fold": fold_tag,
                "holdout_lang": holdout_lang,
                "init_prior": init_prior,
                "true_prior": true_prior,
                "estimated_prior": estimated_prior,
                "abs_error": abs_err,
                "n_eval": int(len(distorted_scores)),
            }
        )

    mae = float(np.mean([row["abs_error"] for row in fold_rows]))
    passed = bool(mae < 0.05)

    out = {
        "target_prior": target_prior,
        "folds": fold_rows,
        "mean_abs_error": mae,
        "pass_criteria": "mae<0.05",
        "passed": passed,
    }
    save_json(out, full_path("cache/calibration_validation.json"))

    print("checkpoint_4_validate")
    print("mean_abs_error", mae)
    print("passed", passed)
    for row in fold_rows:
        print(row)


if __name__ == "__main__":
    main()
