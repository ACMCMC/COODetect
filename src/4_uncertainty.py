import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans

from common import full_path, load_config, stable_sample_indices


def minmax_unit(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def load_train_labels(cfg: dict, expected_n: int) -> np.ndarray:
    y = np.load(full_path(cfg["paths"]["y_train"]))
    if len(y) != expected_n:
        df = pd.read_parquet(full_path(cfg["paths"]["train_parquet"]))
        y = df[cfg["columns"]["train_label"]].to_numpy(dtype=np.int64)
    if len(y) != expected_n:
        y = y[:expected_n]
    return y.astype(np.int64)


def get_leaf_embeddings(cfg: dict, xgb_model, x_train: np.ndarray, x_test: np.ndarray):
    cache_dir = full_path(cfg["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tr_path = cache_dir / "leaf_train_all.npy"
    te_path = cache_dir / "leaf_test_all.npy"

    if tr_path.exists() and te_path.exists():
        tr = np.load(tr_path)
        te = np.load(te_path)
        if tr.shape[0] == x_train.shape[0] and te.shape[0] == x_test.shape[0]:
            return tr.astype(np.float32), te.astype(np.float32)

    tr = xgb_model.apply(x_train)
    te = xgb_model.apply(x_test)
    if tr.ndim == 1:
        tr = tr[:, None]
    if te.ndim == 1:
        te = te[:, None]
    tr = tr.astype(np.float32)
    te = te.astype(np.float32)

    np.save(tr_path, tr)
    np.save(te_path, te)
    return tr, te


def compute_tree_disagreement(cfg: dict, xgb_model, x_train: np.ndarray, x_test: np.ndarray, n_trees: int):
    cache_dir = full_path(cfg["paths"]["cache_dir"])
    tr_path = cache_dir / "tree_disagreement_train.npy"
    te_path = cache_dir / "tree_disagreement_test.npy"

    if tr_path.exists() and te_path.exists():
        tr = np.load(tr_path)
        te = np.load(te_path)
        if len(tr) == len(x_train) and len(te) == len(x_test):
            return tr.astype(np.float32), te.astype(np.float32)

    n_groups = 8
    boundaries = np.linspace(0, n_trees, n_groups + 1, dtype=int)

    train_preds = []
    test_preds = []
    for i in range(n_groups):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        if end <= start:
            continue
        p_tr = xgb_model.predict_proba(x_train, iteration_range=(start, end))[:, 1].astype(np.float32)
        p_te = xgb_model.predict_proba(x_test, iteration_range=(start, end))[:, 1].astype(np.float32)
        train_preds.append(p_tr)
        test_preds.append(p_te)

    tr_stack = np.vstack(train_preds)
    te_stack = np.vstack(test_preds)
    tr_u = np.std(tr_stack, axis=0).astype(np.float32)
    te_u = np.std(te_stack, axis=0).astype(np.float32)

    np.save(tr_path, tr_u)
    np.save(te_path, te_u)
    return tr_u, te_u


def compute_leaf_distance(cfg: dict, leaf_train_all: np.ndarray, leaf_test_all: np.ndarray, seed: int):
    cache_dir = full_path(cfg["paths"]["cache_dir"])
    tr_path = cache_dir / "leaf_distance_train.npy"
    te_path = cache_dir / "leaf_distance_test.npy"

    if tr_path.exists() and te_path.exists():
        tr = np.load(tr_path)
        te = np.load(te_path)
        if len(tr) == len(leaf_train_all) and len(te) == len(leaf_test_all):
            return tr.astype(np.float32), te.astype(np.float32)

    sample_n = int(cfg["processing"].get("uncertainty_train_sample", 200000))
    idx = stable_sample_indices(len(leaf_train_all), sample_n, seed)
    leaf_sample = leaf_train_all[idx]

    n_clusters = min(int(cfg["processing"].get("inducing_points", 256)), 256, len(leaf_sample))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=4096,
        n_init=5,
    )
    kmeans.fit(leaf_sample)

    tr_dist = kmeans.transform(leaf_train_all).min(axis=1).astype(np.float32)
    te_dist = kmeans.transform(leaf_test_all).min(axis=1).astype(np.float32)

    np.save(tr_path, tr_dist)
    np.save(te_path, te_dist)
    return tr_dist, te_dist


def main():
    cfg = load_config()
    seed = int(cfg["seed"])

    x_train_raw = np.load(full_path(cfg["paths"]["x_train_raw"]))
    x_test_raw = np.load(full_path(cfg["paths"]["x_test_raw"]))

    bundle = joblib.load(full_path(cfg["paths"]["models_dir"]) / "feature_bundle.joblib")
    safe_idx = bundle["safe_idx"]

    x_train = x_train_raw[:, safe_idx].astype(np.float32)
    x_test = x_test_raw[:, safe_idx].astype(np.float32)

    y_train = load_train_labels(cfg, expected_n=len(x_train))

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(full_path(cfg["paths"]["models_dir"]) / "xgb.json"))

    # cached leaf embeddings
    leaf_train_all, leaf_test_all = get_leaf_embeddings(cfg, xgb_model, x_train, x_test)
    n_trees = int(leaf_train_all.shape[1])

    # uncertainty component A: tree disagreement
    tr_dis, te_dis = compute_tree_disagreement(cfg, xgb_model, x_train, x_test, n_trees)

    # uncertainty component B: leaf distance
    tr_dist, te_dist = compute_leaf_distance(cfg, leaf_train_all, leaf_test_all, seed)

    # normalize jointly for comparable scale
    dis_all = np.concatenate([tr_dis, te_dis])
    dist_all = np.concatenate([tr_dist, te_dist])
    dis_all_u = minmax_unit(dis_all)
    dist_all_u = minmax_unit(dist_all)

    tr_dis_u = dis_all_u[: len(tr_dis)]
    te_dis_u = dis_all_u[len(tr_dis):]
    tr_dist_u = dist_all_u[: len(tr_dist)]
    te_dist_u = dist_all_u[len(tr_dist):]

    # blend: disagreement + distance
    w_dis, w_dist = 0.6, 0.4
    train_u = (w_dis * tr_dis_u + w_dist * tr_dist_u).astype(np.float32)
    test_u = (w_dis * te_dis_u + w_dist * te_dist_u).astype(np.float32)

    np.save(full_path(cfg["paths"]["test_variance"]), test_u)

    print("checkpoint_4")
    print("train_uncertainty_mean", float(train_u.mean()))
    print("test_uncertainty_mean", float(test_u.mean()))
    print("train_uncertainty_std", float(train_u.std()))
    print("test_uncertainty_std", float(test_u.std()))


if __name__ == "__main__":
    main()
