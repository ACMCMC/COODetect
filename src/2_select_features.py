import numpy as np

from common import full_path, load_config, load_json, save_json


def canon_lang(x: str) -> str:
    low = str(x).strip().lower()
    if "python" in low:
        return "Python"
    if "java" in low:
        return "Java"
    if "c++" in low or low == "cpp":
        return "C++"
    return "OTHER"


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 1.0

    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    vals = np.concatenate([a_sorted, b_sorted])

    cdf_a = np.searchsorted(a_sorted, vals, side="right") / float(a_sorted.size)
    cdf_b = np.searchsorted(b_sorted, vals, side="right") / float(b_sorted.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def main():
    cfg = load_config()

    invariant_path = full_path(cfg["paths"].get("x_train_invariant", cfg["paths"]["x_train_raw"]))
    if invariant_path.exists():
        x_train = np.load(invariant_path)
    else:
        x_train = np.load(full_path(cfg["paths"]["x_train_raw"]))
    langs_raw = np.load(full_path(cfg["paths"]["train_languages"]), allow_pickle=True)
    feature_meta = load_json(full_path("cache/feature_names.json"))
    feature_names = feature_meta["feature_names"]

    langs = np.array([canon_lang(x) for x in langs_raw], dtype=object)

    mask_py = langs == "Python"
    mask_java = langs == "Java"
    mask_cpp = langs == "C++"

    idx_py = np.where(mask_py)[0]
    idx_java = np.where(mask_java)[0]
    idx_cpp = np.where(mask_cpp)[0]

    pairs = [
        ("py_java", idx_py, idx_java),
        ("py_cpp", idx_py, idx_cpp),
        ("java_cpp", idx_java, idx_cpp),
    ]

    dmax_values = []
    d_pair_values = {"py_java": [], "py_cpp": [], "java_cpp": []}

    for feat_idx in range(x_train.shape[1]):
        col = x_train[:, feat_idx].astype(np.float32)
        ds = []
        for pair_name, idx_a, idx_b in pairs:
            d = ks_statistic(col[idx_a], col[idx_b])
            d_pair_values[pair_name].append(float(d))
            ds.append(float(d))
        dmax_values.append(float(max(ds)))

    dmax_arr = np.asarray(dmax_values, dtype=np.float32)
    keep = np.where(dmax_arr < 0.15)[0]

    if keep.size < 10:
        keep = np.argsort(dmax_arr)[:10]

    keep = np.asarray(keep, dtype=np.int64)
    selected_names = [feature_names[i] for i in keep.tolist()]

    out = {
        "selected_indices": keep.tolist(),
        "selected_feature_names": selected_names,
        "pairwise_threshold": 0.15,
        "pair_order": ["py_java", "py_cpp", "java_cpp"],
        "dmax": dmax_arr.tolist(),
        "pairwise_d": d_pair_values,
        "num_features_before": int(x_train.shape[1]),
        "num_features_after": int(keep.size),
        "strategy": "lolo_pairwise_ks_dmax_threshold_with_top10_fallback",
    }

    save_json(out, full_path(cfg["paths"]["selected_features"]))

    print("checkpoint_2")
    print("features_before", int(x_train.shape[1]))
    print("features_after", int(keep.size))
    print("selected_features")
    for name in selected_names:
        print(name)


if __name__ == "__main__":
    main()
