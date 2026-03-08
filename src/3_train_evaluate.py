import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score

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


def make_model(cfg: dict, seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=900,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        tree_method=str(cfg["processing"]["xgb_gpu_tree_method"]),
        device=str(cfg["processing"]["xgb_device"]),
    )


def main():
    cfg = load_config()
    seed = int(cfg["seed"])

    x_train_path = full_path(cfg["paths"].get("x_train_invariant", cfg["paths"]["x_train_raw"]))
    x_test_path = full_path(cfg["paths"].get("x_test_invariant", cfg["paths"]["x_test_raw"]))
    if not x_train_path.exists():
        x_train_path = full_path(cfg["paths"]["x_train_raw"])
    if not x_test_path.exists():
        x_test_path = full_path(cfg["paths"]["x_test_raw"])

    x_train_raw = np.load(x_train_path)
    x_test_raw = np.load(x_test_path)
    y_train = np.load(full_path(cfg["paths"]["y_train"]))
    langs_raw = np.load(full_path(cfg["paths"]["train_languages"]), allow_pickle=True)

    selected = load_json(full_path(cfg["paths"]["selected_features"]))
    safe_idx = np.array(selected["selected_indices"], dtype=np.int64)
    safe_names = selected["selected_feature_names"]

    x_train = x_train_raw[:, safe_idx].astype(np.float32)
    x_test = x_test_raw[:, safe_idx].astype(np.float32)

    langs = np.array([canon_lang(x) for x in langs_raw], dtype=object)

    fold_specs = [
        ("cpp", "C++", ["Python", "Java"], "xgb_fold_cpp.json"),
        ("java", "Java", ["Python", "C++"], "xgb_fold_java.json"),
        ("py", "Python", ["Java", "C++"], "xgb_fold_py.json"),
    ]

    models_dir = full_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    oof_pred = np.zeros(len(x_train), dtype=np.float32)
    fold_f1 = {}
    test_preds = []

    for fold_tag, holdout_lang, train_langs, model_name in fold_specs:
        train_mask = np.isin(langs, train_langs)
        val_mask = langs == holdout_lang

        x_fit = x_train[train_mask]
        y_fit = y_train[train_mask]
        x_val = x_train[val_mask]
        y_val = y_train[val_mask]

        model = make_model(cfg, seed)
        model.fit(x_fit, y_fit)

        p_val = model.predict_proba(x_val)[:, 1].astype(np.float32)
        p_test = model.predict_proba(x_test)[:, 1].astype(np.float32)

        oof_pred[val_mask] = p_val
        test_preds.append(p_test)

        np.save(full_path(f"cache/ood_preds_fold_{fold_tag}.npy"), p_val.astype(np.float32))

        y_hat = (p_val >= 0.5).astype(np.int64)
        f1 = f1_score(y_val, y_hat, average="macro")
        fold_f1[fold_tag] = float(f1)

        model.save_model(str(models_dir / model_name))

    ensemble_test = np.mean(np.vstack(test_preds), axis=0).astype(np.float32)

    np.save(full_path(cfg["paths"]["base_train_probs"]), oof_pred.astype(np.float32))
    np.save(full_path(cfg["paths"]["base_test_probs"]), ensemble_test.astype(np.float32))

    joblib.dump(
        {
            "safe_idx": safe_idx,
            "safe_names": safe_names,
            "lolo_fold_models": ["xgb_fold_cpp.json", "xgb_fold_java.json", "xgb_fold_py.json"],
        },
        models_dir / "feature_bundle.joblib",
    )

    y_oof = (oof_pred >= 0.5).astype(np.int64)
    macro_oof = f1_score(y_train, y_oof, average="macro")

    save_json(
        {
            "fold_macro_f1": fold_f1,
            "macro_f1_oof": float(macro_oof),
            "train_prior": float(y_train.mean()),
            "selected_features": int(len(safe_idx)),
            "strategy": "lolo_xgb_ensemble",
        },
        full_path("cache/phase3_metrics.json"),
    )

    print("checkpoint_3")
    print("fold_macro_f1", fold_f1)
    print("macro_f1_oof", float(macro_oof))


if __name__ == "__main__":
    main()
