import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import full_path, load_config

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def canon_label(label: str) -> str:
    low = str(label).strip().lower()
    if "python" in low:
        return "Python"
    if "java" in low and "javascript" not in low:
        return "Java"
    if "c++" in low or low == "cpp":
        return "C++"
    if low in {"c", "ansi c"}:
        return "C"
    if "c#" in low or "c sharp" in low or "csharp" in low:
        return "C#"
    if "go" == low or "golang" in low:
        return "Go"
    if "php" in low:
        return "PHP"
    if "javascript" in low or low in {"js", "nodejs", "node.js"}:
        return "JavaScript"
    return "Unknown"


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def find_guesslang_data_paths() -> tuple[Path, Path]:
    data_candidates = []
    for base in sys.path:
        p = Path(base)
        data_dir = p / "guesslang" / "data"
        data_candidates.append(data_dir)

    for data_dir in data_candidates:
        model_dir = data_dir / "model"
        langs_file = data_dir / "languages.json"
        if model_dir.exists() and langs_file.exists():
            return model_dir, langs_file

    raise RuntimeError("guesslang data/model not found in sys.path; install guesslang-experimental in the active venv")


class GuessLangRuntime:
    def __init__(self):
        model_dir, langs_file = find_guesslang_data_paths()
        self.model = tf.saved_model.load(str(model_dir))
        lang_map = json.loads(langs_file.read_text())
        self.ext_to_lang = {ext: name for name, ext in lang_map.items()}

    def language_names(self, codes: list[str]) -> list[str]:
        texts = [code if isinstance(code, str) else "" for code in codes]
        pred = self.model.signatures["serving_default"](tf.constant(texts))

        batch_classes = pred["classes"].numpy().tolist()
        batch_scores = pred["scores"].numpy().tolist()
        out = []
        for text, cls_row, score_row in zip(texts, batch_classes, batch_scores):
            if text.strip() == "":
                out.append("Unknown")
                continue
            if len(cls_row) == 0:
                out.append("Unknown")
                continue
            top_idx = int(np.argmax(np.asarray(score_row, dtype=np.float32)))
            top_ext = cls_row[top_idx].decode()
            out.append(self.ext_to_lang.get(top_ext, "Unknown"))
        return out


def main():
    cfg = load_config()
    batch_size = int(os.getenv("GUESSLANG_BATCH", "8000"))
    force_rebuild = os.getenv("FORCE_REBUILD", "0") == "1"
    out_path = full_path(cfg["paths"].get("test_languages_guesslang", "cache/test_languages_guesslang.npy"))
    partial_path = full_path("cache/test_languages_guesslang.partial.npy")

    test_df = pd.read_parquet(full_path(cfg["paths"]["test_parquet"]))
    code_col = cfg["columns"]["test_code"]
    codes = test_df[code_col].fillna("").astype(str).tolist()

    if out_path.exists() and not force_rebuild:
        cached = np.load(out_path, allow_pickle=True)
        if len(cached) == len(codes):
            uniq, cnt = np.unique(cached.astype(object), return_counts=True)
            print("checkpoint_detect_languages")
            print("saved", out_path)
            print("cache_reuse", True)
            for k, v in zip(uniq.tolist(), cnt.tolist()):
                print(k, int(v))
            return

    guess = GuessLangRuntime()
    out = []
    if partial_path.exists() and not force_rebuild:
        partial = np.load(partial_path, allow_pickle=True)
        if len(partial) <= len(codes):
            out = partial.astype(object).tolist()

    start_idx = len(out)
    remaining = codes[start_idx:]

    for batch in tqdm(
        batched(remaining, batch_size),
        total=(len(remaining) + batch_size - 1) // batch_size if len(remaining) > 0 else 0,
        desc="guesslang_batches",
        unit="batch",
    ):
        preds = guess.language_names(batch)
        out.extend([canon_label(pred) for pred in preds])
        np.save(partial_path, np.asarray(out, dtype=object))

    arr = np.asarray(out, dtype=object)
    if len(arr) != len(codes):
        raise ValueError(f"detect_languages size mismatch: expected={len(codes)} got={len(arr)}")

    np.save(out_path, arr)
    if partial_path.exists():
        partial_path.unlink()

    uniq, cnt = np.unique(arr, return_counts=True)
    print("checkpoint_detect_languages")
    print("saved", out_path)
    print("cache_reuse", False)
    for k, v in zip(uniq.tolist(), cnt.tolist()):
        print(k, int(v))


if __name__ == "__main__":
    main()
