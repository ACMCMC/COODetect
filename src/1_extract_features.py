import bz2
import gc
import lzma
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from common import chunk_slices, full_path, load_config, save_json


KW_LIST = ["if", "else", "for", "while", "return", "def", "class", "void", "int", "import"]
KW_SET = set(KW_LIST)
TOP_NGRAMS = 50
WINDOW_SIZE = 100

TOKEN_RE = re.compile(
    r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\\b\\d+\\b|[+\-*/%=<>!&|^~]+|[{}\[\]()]|\\b[a-zA-Z_]\\w*\\b",
    re.DOTALL,
)


def to_float32(arr_like):
    return np.asarray(arr_like, dtype=np.float32)


def shannon_entropy_bytes(byte_values: np.ndarray) -> float:
    if byte_values.size == 0:
        return 0.0
    counts = np.bincount(byte_values, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / float(byte_values.size)
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def entropy_variance(text: str, window_size: int = WINDOW_SIZE) -> float:
    data = text.encode("utf-8", errors="ignore")
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size < window_size:
        return 0.0

    vals = []
    for idx in range(0, arr.size - window_size + 1):
        vals.append(shannon_entropy_bytes(arr[idx : idx + window_size]))
    if len(vals) <= 1:
        return 0.0
    return float(np.var(np.asarray(vals, dtype=np.float32)))


def classify_lexeme(tok: str) -> str:
    if not tok:
        return "UNK"
    first = tok[0]

    if first == '"' or first == "'":
        return "STR"
    if tok[0].isdigit() and tok.isdigit():
        return "NUM"
    if re.fullmatch(r"[+\-*/%=<>!&|^~]+", tok):
        return "OP"
    if re.fullmatch(r"[{}\[\]()]", tok):
        return "BLK"
    if re.fullmatch(r"[a-zA-Z_]\w*", tok):
        low = tok.lower()
        if low in KW_SET:
            return "KW"
        return "ID"
    return "UNK"


def lex_types(text: str) -> list[str]:
    out = []
    for match in TOKEN_RE.finditer(text):
        t = match.group(0)
        out.append(classify_lexeme(t))
    return out


def trigram_counts(type_seq: list[str]) -> Counter:
    cnt = Counter()
    if len(type_seq) < 3:
        return cnt
    for idx in range(len(type_seq) - 2):
        key = f"{type_seq[idx]}-{type_seq[idx + 1]}-{type_seq[idx + 2]}"
        cnt[key] += 1
    return cnt


def build_top_trigrams(codes: list[str], top_k: int) -> list[str]:
    all_counter = Counter()
    for code in tqdm(codes, desc="[train] trigram_vocab", unit="row"):
        seq = lex_types(code)
        all_counter.update(trigram_counts(seq))
    top = [name for name, _ in all_counter.most_common(top_k)]
    if len(top) < top_k:
        top = top + [f"PAD_TRI_{i}" for i in range(top_k - len(top))]
    return top[:top_k]


def info_density_features(text: str) -> tuple[float, float]:
    n_text = max(len(text), 1)
    raw = text.encode("utf-8", errors="ignore")
    lz_ratio = len(lzma.compress(raw)) / float(n_text)
    bz_ratio = len(bz2.compress(raw)) / float(n_text)
    return float(lz_ratio), float(bz_ratio)


def entropy_features(text: str) -> tuple[float, float]:
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) == 0:
        return 0.0, 0.0
    byte_arr = np.frombuffer(raw, dtype=np.uint8)
    ent = shannon_entropy_bytes(byte_arr)
    ent_var = entropy_variance(text, window_size=WINDOW_SIZE)
    return float(ent), float(ent_var)


def visual_rhythm_features(text: str) -> tuple[float, float, float]:
    lines = text.splitlines()
    n_lines = max(len(lines), 1)

    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) == 0:
        indentation_std = 0.0
    else:
        indents = [len(line) - len(line.lstrip(" ")) for line in non_empty]
        indentation_std = float(np.std(np.asarray(indents, dtype=np.float32)))

    line_lengths = [len(line) for line in lines] if lines else [0]
    line_length_std = float(np.std(np.asarray(line_lengths, dtype=np.float32)))

    empty_lines = sum(1 for line in lines if line.strip() == "")
    vertical_density = float(empty_lines / float(n_lines))

    return indentation_std, line_length_std, vertical_density


def symbolic_ngram_features(text: str, trigram_to_idx: dict[str, int], vocab_size: int) -> np.ndarray:
    seq = lex_types(text)
    tri = trigram_counts(seq)
    total = max(sum(tri.values()), 1)
    vec = np.zeros(vocab_size, dtype=np.float32)
    for key, value in tri.items():
        idx = trigram_to_idx.get(key)
        if idx is not None:
            vec[idx] = float(value) / float(total)
    return vec


def extract_one(code: str, trigram_to_idx: dict[str, int], vocab_size: int) -> np.ndarray:
    text = code if isinstance(code, str) else ""

    lz_ratio, bz_ratio = info_density_features(text)
    entropy, entropy_var = entropy_features(text)
    indentation_std, line_length_std, vertical_density = visual_rhythm_features(text)
    tri_vec = symbolic_ngram_features(text, trigram_to_idx, vocab_size)

    core = to_float32(
        [
            lz_ratio,
            bz_ratio,
            entropy,
            entropy_var,
            indentation_std,
            line_length_std,
            vertical_density,
        ]
    )
    out = np.concatenate([core, tri_vec], axis=0)
    return out.astype(np.float32)


def process_split(df: pd.DataFrame, split_name: str, cfg: dict, trigram_vocab: list[str]):
    cache_dir = full_path(cfg["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunk_parts = int(cfg["processing"]["chunk_parts"])
    n_jobs = int(cfg["processing"]["n_jobs"])

    code_col = cfg["columns"]["train_code"] if split_name == "train" else cfg["columns"]["test_code"]
    chunk_prefix = "train_chunk" if split_name == "train" else "test_chunk"
    final_path = full_path(cfg["paths"]["x_train_raw"] if split_name == "train" else cfg["paths"]["x_test_raw"])
    invariant_path = full_path(cfg["paths"]["x_train_invariant"] if split_name == "train" else cfg["paths"]["x_test_invariant"])
    force_rebuild = os.getenv("FORCE_REBUILD", "0") == "1"

    if final_path.exists() and invariant_path.exists() and not force_rebuild:
        print(f"[{split_name}] reuse final cache: {final_path}")
        return

    trigram_to_idx = {name: idx for idx, name in enumerate(trigram_vocab)}
    vocab_size = len(trigram_vocab)

    slices = chunk_slices(len(df), chunk_parts)
    tmp_paths = []
    print(f"[{split_name}] rows={len(df)}, chunks={len(slices)}, n_jobs={n_jobs}")

    for idx, (start, end) in tqdm(enumerate(slices), total=len(slices), desc=f"[{split_name}] chunks", unit="chunk"):
        tmp_path = cache_dir / f"{chunk_prefix}_{idx:04d}.npy"
        if tmp_path.exists() and not force_rebuild:
            tmp_paths.append(tmp_path)
            continue

        chunk_df = df.iloc[start:end]
        codes = chunk_df[code_col].fillna("").astype(str).tolist()

        feats = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(extract_one)(code, trigram_to_idx, vocab_size)
            for code in codes
        )
        arr = np.vstack(feats).astype(np.float32)
        np.save(tmp_path, arr)
        tmp_paths.append(tmp_path)

        del chunk_df, codes, feats, arr
        gc.collect()

    merged = np.concatenate(
        [np.load(path).astype(np.float32) for path in tqdm(tmp_paths, desc=f"[{split_name}] merge", unit="file")],
        axis=0,
    )
    np.save(final_path, merged)
    np.save(invariant_path, merged)

    print(f"[{split_name}] done shape={merged.shape}, dtype={merged.dtype}")


def main():
    cfg = load_config()
    np.random.seed(int(cfg["seed"]))

    train_df = pd.read_parquet(full_path(cfg["paths"]["train_parquet"]))
    test_df = pd.read_parquet(full_path(cfg["paths"]["test_parquet"]))

    sample_n = int(os.getenv("SAMPLE_N_ROWS", "0"))
    if sample_n > 0:
        train_df = train_df.iloc[:sample_n].copy()
        test_df = test_df.iloc[:sample_n].copy()
        print(f"sample_mode rows={sample_n}")

    train_codes = train_df[cfg["columns"]["train_code"]].fillna("").astype(str).tolist()
    top_trigrams = build_top_trigrams(train_codes, TOP_NGRAMS)

    process_split(train_df, "train", cfg, top_trigrams)
    process_split(test_df, "test", cfg, top_trigrams)

    x_train_invariant = full_path(cfg["paths"]["x_train_invariant"])
    x_test_invariant = full_path(cfg["paths"]["x_test_invariant"])
    x_train_raw = full_path(cfg["paths"]["x_train_raw"])
    x_test_raw = full_path(cfg["paths"]["x_test_raw"])
    if not x_train_invariant.exists() and x_train_raw.exists():
        np.save(x_train_invariant, np.load(x_train_raw).astype(np.float32))
    if not x_test_invariant.exists() and x_test_raw.exists():
        np.save(x_test_invariant, np.load(x_test_raw).astype(np.float32))

    y_train = train_df[cfg["columns"]["train_label"]].to_numpy(dtype=np.int64)
    train_lang = train_df[cfg["columns"]["train_language"]].astype(str).to_numpy()
    test_ids = test_df[cfg["columns"]["test_id"]].to_numpy()

    np.save(full_path(cfg["paths"]["y_train"]), y_train)
    np.save(full_path(cfg["paths"]["train_languages"]), train_lang)
    np.save(full_path(cfg["paths"]["test_ids"]), test_ids)

    feature_names = [
        "lzma_ratio",
        "bz2_ratio",
        "entropy",
        "entropy_variance",
        "indentation_std",
        "line_length_std",
        "vertical_density",
    ] + [f"tri_{name}" for name in top_trigrams]

    save_json({"feature_names": feature_names, "top_symbolic_trigrams": top_trigrams}, full_path("cache/feature_names.json"))

    x_train = np.load(full_path(cfg["paths"]["x_train_raw"]))
    x_test = np.load(full_path(cfg["paths"]["x_test_raw"]))

    print("checkpoint_1")
    print("x_train_shape", x_train.shape)
    print("x_test_shape", x_test.shape)
    print("x_train_dtype", x_train.dtype)
    print("x_test_dtype", x_test.dtype)


if __name__ == "__main__":
    main()
