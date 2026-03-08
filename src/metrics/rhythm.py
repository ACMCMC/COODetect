import numpy as np

FEATURE_NAMES = [
    "rh_line_len_mean_norm",
    "rh_line_len_std_norm",
    "rh_indent_mean_norm",
    "rh_indent_std_norm",
    "rh_indent_change_ratio",
    "rh_blank_ratio",
    "rh_nonascii_ratio",
]


def extract_features(code: str) -> np.ndarray:
    text = code if isinstance(code, str) else ""
    lines = text.splitlines()
    if len(lines) == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    lens = np.array([len(x) for x in lines], dtype=np.float32)
    inds = np.array([len(x) - len(x.lstrip(' \t')) for x in lines], dtype=np.float32)

    n_lines = max(len(lines), 1)
    n_chars = max(len(text), 1)

    diffs = np.abs(np.diff(inds)) if len(inds) > 1 else np.array([0.0], dtype=np.float32)

    out = np.array([
        float(np.mean(lens) / 200.0),
        float(np.std(lens) / 200.0),
        float(np.mean(inds) / 40.0),
        float(np.std(inds) / 40.0),
        float(np.mean(diffs > 0)),
        float(np.mean([1.0 if x.strip() == '' else 0.0 for x in lines])),
        float(sum(1 for ch in text if ord(ch) > 127) / n_chars),
    ], dtype=np.float32)
    return out
