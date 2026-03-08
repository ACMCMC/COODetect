import re
import numpy as np

FEATURE_NAMES = [
    "sym_identifier_density",
    "sym_number_density",
    "sym_string_density",
    "sym_operator_density",
    "sym_keyword_density",
    "sym_control_density",
    "sym_entropy_norm",
]

TOKEN_RE = re.compile(
    r"'([^'\\]|\\.)*'|\"([^\"\\]|\\.)*\"|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||\+\+|--|=>|[+\-*/%=&|^~<>!]"
)
ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")
STR_RE = re.compile(r"^'([^'\\]|\\.)*'$|^\"([^\"\\]|\\.)*\"$")
OP_RE = re.compile(r"^(==|!=|<=|>=|&&|\|\||\+\+|--|=>|[+\-*/%=&|^~<>!])$")

KEYWORDS = {
    "if", "else", "for", "while", "return", "class", "def", "function", "try", "except", "catch",
    "switch", "case", "break", "continue", "import", "from", "new", "null", "true", "false"
}
CONTROL = {"if", "else", "for", "while", "switch", "case", "break", "continue", "return", "try", "except", "catch"}


def extract_features(code: str) -> np.ndarray:
    text = code if isinstance(code, str) else ""
    toks = TOKEN_RE.findall(text)
    # regex has groups, so rebuild robustly
    tokens = [m.group(0) for m in TOKEN_RE.finditer(text)]
    n = max(len(tokens), 1)

    id_count = 0
    num_count = 0
    str_count = 0
    op_count = 0
    kw_count = 0
    ctl_count = 0

    for tok in tokens:
        low = tok.lower()
        if STR_RE.match(tok):
            str_count += 1
            continue
        if NUM_RE.match(tok):
            num_count += 1
            continue
        if OP_RE.match(tok):
            op_count += 1
            continue
        if ID_RE.match(tok):
            id_count += 1
            if low in KEYWORDS:
                kw_count += 1
            if low in CONTROL:
                ctl_count += 1

    # token-type entropy on coarse symbols
    buckets = np.array([id_count, num_count, str_count, op_count, kw_count], dtype=np.float64)
    probs = buckets / buckets.sum() if buckets.sum() > 0 else np.array([1.0], dtype=np.float64)
    probs = probs[probs > 0]
    ent = -np.sum(probs * np.log2(probs))
    max_ent = np.log2(len(probs)) if len(probs) > 1 else 1.0
    entropy_norm = float(ent / max_ent)

    out = np.array([
        id_count / n,
        num_count / n,
        str_count / n,
        op_count / n,
        kw_count / n,
        ctl_count / n,
        entropy_norm,
    ], dtype=np.float32)
    return out
