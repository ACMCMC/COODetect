import re
import numpy as np

FEATURE_NAMES = [
    "ast_u_ctrl_density",
    "ast_u_return_density",
    "ast_u_call_density",
    "ast_u_assign_density",
    "ast_u_decl_density",
    "ast_u_loop_density",
    "ast_u_branch_density",
]

CTRL_RE = re.compile(r"\b(if|else|switch|case|when)\b", re.IGNORECASE)
RET_RE = re.compile(r"\b(return|yield)\b", re.IGNORECASE)
CALL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\(")
ASSIGN_RE = re.compile(r"(?<![=!<>])=(?!=)")
DECL_RE = re.compile(r"\b(class|def|function|func|interface|struct|enum|namespace|module)\b", re.IGNORECASE)
LOOP_RE = re.compile(r"\b(for|while|foreach|do)\b", re.IGNORECASE)
BRANCH_RE = re.compile(r"\b(try|except|catch|finally|throw|raise)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|&&|\|\||\+\+|--|=>|[+\-*/%=&|^~<>!]")


def extract_features(code: str) -> np.ndarray:
    text = code if isinstance(code, str) else ""
    n_tokens = max(len(TOKEN_RE.findall(text)), 1)

    out = np.array([
        len(CTRL_RE.findall(text)) / n_tokens,
        len(RET_RE.findall(text)) / n_tokens,
        len(CALL_RE.findall(text)) / n_tokens,
        len(ASSIGN_RE.findall(text)) / n_tokens,
        len(DECL_RE.findall(text)) / n_tokens,
        len(LOOP_RE.findall(text)) / n_tokens,
        len(BRANCH_RE.findall(text)) / n_tokens,
    ], dtype=np.float32)
    return out
