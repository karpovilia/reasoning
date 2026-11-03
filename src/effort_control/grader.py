#!/usr/bin/env python3
r"""
Grade gpt-oss-20b cleaned trajectories: add is_correct field.

Two normalization strategies:
  - normalize_text  (hotpotqa, musique): lowercase, strip punctuation, remove
                    articles a/an/the, collapse whitespace.
  - normalize_math  (math500, gpqa_diamond, hle): all of the above PLUS LaTeX
                    symbol stripping (\frac, \sqrt, \pi, …) and final removal
                    of all whitespace so spacing differences are ignored.

is_correct = True when:
  1. normalize(ca) == normalize(ma)   [exact match after normalization]
  2. Both normalize to a parseable number (int/float/fraction) and are equal
     within floating-point precision.
is_correct = False when model_answer is None or no rule fires.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------

_UNICODE_SPACES = re.compile(r"[   ​⁠　]+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")  # non-alphanumeric, non-space


def normalize_text(s: str) -> str:
    """Normalization for text / QA datasets (hotpotqa, musique)."""
    s = _UNICODE_SPACES.sub(" ", s).strip()
    s = s.lower()
    s = _PUNCT.sub("", s)          # remove all punctuation
    s = _ARTICLES.sub("", s)       # remove articles
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_math(s: str) -> str:
    """
    Normalization for math / science datasets (math500, gpqa_diamond, hle).

    Order matters: LaTeX structural rules first, then symbol substitutions,
    then case-folding and whitespace removal.
    """
    s = _UNICODE_SPACES.sub(" ", s).strip()

    # --- Strip outermost LaTeX display/inline math delimiters ---------------
    for pat in (
        r"^\$\$(.+)\$\$$",
        r"^\$(.+)\$$",
        r"^\\\[(.+)\\\]$",
        r"^\\\((.+)\\\)$",
    ):
        m = re.match(pat, s, re.DOTALL)
        if m:
            s = m.group(1).strip()
            break

    # --- Structural LaTeX rules (modify the expression) ---------------------

    # \text{content} / \mbox{content} → content
    for tex_cmd in (r"\\text", r"\\mbox"):
        while re.search(tex_cmd + r"\{", s):
            s = re.sub(tex_cmd + r"\{([^{}]*)\}", r"\1", s)

    # \left / \right → remove (keep the bracket that follows)
    s = re.sub(r"\\(?:left|right)\s*", "", s)

    # \mathbb{X} / \mathbf{X} / \mathrm{X} → X
    s = re.sub(r"\\math(?:bb|bf|rm|cal|scr)\{([A-Za-z]+)\}", r"\1", s)

    # Subscript braces: _{X} → _X  (_{foo} → _foo)
    s = re.sub(r"_\{([^{}]+)\}", r"_\1", s)

    # \begin{pmatrix/bmatrix/vmatrix}…\end{...} → strip env, keep contents
    # NOTE: must run BEFORE the spacing-command rule so that \\ row separators
    # are not partially consumed before they can be converted to commas.
    s = re.sub(r"\\begin\{[^}]+\}", "", s)
    s = re.sub(r"\\end\{[^}]+\}", "", s)
    s = re.sub(r"\\\\", ",", s)    # row separator → comma
    s = re.sub(r"&", ",", s)       # column separator → comma

    # LaTeX spacing commands → remove  (\, \! \; \: \ )
    s = re.sub(r"\\[,!;: ]", "", s)

    # \sqrt{X} → sqrtX  (no parens — consistent with bare √X)
    # Apply BEFORE \frac so nested \frac{3\sqrt{3}}{4} works correctly.
    for _ in range(2):
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt\1", s)
    s = re.sub(r"\\sqrt\s*(\d+)", r"sqrt\1", s)   # \sqrt2 / \sqrt 2
    # Unicode √ → sqrt (bare, without parens)
    s = re.sub(r"√\{([^{}]+)\}", r"sqrt\1", s)
    s = s.replace("√", "sqrt")

    # \frac / \dfrac / \tfrac {a}{b} → a/b  (two passes for nesting)
    for _ in range(2):
        s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    # \frac without braces: single-char / single-digit variants
    s = re.sub(r"\\[dt]?frac(\d)\{([^{}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}(\d)", r"\1/\2", s)
    s = re.sub(r"\\[dt]?frac(\d)(\d)", r"\1/\2", s)
    # \frac with space before digits (e.g. \frac 59 → 5/9)
    s = re.sub(r"\\[dt]?frac\s+(\d)\s*(\d)", r"\1/\2", s)

    # --- Symbol substitutions -----------------------------------------------

    # Greek letters
    _GREEK = {
        "pi": "pi", "alpha": "alpha", "beta": "beta", "gamma": "gamma",
        "delta": "delta", "epsilon": "epsilon", "zeta": "zeta",
        "eta": "eta", "theta": "theta", "lambda": "lambda", "mu": "mu",
        "nu": "nu", "xi": "xi", "rho": "rho", "sigma": "sigma",
        "tau": "tau", "phi": "phi", "chi": "chi", "psi": "psi",
        "omega": "omega",
    }
    for latex_name, ascii_name in _GREEK.items():
        s = re.sub(r"\\" + latex_name + r"\b", ascii_name, s)
    s = s.replace("π", "pi")

    # \infty / ∞ → inf
    s = re.sub(r"\\infty\b", "inf", s)
    s = s.replace("∞", "inf")

    # Trig / math functions: strip backslash
    for fn in (
        "sin", "cos", "tan", "cot", "sec", "csc",
        "arcsin", "arccos", "arctan",
        "log", "ln", "exp",
        "max", "min", "gcd", "lcm", "det", "tr", "deg",
        "lim", "sup", "inf",
    ):
        s = re.sub(r"\\" + fn + r"\b", fn, s)

    # Multiplication / dot operators → *
    s = re.sub(r"\\cdot\b", "*", s)
    s = re.sub(r"\\times\b", "*", s)
    s = s.replace("×", "*").replace("·", "*")

    # \pm → ±,  \mp → ∓ (keep as-is; just strip backslash)
    s = re.sub(r"\\pm\b", "pm", s)
    s = re.sub(r"\\mp\b", "mp", s)

    # Degree: ^\circ → empty (strip degree symbol; 90° ≡ 90 for comparison)
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = re.sub(r"\\circ\b", "", s)
    s = s.replace("°", "")

    # \approx / ≈ → = (approximate equality treated as equality for grading)
    s = re.sub(r"\\approx\b", "=", s)
    s = s.replace("≈", "=")

    # Set/interval union ∪ / \cup → remove (treated as separator like comma)
    s = re.sub(r"\\cup\b", "", s)
    s = s.replace("∪", "")

    # Dollar-sign prefix: \$36 → 36 (monetary amounts)
    s = re.sub(r"\\\$", "", s)
    s = s.replace("$", "")

    # Strip remaining backslash-command tokens (unknown commands → remove)
    s = re.sub(r"\\[a-zA-Z]+\b\s*", "", s)

    # --- Final cleanup -------------------------------------------------------

    s = s.lower()
    # Remove all whitespace so spacing is irrelevant
    s = re.sub(r"\s+", "", s)
    # Strip outer parentheses only when the opening paren matches the closing
    # one — i.e., they are a single wrapping pair, not multi-interval notation
    # like (-inf,2)(3,inf).
    def _strip_outer_parens(t: str) -> str:
        while t.startswith("(") and t.endswith(")"):
            depth = 0
            stripped = False
            for i, ch in enumerate(t):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        if i == len(t) - 1:
                            t = t[1:-1]
                            stripped = True
                        else:
                            return t  # outer ( closes before end → multi-interval
                        break
            # Unbalanced string (depth never returned to 0 at a closing paren):
            # the scan changed nothing, so a plain `continue` would spin forever
            # on inputs like "((a)". Bail out when no progress was made.
            if not stripped:
                break
        return t

    s = _strip_outer_parens(s)

    return s


# ---------------------------------------------------------------------------
# Numeric comparison
# ---------------------------------------------------------------------------

_FRACTION_RE = re.compile(r"^(-?\d+)/(\d+)$")


def _to_number(s: str) -> float | None:
    """
    Try to interpret a fully-normalized (space-free, lowercase) string as a
    number. Returns None if it cannot be parsed as a plain number or fraction.
    """
    if not s:
        return None
    # Direct float (handles int, decimal, scientific notation like 1e-3)
    try:
        return float(s)
    except ValueError:
        pass
    # Simple a/b fraction
    m = _FRACTION_RE.fullmatch(s)
    if m:
        denom = int(m.group(2))
        if denom != 0:
            return int(m.group(1)) / denom
    return None


def _numbers_equal(a: float, b: float) -> bool:
    """Floating-point equality with tight tolerance."""
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) < 1e-9 * scale


# ---------------------------------------------------------------------------
# F1 token matching
# ---------------------------------------------------------------------------

_F1_THRESHOLD = 0.5


def _token_f1(ca: str, ma: str) -> float:
    """
    SQuAD-style token F1 on normalize_text-processed strings.
    Returns a float in [0, 1].
    """
    gold_tokens = normalize_text(ca).split()
    pred_tokens = normalize_text(ma).split()
    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)
    from collections import Counter
    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Per-dataset is_correct
# ---------------------------------------------------------------------------

# hotpotqa / musique: normalize_text exact match, then F1 fallback
_TEXT_DATASETS = {"hotpotqa", "musique"}
# math500 / gpqa_diamond: normalize_math only
_MATH_DATASETS = {"math500", "gpqa_diamond"}
# hle: normalize_math first (catches numeric/formula answers), then F1 fallback
_HLE_DATASETS = {"hle"}


def is_correct(correct_answer: str, model_answer: str | None, dataset: str) -> bool:
    if model_answer is None:
        return False

    ca = correct_answer
    ma = model_answer

    if dataset in _TEXT_DATASETS:
        if normalize_text(ca) == normalize_text(ma):
            return True
        return _token_f1(ca, ma) >= _F1_THRESHOLD

    if dataset in _HLE_DATASETS:
        # Try math normalization first (numbers, fractions, LaTeX formulas)
        norm_ca = normalize_math(ca)
        norm_ma = normalize_math(ma)
        if norm_ca == norm_ma:
            return True
        ca_num = _to_number(norm_ca)
        ma_num = _to_number(norm_ma)
        if ca_num is not None and ma_num is not None:
            return _numbers_equal(ca_num, ma_num)
        # Fall back to F1 for text-style answers
        return _token_f1(ca, ma) >= _F1_THRESHOLD

    # Math/science datasets (math500, gpqa_diamond)
    norm_ca = normalize_math(ca)
    norm_ma = normalize_math(ma)

    if norm_ca == norm_ma:
        return True

    # Numeric fallback
    ca_num = _to_number(norm_ca)
    ma_num = _to_number(norm_ma)
    if ca_num is not None and ma_num is not None:
        return _numbers_equal(ca_num, ma_num)

    return False


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def _grade_file(path: Path) -> dict:
    # split on "\n" only — splitlines() breaks on U+2028/U+2029/U+0085 that
    # json.dumps(ensure_ascii=False) leaves raw inside string values.
    lines = [l for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
    stats: dict = {"total": len(lines), "correct": 0, "wrong": 0}
    graded: list[str] = []

    for line in lines:
        record = json.loads(line)
        result = is_correct(
            record["correct_answer"],
            record.get("model_answer"),
            record["dataset"],
        )
        record["is_correct"] = result
        graded.append(json.dumps(record, ensure_ascii=False))
        if result:
            stats["correct"] += 1
        else:
            stats["wrong"] += 1

    text = "\n".join(graded) + "\n"
    for ch, esc in ((chr(0x2028), "\\u2028"), (chr(0x2029), "\\u2029"), (chr(0x85), "\\u0085")):
        text = text.replace(ch, esc)
    path.write_text(text, encoding="utf-8")
    stats["accuracy"] = round(stats["correct"] / stats["total"], 4) if stats["total"] else 0.0
    return stats


def main() -> None:
    clean_dir = Path(__file__).parent.parent / "gpt-oss-20b"
    if not clean_dir.is_dir():
        print(f"Directory not found: {clean_dir}", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(clean_dir.glob("clean_*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found.", file=sys.stderr)
        sys.exit(1)

    for path in jsonl_files:
        stats = _grade_file(path)
        print(
            f"{path.name}: "
            f"accuracy={stats['accuracy']:.1%}  "
            f"({stats['correct']}/{stats['total']})"
        )


if __name__ == "__main__":
    main()
