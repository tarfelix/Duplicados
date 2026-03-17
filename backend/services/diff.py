"""HTML diff rendering — ported from src/components/ui.py."""
import html
import re
from difflib import SequenceMatcher
from typing import Tuple


def render_diff(a: str, b: str, limit: int = 12000) -> Tuple[str, str]:
    if (len(a) + len(b)) > limit:
        a, b = a[: limit // 2], b[: limit // 2]
        note = "<div class='text-sm text-gray-500'>⚠️ Diff parcial (textos muito grandes)</div>"
    else:
        note = ""

    tokens1 = [t for t in re.split(r"(\W+)", a or "") if t]
    tokens2 = [t for t in re.split(r"(\W+)", b or "") if t]
    sm = SequenceMatcher(None, tokens1, tokens2, autojunk=False)

    out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        s1 = html.escape("".join(tokens1[i1:i2]))
        s2 = html.escape("".join(tokens2[j1:j2]))
        if tag == "equal":
            out1.append(s1)
            out2.append(s2)
        elif tag == "replace":
            out1.append(f"<span class='diff-del'>{s1}</span>")
            out2.append(f"<span class='diff-ins'>{s2}</span>")
        elif tag == "delete":
            out1.append(f"<span class='diff-del'>{s1}</span>")
        elif tag == "insert":
            out2.append(f"<span class='diff-ins'>{s2}</span>")

    return (note + "".join(out1), note + "".join(out2))
