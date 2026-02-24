import streamlit as st
import html
import re
from difflib import SequenceMatcher
from typing import Tuple

def apply_styles():
    st.markdown("""
    <style>
        pre.highlighted-text {
            white-space: pre-wrap; word-wrap: break-word; font-family: monospace;
            font-size: .9em; padding: 10px; border: 1px solid #ddd;
            border-radius: 5px; background-color: #f9f9f9; height: 360px; overflow-y: auto;
        }
        .diff-del { background-color: #ffcdd2 !important; }
        .diff-ins { background-color: #c8e6c9 !important; }
        .card { border-left: 5px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #fff; }
        .card-cancelado { background-color: #f5f5f5; border-left: 5px solid #e0e0e0; }
        .card-principal { border-left: 5px solid #4CAF50; }
        .similarity-badge { padding: 3px 6px; border-radius: 5px; color: black; font-weight: 600; display: inline-block; margin-bottom: 6px; }
        .badge-green { background:#C8E6C9; }
        .badge-yellow { background:#FFF9C4; }
        .badge-red { background:#FFCDD2; }
    </style>
    """, unsafe_allow_html=True)

def render_diff(a: str, b: str, limit: int = 12000) -> Tuple[str, str]:
    if (len(a) + len(b)) > limit:
        a, b = a[:limit//2], b[:limit//2]
        note = "<div class='small-muted'>⚠️ Diff parcial</div>"
    else: note = ""

    tokens1 = [t for t in re.split(r'(\W+)', a or "") if t]
    tokens2 = [t for t in re.split(r'(\W+)', b or "") if t]
    sm = SequenceMatcher(None, tokens1, tokens2, autojunk=False)
    
    out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        s1, s2 = html.escape("".join(tokens1[i1:i2])), html.escape("".join(tokens2[j1:j2]))
        if tag == 'equal':
            out1.append(s1); out2.append(s2)
        elif tag == 'replace':
            out1.append(f"<span class='diff-del'>{s1}</span>"); out2.append(f"<span class='diff-ins'>{s2}</span>")
        elif tag == 'delete':
            out1.append(f"<span class='diff-del'>{s1}</span>")
        elif tag == 'insert':
            out2.append(f"<span class='diff-ins'>{s2}</span>")
            
    return (note + f"<pre class='highlighted-text'>{''.join(out1)}</pre>", 
            note + f"<pre class='highlighted-text'>{''.join(out2)}</pre>")
