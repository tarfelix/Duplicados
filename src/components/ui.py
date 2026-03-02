import streamlit as st
import html
import re
from difflib import SequenceMatcher
from typing import Tuple, List, Dict, Set, Any
import pandas as pd
from zoneinfo import ZoneInfo
from src.config import SK, TZ_SP, TZ_UTC

# We will pass logic functions (combined_score, log_action) as arguments to avoid circular imports

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
        .small-muted { color:#777; font-size:0.85em; }
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

@st.dialog("Ver diferenças", width="large")
def show_diff_dialog(principal: Dict, comparado: Dict, diff_limit: int, explain_fn: Any = None):
    st.markdown("""
        <div style='margin-bottom: 10px;'><strong>Legenda:</strong>
           <span style='background-color: #c8e6c9; padding: 2px 5px; border-radius: 3px;'>Texto adicionado</span>
           <span style='background-color: #ffcdd2; padding: 2px 5px; border-radius: 3px; margin-left: 10px;'>Texto removido</span>
        </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(f"**Texto mantido (principal)** — ID `{principal['activity_id']}`")
    c2.markdown(f"**Outro item** — ID `{comparado['activity_id']}`")
    
    hA, hB = render_diff(principal.get("Texto", ""), comparado.get("Texto", ""), diff_limit)
    c1.markdown(hA, unsafe_allow_html=True)
    c2.markdown(hB, unsafe_allow_html=True)
    if explain_fn:
        st.markdown("---")
        if st.button("Explicar diferenças em linguagem simples (IA)"):
            with st.spinner("Gerando explicação..."):
                try:
                    text = explain_fn(principal.get("Texto", ""), comparado.get("Texto", ""))
                    if text:
                        st.info(text)
                    else:
                        st.warning("Não foi possível gerar a explicação.")
                except Exception as e:
                    st.error(f"Erro ao chamar IA: {e}")

def render_group(group_rows: List[Dict], 
                 params: Dict, 
                 get_best_principal_fn: Any,
                 combined_score_fn: Any,
                 explain_fn: Any = None):
    """Renderiza um único grupo de atividades duplicadas com toda a lógica de interação."""
    group_id = group_rows[0]["activity_id"]
    
    # State management
    if SK.GROUP_STATES not in st.session_state:
        st.session_state[SK.GROUP_STATES] = {}
        
    state = st.session_state[SK.GROUP_STATES].setdefault(group_id, {
        "principal_id": None,
        "cancelados": set()
    })

    # Auto-calculate principal
    if state["principal_id"] is None:
        state["principal_id"] = get_best_principal_fn(group_rows, params['min_sim'] * 100, params['min_containment'])

    principal = next((r for r in group_rows if r["activity_id"] == state["principal_id"]), group_rows[0])
    p_norm = principal.get("_norm", "")
    p_meta = principal.get("_meta", {})

    open_count = sum(1 for r in group_rows if r.get("activity_status") == "Aberta")
    expander_title = f"Grupo: {len(group_rows)} itens ({open_count} abertas) | Pasta: {group_rows[0].get('activity_folder')} | Manter: #{state['principal_id']}"
    
    with st.expander(expander_title):
        st.caption("**Passo a passo:** 1. Escolha qual item manter → 2. Marque os repetidos para cancelar → 3. Use 'Processar Marcados' abaixo.")
        cols = st.columns(3)
        with cols[0]:
            if st.button("Recalcular qual manter", key=f"recalc_{group_id}", help="Recalcula qual item será mantido (principal)."):
                state["principal_id"] = get_best_principal_fn(group_rows, params['min_sim'] * 100, params['min_containment'])
                st.rerun()
        with cols[1]:
            if st.button("Marcar todos para cancelar", key=f"all_{group_id}", help="Marca todos os outros itens como repetidos para cancelamento."):
                state['cancelados'].update({r['activity_id'] for r in group_rows if r['activity_id'] != state['principal_id']})
                st.rerun()
        with cols[2]:
            if st.button("Este grupo não é duplicata", key=f"ign_{group_id}", help="Remove este grupo da lista (não processará cancelamentos aqui)."):
                st.session_state[SK.IGNORED_GROUPS].add(group_id)
                st.rerun()

        st.divider()

        for row in group_rows:
            rid = row["activity_id"]
            is_p = (rid == state["principal_id"])
            is_c = (rid in state["cancelados"])
            
            card_class = "card"
            if is_p: card_class += " card-principal"
            if is_c: card_class += " card-cancelado"
            
            st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
            c1, c2 = st.columns([0.7, 0.3])
            
            with c1:
                dt = pd.to_datetime(row.get("activity_date")).tz_localize(TZ_UTC).tz_convert(TZ_SP) if row.get("activity_date") else None
                date_str = dt.strftime('%d/%m/%Y %H:%M') if dt else "N/A"
                st.markdown(f"**ID:** `{rid}` { '⭐ **Manter este**' if is_p else ''} { '🗑️ **Marcado para cancelar**' if is_c else ''}")
                st.caption(f"{date_str} | {row.get('activity_status')} | {row.get('user_profile_name')}")
                
                if not is_p:
                    score, details = combined_score_fn(p_norm, row.get("_norm", ""), p_meta, row.get("_meta", {}))
                    min_sim_pct = params['min_sim'] * 100
                    badge_class = "badge-green" if score >= min_sim_pct + 5 else "badge-yellow" if score >= min_sim_pct else "badge-red"
                    tooltip = f"Detalhes: Set {details.get('set', 0):.0f}% | Sort {details.get('sort', 0):.0f}% | Contain {details.get('contain', 0):.0f}% | Bônus {details.get('bonus', 0)}"
                    if score >= 95:
                        label = "Muito parecido"
                    elif score >= min_sim_pct:
                        label = "Parecido"
                    else:
                        label = "Atenção: pode ter diferenças"
                    st.markdown(f"<span class='similarity-badge {badge_class}' title='{html.escape(tooltip)}'>{label} — {score:.0f}%</span>", unsafe_allow_html=True)

                st.text_area("Texto", row.get("Texto", ""), height=100, disabled=True, key=f"txt_{rid}_{group_id}")

            with c2:
                if not is_p:
                    cancel = st.checkbox("Descartar como repetido", value=is_c, key=f"chk_{rid}_{group_id}", help="Marcar este item para cancelamento.")
                    if cancel != is_c:
                        if cancel: state["cancelados"].add(rid)
                        else: state["cancelados"].discard(rid)
                        st.rerun()
                    
                    if st.button("Manter este", key=f"best_{rid}_{group_id}", use_container_width=True, help="Escolher este item como o que será mantido."):
                        state["principal_id"] = rid
                        st.rerun()
                    
                    if st.button("Ver diferenças", key=f"cmp_{rid}_{group_id}", use_container_width=True):
                        show_diff_dialog(principal, row, params.get('diff_limit', 12000), explain_fn)

            st.markdown("</div>", unsafe_allow_html=True)
