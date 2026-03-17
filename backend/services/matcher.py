"""
Duplicate detection engine — ported from src/core/matcher.py without Streamlit dependencies.
"""
import re
import pandas as pd
from unidecode import unidecode
from rapidfuzz import fuzz
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

CNJ_RE = re.compile(r"(?:\b|^)(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})(?:\b|$)")
URL_RE = re.compile(r"https?://\S+")
DATENUM_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
NUM_RE = re.compile(r"\b\d+\b")

STOPWORDS_BASE = set("""
    de da do das dos e em a o os as na no para por com que ao aos às à um uma umas uns
    tipo titulo inteiro teor publicado publicacao disponibilizacao orgao vara tribunal
    processo recurso intimacao notificacao justica nacional diario djen poder judiciario trabalho
""".split())


def extract_meta(text: str) -> Dict[str, str]:
    t = text or ""
    meta = {}
    cnj_match = CNJ_RE.search(t)
    cnj = cnj_match.group(1) if cnj_match else None
    if not cnj:
        proc_match = re.search(r"PROCESSO:\s*([0-9\-.]+)", t, re.IGNORECASE)
        if proc_match:
            cnj = proc_match.group(1)
    meta["processo"] = cnj or ""

    patterns = {
        "orgao": r"\bORGAO:\s*([^-\n\r]+)",
        "vara": r"\bVARA\s+DO\s+TRABALHO\s+[^-\n\r]+",
        "tipo_doc": r"\bTIPO\s+DE\s+DOCUMENTO:\s*([^-]+)",
        "tipo_com": r"\bTIPO\s+DE\s+COMUNICACAO:\s*([^-]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, t, re.IGNORECASE)
        if match:
            meta[key] = match.group(1).strip() if key != "vara" else match.group(0).strip()
    return meta


def normalize_text(text: str, stopwords_extra: List[str]) -> str:
    if not isinstance(text, str):
        return ""
    t = URL_RE.sub(" url ", text)
    t = CNJ_RE.sub(" numproc ", t)
    t = DATENUM_RE.sub(" data ", t)
    t = NUM_RE.sub(" # ", t)
    t = unidecode(t.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    all_stopwords = STOPWORDS_BASE.union(stopwords_extra)
    tokens = [w for w in t.split() if w not in all_stopwords]
    return " ".join(tokens)


def combined_score(
    a_norm: str, b_norm: str, meta_a: Dict[str, str], meta_b: Dict[str, str]
) -> Tuple[float, Dict[str, float]]:
    set_ratio = fuzz.token_set_ratio(a_norm, b_norm)
    sort_ratio = fuzz.token_sort_ratio(a_norm, b_norm)

    tokens_a, tokens_b = a_norm.split(), b_norm.split()
    if not tokens_a or not tokens_b:
        contain = 0.0
    else:
        small, big = (tokens_a, set(tokens_b)) if len(tokens_a) <= len(tokens_b) else (tokens_b, set(tokens_a))
        contain = 100.0 * (sum(1 for t in small if t in big) / len(small))

    len_a, len_b = len(a_norm), len(b_norm)
    lp = 1.0
    if len_a == 0 or len_b == 0:
        lp = 0.7
    elif len_a > 0 and len_b > 0:
        diff_ratio = abs(len_a - len_b) / max(len_a, len_b)
        lp = max(0.7, 1.0 - diff_ratio * 0.4)
        if (meta_a.get("processo") and meta_a.get("processo") == meta_b.get("processo")) or (
            meta_a.get("orgao") and meta_a.get("orgao") == meta_b.get("orgao")
        ):
            lp = max(lp, 0.85)

    base_score = 0.6 * set_ratio + 0.2 * sort_ratio + 0.2 * contain
    base_after_lp = base_score * lp
    bonus = 0
    if base_after_lp >= 80:
        if meta_a.get("processo") and meta_a.get("processo") == meta_b.get("processo"):
            bonus += 6
        if meta_a.get("orgao") and meta_a.get("orgao") == meta_b.get("orgao"):
            bonus += 3
        if meta_a.get("tipo_doc") and meta_a.get("tipo_doc") == meta_b.get("tipo_doc"):
            bonus += 3
        if meta_a.get("tipo_com") and meta_a.get("tipo_com") == meta_b.get("tipo_com"):
            bonus += 2
        bonus = min(bonus, 5)
    final_score = max(0.0, min(100.0, base_after_lp + bonus))

    return final_score, {"set": set_ratio, "sort": sort_ratio, "contain": contain, "len_pen": lp, "bonus": bonus}


def get_best_principal_id(group_rows: List[Dict], min_sim_pct: float, min_containment_pct: float) -> str:
    if not group_rows:
        return ""

    closed_ids = {r["activity_id"] for r in group_rows if r.get("activity_status") != "Aberta"}
    candidates = sorted(group_rows, key=lambda x: x["activity_id"] not in closed_ids)

    best_id, max_avg_score = None, -1.0

    for candidate in candidates:
        candidate_id = candidate["activity_id"]
        c_norm = candidate.get("_norm") or normalize_text(candidate.get("Texto", ""), [])
        c_meta = candidate.get("_meta") or extract_meta(candidate.get("Texto", ""))

        scores = []
        for other in group_rows:
            if other["activity_id"] == candidate_id:
                continue
            o_norm = other.get("_norm") or normalize_text(other.get("Texto", ""), [])
            o_meta = other.get("_meta") or extract_meta(other.get("Texto", ""))

            score, details = combined_score(c_norm, o_norm, c_meta, o_meta)
            if score >= min_sim_pct and details["contain"] >= min_containment_pct:
                scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        if best_id is None or avg_score > max_avg_score:
            max_avg_score, best_id = avg_score, candidate_id

        if candidate_id not in closed_ids and best_id in closed_ids:
            break

    return str(best_id or group_rows[0]["activity_id"])


def create_groups(df: pd.DataFrame, params: Dict) -> List[List[Dict]]:
    if df.empty:
        return []

    work_df = df.copy()
    stopwords_extra = params.get("stopwords_extra", [])
    if isinstance(stopwords_extra, str):
        stopwords_extra = [s.strip() for s in stopwords_extra.split(",") if s.strip()]
    work_df["_meta"] = work_df["Texto"].apply(extract_meta)
    work_df["_norm"] = work_df["Texto"].apply(lambda t: normalize_text(t, stopwords_extra))

    cutoffs_map = params.get("cutoffs_map", {})
    min_tokens = int(params.get("min_tokens", 3))

    def _normalize_sim_value(raw) -> float:
        v = float(raw)
        return v / 100.0 if v > 1 else v

    buckets = defaultdict(list)
    for i, row in work_df.iterrows():
        key = f"folder::{row.get('activity_folder', 'SEM')}"
        if params.get("use_cnj"):
            key += f"::cnj::{row['_meta']['processo'] or 'SEM'}"
        buckets[key].append(i)

    groups = []
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue

        n = len(idxs)
        local_visited = [False] * n
        for i in range(n):
            if local_visited[i]:
                continue

            cluster = [idxs[i]]
            queue = deque([i])
            local_visited[i] = True

            while queue:
                curr = queue.popleft()
                for j in range(n):
                    if not local_visited[j]:
                        row_i, row_j = work_df.loc[idxs[curr]], work_df.loc[idxs[j]]

                        norm_i = row_i["_norm"]
                        norm_j = row_j["_norm"]
                        tokens_i = norm_i.split() if isinstance(norm_i, str) else []
                        tokens_j = norm_j.split() if isinstance(norm_j, str) else []
                        if len(tokens_i) < min_tokens or len(tokens_j) < min_tokens:
                            continue

                        folder = row_i.get("activity_folder")
                        raw_min_sim = cutoffs_map.get(folder, params["min_sim"])
                        min_sim = _normalize_sim_value(raw_min_sim) * 100

                        score, details = combined_score(row_i["_norm"], row_j["_norm"], row_i["_meta"], row_j["_meta"])
                        if score >= min_sim and details["contain"] >= params["min_containment"]:
                            local_visited[j] = True
                            cluster.append(idxs[j])
                            queue.append(j)

            if len(cluster) > 1:
                cluster_rows = work_df.loc[cluster].sort_values("activity_date", ascending=False)
                groups.append(cluster_rows.to_dict("records"))

    return groups
