from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# Interpretacja i konfiguracja (user-friendly)
# -------------------------
@dataclass
class SimilarityConfig:
    """
    similarity_threshold_pct:
        Próg podobieństwa w % (0–100). Np. 30 oznacza 30%.

    method:
        - "word_tfidf": podobieństwo tematyczne (czy teksty są o tym samym)
        - "char_tfidf": podobieństwo tekstowe/szablonowe (czy mają dużo identycznych fragmentów)
        - "hybrid": łączy oba (najlepsze do kanibalizacji SEO)

    boilerplate_line_df_pct:
        Linie/fragmenty występujące w >= X% dokumentów uznajemy za boilerplate i usuwamy.
        To mocno poprawia jakość porównań (CTA, stopki, menu, standardowe bloki).
    """
    similarity_threshold_pct: float = 30.0
    method: str = "hybrid"
    boilerplate_line_df_pct: float = 25.0  # usuń linie wspólne dla >=25% stron
    min_words_per_doc: int = 40            # odrzuć "prawie puste" treści
    max_pairs: int = 2000                  # limit, żeby nie wysadzić UI


# -------------------------
# Tekst: czyszczenie i boilerplate removal
# -------------------------
def _normalize_text(t: str) -> str:
    t = (t or "").replace("\r", "\n")
    t = "\n".join([line.strip() for line in t.split("\n")])
    # usuń nadmiar pustych linii
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    return t.strip()


def _split_markdown_lines(t: str) -> List[str]:
    # Rozbijamy po liniach – w markdownzie to sensowna jednostka boilerplate
    lines = [ln.strip() for ln in (t or "").split("\n")]
    # odfiltruj bardzo krótkie “śmieciowe” linie
    lines = [ln for ln in lines if len(ln) >= 20]
    return lines


def remove_boilerplate_lines(texts: List[str], df_threshold_pct: float) -> List[str]:
    """
    Usuwa linie (po normalizacji), które powtarzają się w >= df_threshold_pct dokumentów.
    """
    if not texts:
        return texts

    df_threshold_pct = float(df_threshold_pct)
    n = len(texts)
    if n < 3:
        # dla małej liczby dokumentów boilerplate bywa trudny do oszacowania
        return [_normalize_text(t) for t in texts]

    # policz w ilu dokumentach występuje dana linia
    line_df: Dict[str, int] = {}
    docs_lines: List[List[str]] = []
    for t in texts:
        t = _normalize_text(t)
        lines = _split_markdown_lines(t)
        docs_lines.append(lines)
        uniq = set(lines)
        for ln in uniq:
            line_df[ln] = line_df.get(ln, 0) + 1

    df_cut = max(2, int(np.ceil((df_threshold_pct / 100.0) * n)))
    boiler = {ln for ln, c in line_df.items() if c >= df_cut}

    cleaned_texts: List[str] = []
    for lines in docs_lines:
        kept = [ln for ln in lines if ln not in boiler]
        cleaned_texts.append("\n".join(kept).strip())

    return cleaned_texts


def _filter_too_short(texts: List[str], min_words: int) -> Tuple[List[str], List[int]]:
    """
    Zwraca:
    - texts_filtered: lista tekstów (też zachowuje indeksy, ale puste dla odrzuconych)
    - valid_idx: indeksy dokumentów, które mają >= min_words słów
    """
    out = []
    valid_idx = []
    for i, t in enumerate(texts):
        t = (t or "").strip()
        if len(t.split()) >= int(min_words):
            out.append(t)
            valid_idx.append(i)
        else:
            out.append("")  # zachowujemy pozycję
    return out, valid_idx


# -------------------------
# Similarity: word TF-IDF / char TF-IDF / hybrid
# -------------------------
def _cosine_sim_matrix_from_vectors(X) -> np.ndarray:
    return cosine_similarity(X)


def _word_tfidf_sim(texts: List[str]) -> np.ndarray:
    vect = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X = vect.fit_transform(texts)
    return _cosine_sim_matrix_from_vectors(X)


def _char_tfidf_sim(texts: List[str]) -> np.ndarray:
    # char_wb redukuje szum (bardziej “w obrębie słów”)
    vect = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 6),
        min_df=1,
        max_df=1.0,
    )
    X = vect.fit_transform(texts)
    return _cosine_sim_matrix_from_vectors(X)


def build_similarity_matrices(
    raw_texts: List[str],
    cfg: SimilarityConfig
) -> Dict[str, np.ndarray]:
    """
    Zwraca słownik macierzy NxN:
    - "word"
    - "char"
    - "hybrid"
    Każda macierz jest dopasowana do kolejności wejściowej.
    """
    # 1) normalize
    texts = [_normalize_text(t) for t in raw_texts]

    # 2) boilerplate removal
    texts = remove_boilerplate_lines(texts, df_threshold_pct=cfg.boilerplate_line_df_pct)

    # 3) filter too short (keep indices)
    texts, valid_idx = _filter_too_short(texts, min_words=cfg.min_words_per_doc)
    n = len(texts)

    if n == 0:
        return {"word": np.zeros((0, 0)), "char": np.zeros((0, 0)), "hybrid": np.zeros((0, 0))}
    if n == 1:
        one = np.array([[1.0]], dtype=float)
        return {"word": one.copy(), "char": one.copy(), "hybrid": one.copy()}

    # jeśli prawie wszystko puste po filtrze
    if len(valid_idx) <= 1:
        eye = np.eye(n, dtype=float)
        return {"word": eye.copy(), "char": eye.copy(), "hybrid": eye.copy()}

    # liczymy tylko na valid_idx, potem wklejamy do NxN
    valid_texts = [texts[i] for i in valid_idx]

    # word sim
    try:
        word_small = _word_tfidf_sim(valid_texts)
    except ValueError:
        # fallback: bez max_df (przy mikrozbiorach)
        vect = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
        )
        X = vect.fit_transform(valid_texts)
        word_small = cosine_similarity(X)

    # char sim
    char_small = _char_tfidf_sim(valid_texts)

    # wbuduj do NxN
    word = np.eye(n, dtype=float)
    char = np.eye(n, dtype=float)

    for a_pos, a_idx in enumerate(valid_idx):
        for b_pos, b_idx in enumerate(valid_idx):
            word[a_idx, b_idx] = float(word_small[a_pos, b_pos])
            char[a_idx, b_idx] = float(char_small[a_pos, b_pos])

    # hybrid: ważone – priorytet “tematyczny”, ale podbite przez podobieństwo tekstowe
    # (to działa dobrze do kanibalizacji SEO)
    hybrid = (0.65 * word) + (0.35 * char)

    return {"word": word, "char": char, "hybrid": hybrid}


# -------------------------
# Raport: pary + grupy
# -------------------------
def similarity_pairs_report(
    df: pd.DataFrame,
    sim: np.ndarray,
    threshold_pct: float,
    max_pairs: int = 2000,
) -> pd.DataFrame:
    """
    Zwraca pary URL-i o podobieństwie >= threshold_pct.
    threshold_pct jest w % (0–100).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "url_a", "title_a", "h1_a",
            "url_b", "title_b", "h1_b",
            "similarity_%"
        ])

    thr = float(threshold_pct) / 100.0

    urls = df["URL"].astype(str).tolist()
    titles = df["title"].astype(str).tolist()
    h1s = df["H1"].astype(str).tolist()

    n = len(urls)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s >= thr:
                rows.append({
                    "url_a": urls[i],
                    "title_a": titles[i],
                    "h1_a": h1s[i],
                    "url_b": urls[j],
                    "title_b": titles[j],
                    "h1_b": h1s[j],
                    "similarity_%": round(s * 100, 2),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("similarity_%", ascending=False).reset_index(drop=True)
    if len(out) > int(max_pairs):
        out = out.head(int(max_pairs)).copy()
    return out


def similarity_groups_report(
    df: pd.DataFrame,
    sim: np.ndarray,
    threshold_pct: float,
) -> pd.DataFrame:
    """
    Buduje grupy (connected components) dla krawędzi sim>=thr.
    Zwraca wiersze: group_id, size, urls (newline separated).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["group_id", "size", "urls"])

    thr = float(threshold_pct) / 100.0
    urls = df["URL"].astype(str).tolist()
    n = len(urls)

    # adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i, j]) >= thr:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    groups: List[List[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        # BFS/DFS component
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) >= 2:  # grupy sensowne dopiero od 2 elementów
            groups.append(sorted(comp))

    # sort by size desc
    groups.sort(key=lambda g: len(g), reverse=True)

    rows = []
    for gid, comp in enumerate(groups, start=1):
        rows.append({
            "group_id": gid,
            "size": len(comp),
            "urls": "\n".join(urls[i] for i in comp),
        })

    return pd.DataFrame(rows, columns=["group_id", "size", "urls"])


# -------------------------
# Tekst instrukcji (dla nietechnicznych)
# -------------------------
def interpretation_help_text() -> str:
    return (
        "### Jak rozumieć „podobieństwo treści” (procenty)\n\n"
        "**To NIE jest ocena jakości tekstu.** To tylko miara tego, jak bardzo dwa artykuły są do siebie podobne.\n\n"
        "#### Co oznacza próg (np. 30%)?\n"
        "- Ustawiając **30%**, prosisz narzędzie: *„Pokaż mi pary artykułów, które są podobne co najmniej w 30%”*.\n"
        "- Im **wyższy próg**, tym mniej wyników (ale bardziej „oczywistych”).\n"
        "- Im **niższy próg**, tym więcej wyników (częściej dotyczy overlapu tematycznego).\n\n"
        "#### Jakie progi ustawiać w praktyce (kanibalizacja SEO)?\n"
        "- **20–30%**: często wykrywa *podobieństwo tematu* (dobre do szukania kanibalizacji).\n"
        "- **30–45%**: zwykle *mocne nakładanie tematów* albo podobna struktura.\n"
        "- **45%+**: często bardzo podobne teksty / duże fragmenty wspólne.\n\n"
        "#### Dlaczego czasem nic nie wychodzi przy 50%?\n"
        "Bo dwa artykuły mogą dotyczyć podobnego zagadnienia, ale być napisane innymi słowami. Wtedy podobieństwo tematyczne "
        "często mieści się np. w zakresie **20–40%**, a nie 50%.\n\n"
        "#### Co robi tryb „hybrid” (zalecany)?\n"
        "- Łączy porównanie *tematu* (słowa) i *fragmentów tekstu/szablonów* (znaki).\n"
        "- Dzięki temu lepiej wychwytuje realną kanibalizację i jednocześnie redukuje fałszywe alarmy od powtarzalnych stopki/CTA.\n"
    )
