from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _clean_texts(texts: List[str]) -> List[str]:
    cleaned = []
    for t in texts:
        t = (t or "").strip()
        # minimalny sensowny próg, żeby nie truć wektoryzera pustkami
        if len(t.split()) >= 5:
            cleaned.append(t)
        else:
            cleaned.append("")  # zachowujemy indeksy, ale pusty tekst
    return cleaned


def build_similarity_matrix(texts: List[str]) -> np.ndarray:
    """
    Builds cosine similarity matrix using TF-IDF.
    Robust against small number of documents / empty documents.

    Returns NxN matrix aligned with input order.
    """
    cleaned = _clean_texts(texts)
    n = len(cleaned)

    # Jeśli za mało dokumentów, zwracamy trivial matrix
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.array([[1.0]], dtype=float)

    # Liczba dokumentów z niepustą treścią
    non_empty_idx = [i for i, t in enumerate(cleaned) if t]
    k = len(non_empty_idx)

    # Jeśli prawie wszystko puste, macierz „bezpieczna”
    if k <= 1:
        m = np.eye(n, dtype=float)
        return m

    # Dynamiczne df:
    # - min_df: 1 (zostawiamy), ale tylko dla non-empty
    # - max_df: jeśli mamy bardzo mało dokumentów, nie ograniczajmy max_df,
    #           bo może się okazać sprzeczne z min_df w praktyce.
    # Dla większych zbiorów 0.95 jest ok.
    max_df = 1.0 if k < 5 else 0.95

    vect = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=max_df,
    )

    # Fit/transform tylko na niepustych (żeby nie psuć df)
    non_empty_texts = [cleaned[i] for i in non_empty_idx]
    X = vect.fit_transform(non_empty_texts)
    sim_small = cosine_similarity(X)  # k x k

    # Wbudowujemy z powrotem do NxN
    sim = np.eye(n, dtype=float)
    for a_pos, a_idx in enumerate(non_empty_idx):
        for b_pos, b_idx in enumerate(non_empty_idx):
            sim[a_idx, b_idx] = float(sim_small[a_pos, b_pos])

    return sim


def compute_similarity_report(
    articles_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Returns dataframe of pairs above similarity threshold.
    """
    if articles_df is None or articles_df.empty:
        return pd.DataFrame(columns=[
            "url_a", "title_a", "h1_a",
            "url_b", "title_b", "h1_b",
            "similarity", "similarity_%"
        ])

    urls = articles_df["URL"].astype(str).tolist()
    titles = articles_df["title"].astype(str).tolist()
    h1s = articles_df["H1"].astype(str).tolist()

    n = len(urls)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if s >= threshold:
                rows.append({
                    "url_a": urls[i],
                    "title_a": titles[i],
                    "h1_a": h1s[i],
                    "url_b": urls[j],
                    "title_b": titles[j],
                    "h1_b": h1s[j],
                    "similarity": round(s, 4),
                    "similarity_%": round(s * 100, 2),
                })

    out = pd.DataFrame(rows, columns=[
        "url_a", "title_a", "h1_a",
        "url_b", "title_b", "h1_b",
        "similarity", "similarity_%"
    ])
    if out.empty:
        return out

    return out.sort_values("similarity", ascending=False).reset_index(drop=True)
