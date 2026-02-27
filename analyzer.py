from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_matrix(texts: List[str]) -> np.ndarray:
    """
    Builds cosine similarity matrix using TF-IDF.
    Robust choice for "universal" use without external model downloads.

    We use a hybrid-like trick: word ngrams + char ngrams, concatenated in a single vectorizer
    via analyzer='char_wb' would lose word ngrams; instead we keep word ngrams and rely on
    unigrams/bigrams + normalization. (You can extend to a FeatureUnion if needed.)
    """
    cleaned = [(t or "").strip() for t in texts]

    # TF-IDF on words (1-2 grams). No stopwords to keep it language-agnostic.
    vect = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X = vect.fit_transform(cleaned)
    sim = cosine_similarity(X)
    return sim


def compute_similarity_report(
    articles_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Returns a dataframe of pairs above similarity threshold.
    """
    if articles_df.empty:
        return pd.DataFrame()

    urls = articles_df["URL"].tolist()
    titles = articles_df["title"].tolist()
    h1s = articles_df["H1"].tolist()

    n = len(urls)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if s >= threshold:
                rows.append(
                    {
                        "url_a": urls[i],
                        "title_a": titles[i],
                        "h1_a": h1s[i],
                        "url_b": urls[j],
                        "title_b": titles[j],
                        "h1_b": h1s[j],
                        "similarity": round(s, 4),
                        "similarity_%": round(s * 100, 2),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=[
            "url_a", "title_a", "h1_a",
            "url_b", "title_b", "h1_b",
            "similarity", "similarity_%"
        ])

    out = pd.DataFrame(rows)
    out = out.sort_values(["similarity", "similarity_%"], ascending=False).reset_index(drop=True)
    return out
