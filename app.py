import streamlit as st
import pandas as pd
import numpy as np

from scraper import scrape_site_articles
from analyzer import (
    compute_similarity_report,
    build_similarity_matrix,
)

st.set_page_config(
    page_title="SEO Cannibalization Auditor",
    layout="wide",
)

st.title("SEO Cannibalization Auditor (Scrape → Markdown CSV → Similarity)")

with st.expander("Uwaga / dobre praktyki", expanded=False):
    st.markdown(
        """
- Narzędzie pobiera treści ze stron WWW. Upewnij się, że masz prawo do scrapowania analizowanej witryny.
- W praktyce warto respektować robots.txt i nie przeciążać serwera (rate limiting jest wbudowany).
- Wyniki podobieństwa to sygnał do audytu — nie zawsze oznaczają realną kanibalizację.
        """.strip()
    )

colA, colB, colC = st.columns([2, 1, 1])

with colA:
    base_url = st.text_input(
        "Adres strony (np. https://hipoteczny.pl)",
        value="https://hipoteczny.pl",
        placeholder="https://example.com",
    )

with colB:
    max_pages = st.number_input(
        "Max liczba artykułów",
        min_value=5,
        max_value=5000,
        value=200,
        step=5,
        help="Limit bezpieczeństwa. Przy dużych serwisach zacznij od 100–300.",
    )

with colC:
    similarity_threshold = st.slider(
        "Próg podobieństwa (%)",
        min_value=50,
        max_value=95,
        value=70,
        step=1,
        help="Pary powyżej progu traktuj jako potencjalną kanibalizację.",
    )

advanced = st.expander("Ustawienia zaawansowane", expanded=False)
with advanced:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        request_timeout = st.number_input("Timeout requestów (s)", 5, 60, 20, 1)
    with c2:
        delay_seconds = st.number_input("Opóźnienie między requestami (s)", 0.0, 5.0, 0.6, 0.1)
    with c3:
        crawl_depth = st.number_input("Maks. głębokość crawl", 1, 8, 3, 1)
    with c4:
        same_subdomain_only = st.checkbox(
            "Tylko ten sam subdomen",
            value=True,
            help="Jeśli odznaczysz, zbierze linki z całej domeny głównej.",
        )

    ua = st.text_input(
        "User-Agent",
        value="Mozilla/5.0 (compatible; SEOContentAuditor/1.0; +https://github.com/your-repo)",
    )

run = st.button("🚀 Uruchom scrapowanie i analizę", type="primary")

if "articles_df" not in st.session_state:
    st.session_state["articles_df"] = None
if "report_df" not in st.session_state:
    st.session_state["report_df"] = None
if "sim_matrix" not in st.session_state:
    st.session_state["sim_matrix"] = None

if run:
    if not base_url or not base_url.startswith(("http://", "https://")):
        st.error("Podaj poprawny URL zaczynający się od http:// lub https://")
        st.stop()

    status = st.status("Startuję…", expanded=True)
    progress = st.progress(0)

    def on_progress(done: int, total: int, message: str):
        if total > 0:
            progress.progress(min(1.0, done / total))
        status.write(message)

    try:
        status.update(label="🔎 Wykrywam i pobieram artykuły…", state="running")
        articles_df = scrape_site_articles(
            base_url=base_url,
            max_pages=int(max_pages),
            timeout=int(request_timeout),
            delay=float(delay_seconds),
            max_depth=int(crawl_depth),
            user_agent=ua,
            same_subdomain_only=bool(same_subdomain_only),
            progress_callback=on_progress,
        )

        if articles_df.empty:
            status.update(label="Nie znaleziono artykułów lub nie udało się pobrać treści.", state="error")
            st.stop()

        st.session_state["articles_df"] = articles_df

        status.update(label="🧠 Liczę podobieństwa…", state="running")
        # Macierz (do wizualizacji) + raport par powyżej progu
        sim_matrix = build_similarity_matrix(articles_df["treść w Markdown"].fillna("").tolist())
        report_df = compute_similarity_report(
            articles_df=articles_df,
            sim_matrix=sim_matrix,
            threshold=float(similarity_threshold) / 100.0,
        )

        st.session_state["report_df"] = report_df
        st.session_state["sim_matrix"] = sim_matrix

        status.update(label="✅ Gotowe", state="complete")
        progress.progress(1.0)

    except Exception as e:
        status.update(label="❌ Błąd", state="error")
        st.exception(e)

articles_df = st.session_state.get("articles_df")
report_df = st.session_state.get("report_df")
sim_matrix = st.session_state.get("sim_matrix")

if articles_df is not None:
    st.subheader("1) Dane artykułów (CSV: URL, H1, title, treść w Markdown)")
    st.dataframe(articles_df, use_container_width=True, height=350)

    csv_bytes = articles_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Pobierz CSV z artykułami",
        data=csv_bytes,
        file_name="articles_markdown.csv",
        mime="text/csv",
    )

if report_df is not None:
    st.subheader("2) Potencjalna kanibalizacja (pary powyżej progu)")
    if report_df.empty:
        st.info("Brak par powyżej ustawionego progu.")
    else:
        st.dataframe(report_df, use_container_width=True, height=350)
        rep_bytes = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Pobierz raport kanibalizacji (CSV)",
            data=rep_bytes,
            file_name="cannibalization_report.csv",
            mime="text/csv",
        )

if sim_matrix is not None and articles_df is not None:
    st.subheader("3) Wizualizacja podobieństwa (heatmap – top 60 URLi)")
    st.caption("Dla czytelności pokazuję maks. 60 pierwszych artykułów (możesz przefiltrować listę w kodzie).")

    import matplotlib.pyplot as plt

    n = min(60, sim_matrix.shape[0])
    m = sim_matrix[:n, :n]
    labels = [f"{i+1}" for i in range(n)]

    fig = plt.figure()
    plt.imshow(m, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(n), labels, rotation=90, fontsize=7)
    plt.yticks(range(n), labels, fontsize=7)
    plt.title("Cosine similarity (TF-IDF) – skrócona macierz")
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("Mapa indeksów → URL", expanded=False):
        tmp = articles_df[["URL", "title"]].head(n).copy()
        tmp.insert(0, "idx", np.arange(1, n + 1))
        st.dataframe(tmp, use_container_width=True, height=300)
