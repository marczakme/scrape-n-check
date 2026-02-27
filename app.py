import io
import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from scraper import (
    scrape_site_articles,
    scrape_articles_from_urls,
    normalize_url_public,
    filter_internal_urls_public,
)

from analyzer import (
    SimilarityConfig,
    build_similarity_matrices,
    similarity_pairs_report,
    similarity_groups_report,
    interpretation_help_text,
)

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="SEO Content Similarity & Cannibalization Audit",
    layout="wide",
)

st.title("SEO: analiza podobieństwa treści i kanibalizacji (CSV: URL, H1, title, treść w Markdown)")

st.markdown(
    """
To narzędzie:
- pobiera treści artykułów ze strony (albo z Twojej listy URL),
- zapisuje do CSV: **URL, H1, title, treść w Markdown**,
- liczy podobieństwo treści różnymi metodami,
- wskazuje pary i grupy potencjalnej kanibalizacji.
"""
)

# -------------------------
# Helpers
# -------------------------
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _read_urls_from_uploaded_csv(file) -> list[str]:
    """
    Akceptuje CSV z kolumną 'URL' albo pierwszą kolumną jako URL.
    """
    df = pd.read_csv(file)
    if df.empty:
        return []
    if "URL" in df.columns:
        urls = df["URL"].astype(str).tolist()
    else:
        urls = df.iloc[:, 0].astype(str).tolist()
    urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    return urls

def _extract_urls_from_text(text: str) -> list[str]:
    """
    Zbiera URL-e z textarea (każda linia = URL; toleruje śmieci).
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines()]
    urls = []
    for ln in lines:
        if not ln:
            continue
        # prosta walidacja
        if ln.startswith("http://") or ln.startswith("https://"):
            urls.append(ln)
    return urls

def _progress_callback_factory(status_el, bar_el):
    def cb(done, total, message):
        # total może być 0 w callbackach diagnostycznych – obsłuż
        if total and total > 0:
            bar_el.progress(min(1.0, max(0.0, done / total)))
            status_el.write(message)
        else:
            status_el.write(message)
    return cb

def _download_csv_button(df: pd.DataFrame, label: str, filename: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )

def _plot_similarity_hist(sim: np.ndarray, title: str):
    # bierzemy tylko górny trójkąt bez przekątnej
    if sim.size == 0:
        st.info("Brak danych do wykresu.")
        return
    n = sim.shape[0]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(float(sim[i, j]))
    if not vals:
        st.info("Brak par do wykresu.")
        return

    fig = plt.figure()
    plt.hist(vals, bins=30)
    plt.title(title)
    plt.xlabel("Podobieństwo (0–1)")
    plt.ylabel("Liczba par")
    st.pyplot(fig)


# -------------------------
# Sidebar: input options
# -------------------------
st.sidebar.header("1) Dane wejściowe")

base_url = st.sidebar.text_input(
    "Adres strony (base URL)",
    value="https://marczak.me",
    help="Podaj domenę startową. Przykład: https://marczak.me",
)

mode = st.sidebar.radio(
    "Skąd wziąć URL-e do analizy?",
    options=[
        "Auto (sitemap/RSS) – pobierz artykuły ze strony",
        "Wklej URL-e ręcznie",
        "Wgraj CSV z URL-ami",
    ],
    index=0,
)

max_pages = st.sidebar.number_input(
    "Limit URL-i / artykułów",
    min_value=1,
    max_value=2000,
    value=200,
    step=10,
)

delay = st.sidebar.slider(
    "Opóźnienie między requestami (sekundy)",
    min_value=0.0,
    max_value=3.0,
    value=0.6,
    step=0.1,
)

timeout = st.sidebar.number_input(
    "Timeout requestu (sekundy)",
    min_value=5,
    max_value=120,
    value=20,
    step=5,
)

same_subdomain_only = st.sidebar.checkbox(
    "Tylko ta sama subdomena (stricte ten sam host)",
    value=True,
    help="Jeśli odznaczysz, scraper może brać też subdomeny w ramach tej samej domeny głównej.",
)

st.sidebar.header("2) Ustawienia podobieństwa (w %)")
st.sidebar.markdown("To są progi w **procentach (0–100)** – wygodne dla nietechnicznych.")

method = st.sidebar.selectbox(
    "Metoda porównania",
    ["hybrid", "word_tfidf", "char_tfidf"],
    index=0,
    help="Hybrid jest zwykle najlepszy do kanibalizacji SEO.",
)

threshold_pct = st.sidebar.slider(
    "Próg podobieństwa (%)",
    min_value=0,
    max_value=100,
    value=30,
    step=1,
    help="Np. 30% pokaże pary podobne co najmniej w 30%.",
)

boiler_df_pct = st.sidebar.slider(
    "Usuwanie powtarzalnych fragmentów (boilerplate) – próg (%)",
    min_value=5,
    max_value=60,
    value=25,
    step=5,
    help="25% oznacza: usuń linie, które pojawiają się w >=25% dokumentów.",
)

min_words = st.sidebar.number_input(
    "Minimalna liczba słów w dokumencie (po czyszczeniu)",
    min_value=10,
    max_value=500,
    value=40,
    step=10,
)

max_pairs = st.sidebar.number_input(
    "Limit par w raporcie",
    min_value=100,
    max_value=20000,
    value=2000,
    step=100,
)

st.sidebar.header("3) Uruchomienie")
run_btn = st.sidebar.button("🚀 Start: pobierz i policz podobieństwo", type="primary")


# -------------------------
# Main: explanation for non-technical users
# -------------------------
with st.expander("Instrukcja: jak rozumieć próg podobieństwa (dla nietechnicznych)", expanded=True):
    st.markdown(interpretation_help_text())

# -------------------------
# Optional: manual URLs input
# -------------------------
manual_urls_text = ""
uploaded_csv = None

if mode == "Wklej URL-e ręcznie":
    manual_urls_text = st.text_area(
        "Wklej URL-e (1 linia = 1 URL)",
        height=180,
        placeholder="https://marczak.me/jakis-artykul/\nhttps://marczak.me/inny-artykul/",
    )

if mode == "Wgraj CSV z URL-ami":
    uploaded_csv = st.file_uploader("Wgraj CSV z URL-ami (kolumna 'URL' lub pierwsza kolumna)", type=["csv"])


# -------------------------
# Run pipeline
# -------------------------
if run_btn:
    base_url = (base_url or "").strip()
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        st.error("Base URL musi zaczynać się od http:// lub https://")
        st.stop()

    status = st.empty()
    bar = st.progress(0.0)
    cb = _progress_callback_factory(status, bar)

    # 1) Get articles_df
    try:
        if mode == "Auto (sitemap/RSS) – pobierz artykuły ze strony":
            status.write("Start: auto-wykrywanie URL-i (sitemap/RSS) i pobieranie artykułów…")
            articles_df = scrape_site_articles(
                base_url=base_url,
                max_pages=int(max_pages),
                timeout=int(timeout),
                delay=float(delay),
                max_depth=3,
                same_subdomain_only=bool(same_subdomain_only),
                progress_callback=cb,
            )

        elif mode == "Wklej URL-e ręcznie":
            urls = _extract_urls_from_text(manual_urls_text)
            urls = [normalize_url_public(u) for u in urls]
            urls = filter_internal_urls_public(urls, base_url=base_url, same_subdomain_only=bool(same_subdomain_only))
            if not urls:
                st.error("Nie wykryłam żadnych poprawnych URL-i do analizy.")
                st.stop()

            status.write(f"Start: pobieranie treści dla {len(urls)} URL-i…")
            articles_df = scrape_articles_from_urls(
                base_url=base_url,
                urls=urls,
                max_pages=int(max_pages),
                timeout=int(timeout),
                delay=float(delay),
                same_subdomain_only=bool(same_subdomain_only),
                progress_callback=cb,
            )

        else:  # CSV
            if uploaded_csv is None:
                st.error("Wgraj plik CSV z URL-ami.")
                st.stop()

            urls = _read_urls_from_uploaded_csv(uploaded_csv)
            urls = [normalize_url_public(u) for u in urls]
            urls = filter_internal_urls_public(urls, base_url=base_url, same_subdomain_only=bool(same_subdomain_only))
            if not urls:
                st.error("W CSV nie znalazłam URL-i pasujących do domeny / filtra.")
                st.stop()

            status.write(f"Start: pobieranie treści dla {len(urls)} URL-i z CSV…")
            articles_df = scrape_articles_from_urls(
                base_url=base_url,
                urls=urls,
                max_pages=int(max_pages),
                timeout=int(timeout),
                delay=float(delay),
                same_subdomain_only=bool(same_subdomain_only),
                progress_callback=cb,
            )

    except Exception as e:
        st.exception(e)
        st.stop()

    bar.progress(1.0)

    # basic validation
    if articles_df is None or articles_df.empty:
        st.warning("Nie udało się pobrać żadnych artykułów. Sprawdź URL, blokady anty-bot, limit, itp.")
        st.stop()

    # 2) Show and download articles CSV
    st.subheader("1) Dane artykułów (CSV: URL, H1, title, treść w Markdown)")
    st.dataframe(articles_df, use_container_width=True, height=300)
    _download_csv_button(articles_df, "⬇️ Pobierz CSV z artykułami", "articles.csv")

    # 3) Similarity analysis
    st.subheader("2) Analiza podobieństwa treści (kanibalizacja)")

    cfg = SimilarityConfig(
        similarity_threshold_pct=float(threshold_pct),
        method=method,
        boilerplate_line_df_pct=float(boiler_df_pct),
        min_words_per_doc=int(min_words),
        max_pairs=int(max_pairs),
    )

    texts = articles_df["treść w Markdown"].fillna("").astype(str).tolist()

    with st.spinner("Liczę podobieństwa (word/char/hybrid)…"):
        mats = build_similarity_matrices(texts, cfg)

    sim = mats["hybrid"] if method == "hybrid" else mats["word"] if method == "word_tfidf" else mats["char"]

    colA, colB = st.columns([1, 1])

    with colA:
        st.markdown("**Wykres rozkładu podobieństw (wszystkie pary)**")
        _plot_similarity_hist(sim, title=f"Similarity distribution ({method})")

    with colB:
        # szybkie statystyki
        n = sim.shape[0]
        if n >= 2:
            vals = []
            for i in range(n):
                for j in range(i + 1, n):
                    vals.append(float(sim[i, j]))
            if vals:
                st.metric("Maksymalne podobieństwo", f"{max(vals)*100:.1f}%")
                st.metric("Średnie podobieństwo", f"{(sum(vals)/len(vals))*100:.1f}%")
            else:
                st.info("Brak par do statystyk.")
        else:
            st.info("Za mało dokumentów do statystyk.")

    # Reports
    pairs_df = similarity_pairs_report(
        articles_df,
        sim,
        threshold_pct=cfg.similarity_threshold_pct,
        max_pairs=cfg.max_pairs,
    )
    groups_df = similarity_groups_report(
        articles_df,
        sim,
        threshold_pct=cfg.similarity_threshold_pct,
    )

    st.markdown(f"### 2.1 Pary artykułów powyżej progu: **{threshold_pct}%** ({method})")
    if pairs_df is None or pairs_df.empty:
        st.info(
            "Brak par powyżej progu. "
            "Spróbuj obniżyć próg (np. 20–30%) lub przełącz metodę na 'hybrid' (zalecane)."
        )
    else:
        st.dataframe(pairs_df, use_container_width=True, height=350)
        _download_csv_button(pairs_df, "⬇️ Pobierz CSV: pary podobnych artykułów", "similarity_pairs.csv")

    st.markdown(f"### 2.2 Grupy (klastry) potencjalnej kanibalizacji powyżej progu: **{threshold_pct}%** ({method})")
    if groups_df is None or groups_df.empty:
        st.info("Brak grup (minimum 2 URL-e połączone podobieństwem >= próg).")
    else:
        st.dataframe(groups_df, use_container_width=True, height=300)
        _download_csv_button(groups_df, "⬇️ Pobierz CSV: grupy kanibalizacji", "similarity_groups.csv")

    st.success("Gotowe ✅")
