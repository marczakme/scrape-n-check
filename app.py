import io
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

st.set_page_config(page_title="SEO Content Similarity & Cannibalization Audit", layout="wide")
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
# Helpers: encoding + mojibake fix
# -------------------------
def _try_read_csv_with_encodings(uploaded_file) -> pd.DataFrame:
    """
    Próbuje odczytać CSV kilkoma typowymi encodingami.
    Najczęściej działa: utf-8-sig.
    """
    raw = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "cp1250", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def _fix_mojibake(s: str) -> str:
    """
    Naprawia najczęstszy przypadek: UTF-8 zdekodowane jako latin1/cp1252,
    co daje: 'Å›', 'Ã³' itd.
    Jeżeli tekst jest OK, zwraca bez zmian.
    """
    if not isinstance(s, str) or not s:
        return s
    # szybka heurystyka: jeśli ma typowe sekwencje krzaków, próbujemy naprawy
    bad_markers = ["Ã", "Å", "Ä", "â", "Ê", "Ë", "Ð", "Þ"]
    if not any(m in s for m in bad_markers):
        return s
    try:
        repaired = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        # jeśli po naprawie wciąż wygląda źle, zostaw oryginał
        if repaired and repaired != s:
            return repaired
        return s
    except Exception:
        return s


def _fix_dataframe_text(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).apply(_fix_mojibake)
    return out


def _read_urls_from_uploaded_csv(file) -> list[str]:
    df = _try_read_csv_with_encodings(file)
    if df.empty:
        return []
    if "URL" in df.columns:
        urls = df["URL"].astype(str).tolist()
    else:
        urls = df.iloc[:, 0].astype(str).tolist()
    urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    return urls


def _extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines()]
    urls = []
    for ln in lines:
        if not ln:
            continue
        if ln.startswith("http://") or ln.startswith("https://"):
            urls.append(ln)
    return urls


def _progress_callback_factory(status_el, bar_el):
    def cb(done, total, message):
        if total and total > 0:
            bar_el.progress(min(1.0, max(0.0, done / total)))
            status_el.write(message)
        else:
            status_el.write(message)
    return cb


def _download_csv_button(df: pd.DataFrame, label: str, filename: str):
    """
    Zapis w utf-8-sig: działa w Excelu bez krzaków.
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")


def _plot_similarity_hist(sim: np.ndarray, title: str):
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
# Sidebar
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

max_pages = st.sidebar.number_input("Limit URL-i / artykułów", 1, 2000, 200, 10)
delay = st.sidebar.slider("Opóźnienie między requestami (sekundy)", 0.0, 3.0, 0.6, 0.1)
timeout = st.sidebar.number_input("Timeout requestu (sekundy)", 5, 120, 20, 5)

same_subdomain_only = st.sidebar.checkbox(
    "Tylko ta sama subdomena (stricte ten sam host)",
    value=True,
    help="Jeśli odznaczysz, scraper może brać też subdomeny w ramach tej samej domeny głównej.",
)

st.sidebar.header("2) Ustawienia podobieństwa (w %)")
method = st.sidebar.selectbox(
    "Metoda porównania",
    ["hybrid", "word_tfidf", "char_tfidf"],
    index=0,
    help="Hybrid jest zwykle najlepszy do kanibalizacji SEO.",
)

threshold_pct = st.sidebar.slider("Próg podobieństwa (%)", 0, 100, 30, 1)
boiler_df_pct = st.sidebar.slider("Usuwanie boilerplate – próg (%)", 5, 60, 25, 5)
min_words = st.sidebar.number_input("Min. liczba słów (po czyszczeniu)", 10, 500, 40, 10)
max_pairs = st.sidebar.number_input("Limit par w raporcie", 100, 20000, 2000, 100)

st.sidebar.header("3) Uruchomienie")
run_btn = st.sidebar.button("🚀 Start: pobierz i policz podobieństwo", type="primary")


with st.expander("Instrukcja: jak rozumieć próg podobieństwa (dla nietechnicznych)", expanded=True):
    st.markdown(interpretation_help_text())


manual_urls_text = ""
uploaded_csv = None

if mode == "Wklej URL-e ręcznie":
    manual_urls_text = st.text_area(
        "Wklej URL-e (1 linia = 1 URL)",
        height=180,
        placeholder="https://marczak.me/jakis-artykul/\nhttps://marczak.me/inny-artykul/",
    )

if mode == "Wgraj CSV z URL-ami":
    uploaded_csv = st.file_uploader(
        "Wgraj CSV z URL-ami (kolumna 'URL' lub pierwsza kolumna). "
        "Obsługiwane kodowania: UTF-8 (zalecane), Windows-1250.",
        type=["csv"],
    )


# -------------------------
# Run
# -------------------------
if run_btn:
    base_url = (base_url or "").strip()
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        st.error("Base URL musi zaczynać się od http:// lub https://")
        st.stop()

    status = st.empty()
    bar = st.progress(0.0)
    cb = _progress_callback_factory(status, bar)

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

    if articles_df is None or articles_df.empty:
        st.warning("Nie udało się pobrać żadnych artykułów.")
        st.stop()

    # Napraw polskie znaki w danych (na wszelki wypadek) przed wyświetleniem i eksportem
    articles_df = _fix_dataframe_text(articles_df, ["H1", "title", "treść w Markdown"])

    st.subheader("1) Dane artykułów (CSV: URL, H1, title, treść w Markdown)")
    st.dataframe(articles_df, use_container_width=True, height=300)
    _download_csv_button(articles_df, "⬇️ Pobierz CSV z artykułami (UTF-8)", "articles.csv")

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

    pairs_df = similarity_pairs_report(articles_df, sim, threshold_pct=cfg.similarity_threshold_pct, max_pairs=cfg.max_pairs)
    groups_df = similarity_groups_report(articles_df, sim, threshold_pct=cfg.similarity_threshold_pct)

    # Fix mojibake in reports too
    pairs_df = _fix_dataframe_text(pairs_df, ["title_a", "h1_a", "title_b", "h1_b"])
    groups_df = _fix_dataframe_text(groups_df, ["urls"])

    st.markdown(f"### 2.1 Pary artykułów powyżej progu: **{threshold_pct}%** ({method})")
    if pairs_df is None or pairs_df.empty:
        st.info("Brak par powyżej progu. Obniż próg (np. 20–30%) albo użyj 'hybrid'.")
    else:
        st.dataframe(pairs_df, use_container_width=True, height=350)
        _download_csv_button(pairs_df, "⬇️ Pobierz CSV: pary podobnych artykułów (UTF-8)", "similarity_pairs.csv")

    st.markdown(f"### 2.2 Grupy (klastry) powyżej progu: **{threshold_pct}%** ({method})")
    if groups_df is None or groups_df.empty:
        st.info("Brak grup (min. 2 URL-e połączone podobieństwem >= próg).")
    else:
        st.dataframe(groups_df, use_container_width=True, height=300)
        _download_csv_button(groups_df, "⬇️ Pobierz CSV: grupy kanibalizacji (UTF-8)", "similarity_groups.csv")

    st.success("Gotowe ✅")
