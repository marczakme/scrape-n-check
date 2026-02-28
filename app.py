import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

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
)

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Content Similarity Audit", layout="wide")

ACCENT = "#d43584"  # marczak.me

# -------------------------
# CSS
# -------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}}

h1 {{ font-size: 1.45rem !important; margin: 0.35rem 0 0.35rem 0 !important; }}
h2 {{ font-size: 1.15rem !important; margin: 0.65rem 0 0.35rem 0 !important; }}
h3 {{ font-size: 1.00rem !important; margin: 0.6rem 0 0.3rem 0 !important; }}

small {{
  color: rgba(0,0,0,0.60);
  font-size: 0.86rem;
}}

[data-testid="stMainBlockContainer"] {{
  padding-top: 0.95rem;
  padding-bottom: 0.75rem;
  max-height: 900px;
  overflow-y: auto;
}}
[data-testid="stAppViewContainer"] {{
  max-height: 900px;
}}

div.stButton > button {{
  background: {ACCENT} !important;
  color: #ffffff !important;
  border: 1px solid {ACCENT} !important;
  border-radius: 12px !important;
  padding: 0.55rem 0.9rem !important;
  font-weight: 700 !important;
}}
div.stButton > button:hover {{
  filter: brightness(0.95);
}}

div[data-testid="stVerticalBlock"] div[data-testid="stContainer"] {{
  border-radius: 16px !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  background: rgba(255,255,255,0.88) !important;
  box-shadow: 0 4px 18px rgba(0,0,0,0.04) !important;
  padding: 12px 12px 10px 12px !important;
}}

label, .stMarkdown p {{
  font-size: 0.92rem !important;
}}

div[data-testid="stDataFrame"] {{
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(0,0,0,0.08);
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 10px; }}

.step-title {{
  font-weight: 800;
  font-size: 0.95rem;
  margin: 0 0 0.4rem 0;
}}
.mini-muted {{
  color: rgba(0,0,0,0.55);
  font-size: 0.86rem;
}}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------
def _try_read_csv_with_encodings(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "cp1250", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def _download_csv_button(df: pd.DataFrame, label: str, filename: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")


def _extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln.startswith(("http://", "https://"))]


def _read_urls_from_uploaded_csv(file) -> list[str]:
    df = _try_read_csv_with_encodings(file)
    if df.empty:
        return []
    if "URL" in df.columns:
        urls = df["URL"].astype(str).tolist()
    else:
        urls = df.iloc[:, 0].astype(str).tolist()
    urls = [u.strip() for u in urls if u and str(u).startswith(("http://", "https://"))]
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _parse_sitemap_xml(xml_text: str) -> list[str]:
    if not xml_text:
        return []
    soup = BeautifulSoup(xml_text, "xml")
    locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
    urls = [u.strip() for u in locs if u and u.strip().startswith(("http://", "https://"))]
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _read_sitemap_from_upload(uploaded_file) -> list[str]:
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".xml"):
        try:
            txt = raw.decode("utf-8", errors="ignore")
        except Exception:
            txt = raw.decode("latin1", errors="ignore")
        return _parse_sitemap_xml(txt)

    df = _try_read_csv_with_encodings(uploaded_file)
    if df is None or df.empty:
        return []
    if "URL" in df.columns:
        urls = df["URL"].astype(str).tolist()
    else:
        urls = df.iloc[:, 0].astype(str).tolist()
    urls = [u.strip() for u in urls if u and str(u).startswith(("http://", "https://"))]
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _progress_callback_factory(status_el, bar_el, log_list: list[str]):
    def cb(done, total, message):
        if message:
            log_list.append(str(message))
        if total and total > 0:
            bar_el.progress(min(1.0, max(0.0, done / total)))
            status_el.write(message)
        else:
            status_el.write(message)
    return cb


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


def _preset_settings(preset_name: str):
    presets = {
        "kanibalizacja": ("hybrid", 30, 25),
        "overlap tematu": ("word_tfidf", 22, 25),
        "duplikacja": ("char_tfidf", 75, 20),
        "ostrożnie": ("hybrid", 38, 30),
    }
    return presets.get(preset_name, ("hybrid", 30, 25))


def _urls_from_groups_row(urls_cell: str) -> list[str]:
    if not isinstance(urls_cell, str) or not urls_cell.strip():
        return []
    return [u.strip() for u in urls_cell.splitlines() if u.strip()]


def _make_group_export_df(group_id: int, urls: list[str], primary_url: str = "") -> pd.DataFrame:
    rows = []
    for u in urls:
        rows.append({"group_id": group_id, "url": u, "is_primary": "YES" if primary_url and u == primary_url else ""})
    return pd.DataFrame(rows, columns=["group_id", "url", "is_primary"])


def _gsc_checklist_text(primary_url: str, urls: list[str]) -> str:
    other_count = len([u for u in urls if u != primary_url])
    return (
        "### Cel\n"
        "Sprawdzamy, czy strony w grupie konkurują o te same zapytania, przez co Google miesza URL-e w wynikach.\n\n"
        "### Krok 1\n"
        "- GSC → Skuteczność → Wyniki wyszukiwania\n"
        "- Zakres: 3 miesiące, potem 16 miesięcy\n"
        "- Metryki: Kliknięcia, Wyświetlenia, CTR, Śr. pozycja\n\n"
        "### Krok 2\n"
        "- Filtr Strona = Primary URL → zobacz Zapytania\n"
        "- Porównaj z pozostałymi URL-ami\n\n"
        "### Krok 3\n"
        "- Weź top zapytania z Primary URL\n"
        "- Dla każdego: filtr Zapytanie → zakładka Strony\n"
        "- Jeśli pojawiają się 2–3 URL-e z grupy na to samo zapytanie, to silny sygnał kanibalizacji\n\n"
        "### Decyzja\n"
        "Jeśli intencja ta sama: scal treść do Primary URL + 301/canonical + internal linking.\n"
        "Jeśli intencje różne: rozdziel zakres + doprecyzuj title/H1 + linki między artykułami.\n\n"
        f"Primary URL: {primary_url or '(nie wybrano)'}\n"
        f"Pozostałe URL-e: {other_count}\n"
    )


# -------------------------
# Session state
# -------------------------
for key in ["articles_df", "pairs_df", "groups_df", "sim_matrix", "run_log"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "run_log" else []

# -------------------------
# Header
# -------------------------
st.markdown("## Content Similarity Audit")
st.markdown("<small>Scrape → Markdown → Similarity → Cannibalization</small>", unsafe_allow_html=True)
st.markdown("")

# -------------------------
# TOP ROW: config (left) + empty/help (right)
# -------------------------
left, right = st.columns([1.05, 1.0], gap="large")

# We'll render results in a FULL-WIDTH section below.
# Progress/status/CTA will live in its own left-card.

# Placeholders for status UI (will be created inside the "Start" card)
status_box = None
progress_bar = None
error_box = None
log_box = None

# Inputs store
with left:
    with st.container(border=True):
        r1a, r1b = st.columns([0.34, 0.66], vertical_alignment="center")
        with r1a:
            st.markdown('<div class="step-title">1. Podaj adres strony</div>', unsafe_allow_html=True)
        with r1b:
            base_url = st.text_input(
                label="",
                value="https://marczak.me",
                placeholder="https://marczak.me",
                label_visibility="collapsed",
            ).strip()

    with st.container(border=True):
        r2a, r2b = st.columns([0.34, 0.66], vertical_alignment="center")
        with r2a:
            st.markdown('<div class="step-title">Źródło URL-i</div>', unsafe_allow_html=True)
        with r2b:
            mode = st.radio(
                label="",
                options=[
                    "auto (sitemap, rss)",
                    "url sitemapy",
                    "csv z sitemapą",
                    "wklej URL-e",
                    "wgraj CSV z URL-ami",
                ],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
            )

        sitemap_url = ""
        sitemap_upload = None
        manual_urls_text = ""
        uploaded_urls_csv = None

        if mode == "url sitemapy":
            sitemap_url = st.text_input(
                label="",
                placeholder="URL sitemapy (np. https://marczak.me/sitemap.xml)",
                label_visibility="collapsed",
            ).strip()

        if mode == "csv z sitemapą":
            sitemap_upload = st.file_uploader(
                label="Wgraj sitemapę jako XML lub CSV",
                type=["xml", "csv"],
            )

        if mode == "wklej URL-e":
            manual_urls_text = st.text_area(
                label="",
                height=110,
                placeholder="Wklej URL-e (1 linia = 1 URL)",
                label_visibility="collapsed",
            )

        if mode == "wgraj CSV z URL-ami":
            uploaded_urls_csv = st.file_uploader(
                label="Wgraj CSV z URL-ami (kolumna URL lub pierwsza kolumna)",
                type=["csv"],
            )

    with st.container(border=True):
        st.markdown('<div class="step-title">2. Ustawienia scrapowania</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        max_pages = c1.number_input("Limit artykułów", 1, 2000, 200, 10)
        timeout = c2.number_input("Timeout (s)", 5, 120, 20, 5)
        delay = c3.slider("Delay (s)", 0.0, 3.0, 0.6, 0.1)
        same_subdomain_only = st.checkbox("Tylko ta sama subdomena", value=True)

    with st.container(border=True):
        st.markdown('<div class="step-title">3. Analiza podobieństwa</div>', unsafe_allow_html=True)
        preset = st.selectbox("Preset", options=["kanibalizacja", "overlap tematu", "duplikacja", "ostrożnie"], index=0)
        preset_method, preset_thr, preset_boiler = _preset_settings(preset)

        colx, coly = st.columns([1, 1])
        method = colx.selectbox(
            "Metoda",
            ["hybrid", "word_tfidf", "char_tfidf"],
            index=["hybrid", "word_tfidf", "char_tfidf"].index(preset_method),
        )
        threshold_pct = coly.slider("Próg (%)", 0, 100, int(preset_thr), 1)

        colb1, colb2, colb3 = st.columns(3)
        boiler_df_pct = colb1.slider("Boilerplate (%)", 5, 60, int(preset_boiler), 5)
        min_words = colb2.number_input("Min słów", 10, 500, 40, 10)
        max_pairs = colb3.number_input("Limit par", 100, 20000, 2000, 100)

    # START / STATUS CARD (same width as left)
    with st.container(border=True):
        st.markdown('<div class="step-title">Start / status</div>', unsafe_allow_html=True)
        st.markdown('<div class="mini-muted">Tu pojawią się błędy, postęp i log.</div>', unsafe_allow_html=True)

        run_btn = st.button("Rozpocznij analizę", type="primary")

        # Dedicated UI slots
        error_box = st.empty()
        progress_bar = st.progress(0.0)
        status_box = st.empty()
        log_box = st.empty()


with right:
    # Keep this area light/empty (you can add later tips)
    with st.container(border=True):
        st.markdown('<div class="step-title">Podpowiedzi</div>', unsafe_allow_html=True)
        st.markdown(
            "- Jeśli nic nie wychodzi przy 40–50%, spróbuj progu 20–30% i metody **hybrid**.\n"
            "- Najbardziej użyteczne w praktyce są **grupy**, a potem weryfikacja w **GSC**.\n",
        )

# -------------------------
# RESULTS SECTION (FULL WIDTH BELOW)
# -------------------------
st.markdown("")
with st.container(border=True):
    st.markdown('<div class="step-title">Wyniki</div>', unsafe_allow_html=True)

    a = st.session_state.articles_df
    p = st.session_state.pairs_df
    g = st.session_state.groups_df

    m1, m2, m3 = st.columns(3)
    m1.metric("Artykuły", "—" if a is None else len(a))
    m2.metric("Pary", "—" if p is None else len(p))
    m3.metric("Grupy", "—" if g is None else len(g))

    if a is None:
        st.info("Uruchom analizę, żeby zobaczyć wyniki.")
    else:
        tabs = st.tabs(["GSC workflow", "Pary", "Grupy", "Artykuły", "Eksport", "Diagnostyka"])

        with tabs[0]:
            st.markdown("### GSC workflow")
            if g is None or g.empty:
                st.info("Brak grup. Obniż próg (np. 20–30%) lub użyj presetu kanibalizacja.")
            else:
                group_options = g["group_id"].astype(int).tolist()
                selected_gid = st.selectbox("Wybierz grupę", options=group_options, index=0)

                row = g[g["group_id"] == selected_gid].iloc[0]
                urls = _urls_from_groups_row(row["urls"])
                st.write(f"Rozmiar grupy: **{int(row['size'])}**")

                primary_url = st.selectbox("Primary URL", options=urls, index=0 if urls else None)

                # compact table: url/title/h1
                titles_map, h1_map = {}, {}
                if a is not None and not a.empty:
                    a_map = a.set_index("URL")
                    for u in urls:
                        if u in a_map.index:
                            titles_map[u] = str(a_map.loc[u, "title"])
                            h1_map[u] = str(a_map.loc[u, "H1"])
                        else:
                            titles_map[u] = ""
                            h1_map[u] = ""

                table_rows = []
                for u in urls:
                    table_rows.append(
                        {"primary": "✅" if u == primary_url else "", "url": u, "title": titles_map.get(u, ""), "h1": h1_map.get(u, "")}
                    )
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=240)

                export_df = _make_group_export_df(selected_gid, urls, primary_url=primary_url)
                _download_csv_button(export_df, "Pobierz CSV tej grupy", f"gsc_group_{selected_gid}.csv")

                with st.expander("Co sprawdzić i jak interpretować wyniki?", expanded=False):
                    st.markdown(_gsc_checklist_text(primary_url, urls))

        with tabs[1]:
            st.markdown(f"### Pary powyżej progu: {threshold_pct}% ({method})")
            if p is None or p.empty:
                st.info("Brak par powyżej progu. Zmniejsz próg lub zmień preset.")
            else:
                st.dataframe(p, use_container_width=True, height=360)

        with tabs[2]:
            st.markdown(f"### Grupy powyżej progu: {threshold_pct}% ({method})")
            if g is None or g.empty:
                st.info("Brak grup.")
            else:
                st.dataframe(g, use_container_width=True, height=360)

        with tabs[3]:
            st.markdown("### Artykuły (Markdown)")
            st.dataframe(a, use_container_width=True, height=360)

        with tabs[4]:
            st.markdown("### Eksport")
            if a is not None and not a.empty:
                _download_csv_button(a, "articles.csv", "articles.csv")
            if p is not None and not p.empty:
                _download_csv_button(p, "similarity_pairs.csv", "similarity_pairs.csv")
            if g is not None and not g.empty:
                _download_csv_button(g, "similarity_groups.csv", "similarity_groups.csv")

        with tabs[5]:
            st.markdown("### Diagnostyka")
            if st.session_state.sim_matrix is not None:
                sim = st.session_state.sim_matrix
                _plot_similarity_hist(sim, title=f"Similarity distribution ({method})")
            log = (st.session_state.run_log or [])[-200:]
            st.code("\n".join(log) if log else "Brak logu.")

# -------------------------
# Run pipeline (uses the Start/status box UI)
# -------------------------
if run_btn:
    st.session_state.run_log = []
    if error_box:
        error_box.empty()
    if status_box:
        status_box.empty()
    if log_box:
        log_box.empty()
    if progress_bar:
        progress_bar.progress(0.0)

    def push_log():
        log = (st.session_state.run_log or [])[-200:]
        if log_box is not None:
            log_box.code("\n".join(log) if log else "Log pojawi się tutaj…")

    # validate base url (show error INSIDE the start/status card)
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        if error_box is not None:
            error_box.error("Adres strony musi zaczynać się od http:// lub https://")
        push_log()
        st.stop()

    # build callback wired to our UI
    cb = _progress_callback_factory(status_box, progress_bar, st.session_state.run_log)

    try:
        if mode == "auto (sitemap, rss)":
            cb(0, 1, "Start: auto-wykrywanie URL-i i pobieranie artykułów…")
            articles_df = scrape_site_articles(
                base_url=base_url,
                max_pages=int(max_pages),
                timeout=int(timeout),
                delay=float(delay),
                max_depth=3,
                same_subdomain_only=bool(same_subdomain_only),
                progress_callback=cb,
            )

        elif mode == "url sitemapy":
            if not sitemap_url:
                error_box.error("Podaj URL sitemapy.")
                push_log()
                st.stop()

            cb(0, 1, f"Pobieram sitemapę: {sitemap_url}")
            r = requests.get(sitemap_url, timeout=int(timeout), headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code >= 400:
                error_box.error(f"Nie udało się pobrać sitemapy (HTTP {r.status_code}).")
                push_log()
                st.stop()

            urls = _parse_sitemap_xml(r.text)
            urls = [normalize_url_public(u) for u in urls]
            urls = filter_internal_urls_public(urls, base_url=base_url, same_subdomain_only=bool(same_subdomain_only))
            if not urls:
                error_box.error("W sitemapie nie znaleziono URL-i pasujących do domeny / filtra.")
                push_log()
                st.stop()

            cb(0, 1, f"Start: pobieranie treści dla {min(len(urls), int(max_pages))} URL-i z sitemapy…")
            articles_df = scrape_articles_from_urls(
                base_url=base_url,
                urls=urls,
                max_pages=int(max_pages),
                timeout=int(timeout),
                delay=float(delay),
                same_subdomain_only=bool(same_subdomain_only),
                progress_callback=cb,
            )

        elif mode == "csv z sitemapą":
            if sitemap_upload is None:
                error_box.error("Wgraj sitemapę jako XML lub CSV.")
                push_log()
                st.stop()

            urls = _read_sitemap_from_upload(sitemap_upload)
            urls = [normalize_url_public(u) for u in urls]
            urls = filter_internal_urls_public(urls, base_url=base_url, same_subdomain_only=bool(same_subdomain_only))
            if not urls:
                error_box.error("W pliku sitemapy nie znaleziono URL-i pasujących do domeny / filtra.")
                push_log()
                st.stop()

            cb(0, 1, f"Start: pobieranie treści dla {min(len(urls), int(max_pages))} URL-i z pliku sitemapy…")
            articles_df = scrape_articles_from_urls(
                base_url=base_url,
                urls=urls,
                max_pages=int(max_pages),
                timeout=int(timeout),
                delay=float(delay),
                same_subdomain_only=bool(same_subdomain_only),
                progress_callback=cb,
            )

        elif mode == "wklej URL-e":
            urls = _extract_urls_from_text(manual_urls_text)
            urls = [normalize_url_public(u) for u in urls]
            urls = filter_internal_urls_public(urls, base_url=base_url, same_subdomain_only=bool(same_subdomain_only))
            if not urls:
                error_box.error("Nie wykryto poprawnych URL-i do analizy.")
                push_log()
                st.stop()

            cb(0, 1, f"Start: pobieranie treści dla {min(len(urls), int(max_pages))} URL-i…")
            articles_df = scrape_articles_from_urls(
                base_url=base_url,
                urls=urls,
                max_pages=int(max_pages),
                timeout=int(timeout),
                delay=float(delay),
                same_subdomain_only=bool(same_subdomain_only),
                progress_callback=cb,
            )

        else:  # wgraj CSV z URL-ami
            if uploaded_urls_csv is None:
                error_box.error("Wgraj CSV z URL-ami.")
                push_log()
                st.stop()

            urls = _read_urls_from_uploaded_csv(uploaded_urls_csv)
            urls = [normalize_url_public(u) for u in urls]
            urls = filter_internal_urls_public(urls, base_url=base_url, same_subdomain_only=bool(same_subdomain_only))
            if not urls:
                error_box.error("W CSV nie znaleziono URL-i pasujących do domeny / filtra.")
                push_log()
                st.stop()

            cb(0, 1, f"Start: pobieranie treści dla {min(len(urls), int(max_pages))} URL-i z CSV…")
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
        error_box.exception(e)
        push_log()
        st.stop()

    if articles_df is None or articles_df.empty:
        error_box.warning("Nie udało się pobrać żadnych artykułów.")
        push_log()
        st.stop()

    cfg = SimilarityConfig(
        similarity_threshold_pct=float(threshold_pct),
        method=method,
        boilerplate_line_df_pct=float(boiler_df_pct),
        min_words_per_doc=int(min_words),
        max_pairs=int(max_pairs),
    )

    texts = articles_df["treść w Markdown"].fillna("").astype(str).tolist()

    with st.spinner("Liczę podobieństwa…"):
        mats = build_similarity_matrices(texts, cfg)

    sim = mats["hybrid"] if method == "hybrid" else mats["word"] if method == "word_tfidf" else mats["char"]

    pairs_df = similarity_pairs_report(articles_df, sim, threshold_pct=float(threshold_pct), max_pairs=int(max_pairs))
    groups_df = similarity_groups_report(articles_df, sim, threshold_pct=float(threshold_pct))

    st.session_state.articles_df = articles_df
    st.session_state.pairs_df = pairs_df
    st.session_state.groups_df = groups_df
    st.session_state.sim_matrix = sim

    progress_bar.progress(1.0)
    status_box.success("Gotowe ✅")
    push_log()
    st.rerun()
