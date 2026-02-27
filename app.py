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

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="SEO Content Auditor", layout="wide")

st.title("SEO Content Auditor")
st.caption("Scrape → Markdown → Similarity → Cannibalization → GSC workflow (dla nietechnicznych)")

# -------------------------
# Helpers: encoding + mojibake fix
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


def _fix_mojibake(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    bad_markers = ["Ã", "Å", "Ä", "â", "Ê", "Ë", "Ð", "Þ"]
    if not any(m in s for m in bad_markers):
        return s
    try:
        repaired = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
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


def _download_csv_button(df: pd.DataFrame, label: str, filename: str):
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


def _preset_settings(preset_name: str):
    presets = {
        "Kanibalizacja (polecane)": ("hybrid", 30, 25),
        "Szeroki overlap tematu": ("word_tfidf", 22, 25),
        "Duplikacja / copy-paste": ("char_tfidf", 75, 20),
        "Ostrożnie (mniej false positives)": ("hybrid", 38, 30),
    }
    return presets.get(preset_name, ("hybrid", 30, 25))


def _urls_from_groups_row(urls_cell: str) -> list[str]:
    if not isinstance(urls_cell, str) or not urls_cell.strip():
        return []
    return [u.strip() for u in urls_cell.splitlines() if u.strip()]


def _make_group_export_df(group_id: int, urls: list[str], primary_url: str = "") -> pd.DataFrame:
    rows = []
    for u in urls:
        rows.append({
            "group_id": group_id,
            "url": u,
            "is_primary": "YES" if primary_url and u == primary_url else "",
        })
    return pd.DataFrame(rows, columns=["group_id", "url", "is_primary"])


def _gsc_checklist_text(primary_url: str, other_urls: list[str]) -> str:
    other_count = len([u for u in other_urls if u != primary_url])
    return (
        "## GSC workflow: co sprawdzić i jak to interpretować\n\n"
        "### Cel\n"
        "Sprawdzamy, czy strony w grupie **konkurują o te same zapytania**, przez co Google miesza URL-e w wynikach (kanibalizacja).\n\n"
        "### Krok 1 — Ustaw kontekst\n"
        "- Wejdź w **Google Search Console → Skuteczność (Wyniki wyszukiwania)**\n"
        "- Zakres dat: zacznij od **3 miesięcy**, potem porównaj z **16 miesięcy** (wahania sezonowe)\n"
        "- Włącz metryki: **Kliknięcia, Wyświetlenia, CTR, Śr. pozycja**\n\n"
        "### Krok 2 — Sprawdź, czy to ta sama intencja\n"
        "Dla grupy wybierz jedną stronę jako **Primary URL** (docelową). Potem porównaj z pozostałymi.\n\n"
        "W GSC zrób:\n"
        "1) Filtr **Strona** = Primary URL → zobacz zapytania (Queries)\n"
        "2) Dodaj filtr **Strona** = kolejny URL → porównaj zapytania\n\n"
        "Sygnały kanibalizacji:\n"
        "- Te same lub bardzo podobne zapytania pojawiają się na kilku URL-ach\n"
        "- Wykresy kliknięć/wyświetleń „przechodzą” między URL-ami w czasie\n"
        "- CTR spada, bo Google testuje różne strony\n\n"
        "Sygnały, że to *nie* kanibalizacja (tylko podobny temat):\n"
        "- Zapytania są inne (np. jeden URL informacyjny, drugi transakcyjny)\n"
        "- Różne frazy long-tail, różne intencje\n\n"
        "### Krok 3 — Sprawdź „mieszanie URL-i” na tych samych zapytaniach\n"
        "Najpraktyczniej:\n"
        "- Weź 5–20 głównych zapytań z Primary URL\n"
        "- Dla każdego zapytania włącz filtr **Zapytanie** i zobacz zakładkę **Strony**\n"
        "- Jeśli dla tego samego zapytania wyświetlają się 2–3 URL-e z grupy → to silny sygnał kanibalizacji\n\n"
        "### Krok 4 — Decyzja: co robimy\n"
        "Jeśli intencja jest ta sama (kanibalizacja):\n"
        "1) **Scal treść** (merge) do Primary URL, a pozostałe:\n"
        "   - przekieruj 301 lub ustaw canonical do primary (zależnie od sytuacji)\n"
        "2) Wzmocnij internal linking do Primary URL (anchor + kontekst)\n"
        "3) Ujednolić tytuły i H1: żeby Primary URL jasno komunikował temat\n\n"
        "Jeśli intencje są różne (nie kanibalizacja, tylko overlap):\n"
        "1) **Rozdziel temat**: doprecyzuj zakres i sekcje (np. „dla kogo”, „porównanie”, „krok po kroku”)\n"
        "2) Przepisz title/H1 tak, by się nie dublowały\n"
        "3) Dodaj linki między artykułami (nawigacja tematyczna)\n\n"
        f"### Twoja grupa\n"
        f"- Primary URL: {primary_url or '(nie wybrano)'}\n"
        f"- Pozostałe URL-e: {other_count}\n"
    )


# -------------------------
# Session state
# -------------------------
if "articles_df" not in st.session_state:
    st.session_state.articles_df = None
if "pairs_df" not in st.session_state:
    st.session_state.pairs_df = None
if "groups_df" not in st.session_state:
    st.session_state.groups_df = None
if "sim_matrix" not in st.session_state:
    st.session_state.sim_matrix = None
if "run_log" not in st.session_state:
    st.session_state.run_log = []


# -------------------------
# Optional help (collapsed)
# -------------------------
with st.expander("Instrukcja: jak rozumieć próg podobieństwa (dla nietechnicznych)", expanded=False):
    st.markdown(interpretation_help_text())

st.markdown("---")

left, right = st.columns([1.05, 1.0], gap="large")

# -------------------------
# LEFT: Input + run
# -------------------------
with left:
    st.subheader("Krok 1: Źródło URL-i")

    base_url = st.text_input(
        "Adres strony (base URL)",
        value="https://marczak.me",
        help="Przykład: https://marczak.me",
    ).strip()

    mode = st.radio(
        "Jak chcesz dostarczyć URL-e?",
        options=[
            "Auto (sitemap/RSS) – pobierz artykuły ze strony",
            "Wklej URL-e ręcznie",
            "Wgraj CSV z URL-ami",
        ],
        index=0,
    )

    manual_urls_text = ""
    uploaded_csv = None

    if mode == "Wklej URL-e ręcznie":
        manual_urls_text = st.text_area(
            "Wklej URL-e (1 linia = 1 URL)",
            height=160,
            placeholder="https://marczak.me/jakis-artykul/\nhttps://marczak.me/inny-artykul/",
        )

    if mode == "Wgraj CSV z URL-ami":
        uploaded_csv = st.file_uploader(
            "Wgraj CSV (kolumna `URL` lub pierwsza kolumna). UTF-8 zalecane.",
            type=["csv"],
        )

    st.subheader("Krok 2: Scrapowanie")
    c1, c2, c3 = st.columns(3)
    max_pages = c1.number_input("Limit artykułów", 1, 2000, 200, 10)
    timeout = c2.number_input("Timeout (s)", 5, 120, 20, 5)
    delay = c3.slider("Delay (s)", 0.0, 3.0, 0.6, 0.1)

    same_subdomain_only = st.checkbox(
        "Tylko ta sama subdomena (ten sam host)",
        value=True,
        help="Jeśli odznaczysz, scraper może brać też subdomeny w ramach tej samej domeny głównej.",
    )

    st.subheader("Krok 3: Analiza podobieństwa")
    preset = st.selectbox(
        "Szybkie ustawienia",
        options=[
            "Kanibalizacja (polecane)",
            "Szeroki overlap tematu",
            "Duplikacja / copy-paste",
            "Ostrożnie (mniej false positives)",
        ],
        index=0,
    )
    preset_method, preset_thr, preset_boiler = _preset_settings(preset)

    colx, coly = st.columns([1, 1])
    method = colx.selectbox("Metoda", ["hybrid", "word_tfidf", "char_tfidf"],
                            index=["hybrid", "word_tfidf", "char_tfidf"].index(preset_method))
    threshold_pct = coly.slider("Próg (%)", 0, 100, int(preset_thr), 1)

    colb1, colb2, colb3 = st.columns(3)
    boiler_df_pct = colb1.slider("Boilerplate (%)", 5, 60, int(preset_boiler), 5,
                                 help="Usuń linie powtarzalne w >=X% dokumentów.")
    min_words = colb2.number_input("Min słów", 10, 500, 40, 10)
    max_pairs = colb3.number_input("Limit par", 100, 20000, 2000, 100)

    st.markdown("")
    run_btn = st.button("🚀 Uruchom cały proces", type="primary")

    st.markdown("---")
    st.caption("Tip: jeśli nic nie pokazuje przy 40–50%, spróbuj 20–30% w trybie hybrid.")

# -------------------------
# RIGHT: Results
# -------------------------
with right:
    st.subheader("Wyniki")

    a = st.session_state.articles_df
    p = st.session_state.pairs_df
    g = st.session_state.groups_df

    m1, m2, m3 = st.columns(3)
    m1.metric("Artykuły", "—" if a is None else len(a))
    m2.metric("Pary", "—" if p is None else len(p))
    m3.metric("Grupy", "—" if g is None else len(g))

    if a is None:
        st.info("Uruchom proces po lewej stronie, żeby zobaczyć wyniki.")
    else:
        tabs = st.tabs(["GSC workflow", "Pary", "Grupy", "Artykuły", "Diagnostyka", "Eksport"])

        # -------------------------
        # GSC WORKFLOW TAB
        # -------------------------
        with tabs[0]:
            st.markdown("## GSC workflow (praca na grupach kanibalizacji)")

            if g is None or g.empty:
                st.info("Brak grup. Spróbuj obniżyć próg (np. 20–30%) lub użyj presetu „Kanibalizacja (polecane)”.")
            else:
                # wybór grupy
                group_options = g["group_id"].astype(int).tolist()
                selected_gid = st.selectbox("Wybierz grupę (cluster)", options=group_options, index=0)

                row = g[g["group_id"] == selected_gid].iloc[0]
                urls = _urls_from_groups_row(row["urls"])
                size = int(row["size"])

                st.write(f"**Rozmiar grupy:** {size} URL-i")

                # dodatkowe info z articles_df
                info_rows = []
                titles_map = {}
                h1_map = {}
                if a is not None and not a.empty:
                    a_map = a.set_index("URL")
                    for u in urls:
                        if u in a_map.index:
                            titles_map[u] = str(a_map.loc[u, "title"])
                            h1_map[u] = str(a_map.loc[u, "H1"])
                        else:
                            titles_map[u] = ""
                            h1_map[u] = ""

                # wybór primary URL
                default_primary = urls[0] if urls else ""
                primary_url = st.selectbox("Wybierz Primary URL (docelowa strona)", options=urls, index=0)

                st.markdown("### URL-e w grupie")
                table_rows = []
                for u in urls:
                    table_rows.append({
                        "is_primary": "✅" if u == primary_url else "",
                        "url": u,
                        "title": titles_map.get(u, ""),
                        "h1": h1_map.get(u, ""),
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=260)

                # eksport grupy
                export_df = _make_group_export_df(selected_gid, urls, primary_url=primary_url)
                _download_csv_button(export_df, "⬇️ Pobierz CSV tej grupy (do GSC)", f"gsc_group_{selected_gid}.csv")

                # checklist
                st.markdown(_gsc_checklist_text(primary_url, urls))

                st.markdown("### Szybka interpretacja (decyzja w 2 min)")
                st.markdown(
                    """
**Jeśli widzisz w GSC, że te same zapytania pokazują różne URL-e z grupy** (zakładka: Zapytanie → Strony),  
to jest to najczęściej **realna kanibalizacja**.

**Najczęstsze decyzje:**
- **Merge (scalenie)**: jeden mocny artykuł (Primary URL) + 301 / canonical z pozostałych
- **Split (rozdzielenie intencji)**: doprecyzowanie tematów + title/H1 + linkowanie wewnętrzne

**Kiedy merge jest “oczywisty”:**
- 2 strony mają bardzo podobne title/H1,
- rankują na te same zapytania,
- w czasie Google przełącza URL-e.

**Kiedy split jest “oczywisty”:**
- jedna strona odpowiada na “co to jest / poradnik”, druga na “ranking / narzędzie / cennik”,
- zapytania mają inną intencję,
- strony spełniają różne potrzeby użytkownika.
"""
                )

                st.markdown("### Gotowy mini-brief dla osoby wdrożeniowej")
                st.info(
                    "Skopiuj poniższy tekst do zadania w Jira/Asana/Notion i uzupełnij po analizie w GSC."
                )
                brief = f"""[KANIBALIZACJA] Grupa {selected_gid}

Primary URL: {primary_url}

Pozostałe URL-e:
{chr(10).join([u for u in urls if u != primary_url])}

Co sprawdzić w GSC:
1) Performance (3m + 16m) → Queries dla primary i pozostałych
2) Dla top zapytań → Query filter → Pages (czy Google miesza URL-e)
3) CTR / pozycja / wahania URL-i w czasie

Decyzja (uzupełnij):
- [ ] Merge (scalenie) → 301/canonical
- [ ] Split (rozdzielenie intencji) → zmiany title/H1/sekcji
- [ ] Linkowanie wewnętrzne do primary
- [ ] Inne: ______________________
"""
                st.code(brief)

        # -------------------------
        # Pairs tab
        # -------------------------
        with tabs[1]:
            st.markdown(f"## Pary powyżej progu: **{threshold_pct}%** ({method})")
            if p is None or p.empty:
                st.info("Brak par powyżej progu. Zmniejsz próg lub wybierz inny preset.")
            else:
                st.dataframe(p, use_container_width=True, height=360)

        # -------------------------
        # Groups tab
        # -------------------------
        with tabs[2]:
            st.markdown(f"## Grupy (klastry) powyżej progu: **{threshold_pct}%** ({method})")
            if g is None or g.empty:
                st.info("Brak grup (min. 2 URL-e w grupie).")
            else:
                st.dataframe(g, use_container_width=True, height=360)

        # -------------------------
        # Articles tab
        # -------------------------
        with tabs[3]:
            st.markdown("## Dane artykułów (Markdown)")
            st.dataframe(a, use_container_width=True, height=360)

        # -------------------------
        # Diagnostics tab
        # -------------------------
        with tabs[4]:
            st.markdown("## Diagnostyka")
            if st.session_state.sim_matrix is not None:
                sim = st.session_state.sim_matrix
                st.markdown("**Histogram podobieństw (wszystkie pary)**")
                _plot_similarity_hist(sim, title=f"Similarity distribution ({method})")

                n = sim.shape[0]
                if n >= 2:
                    vals = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            vals.append(float(sim[i, j]))
                    if vals:
                        st.write(
                            f"- Maks: **{max(vals)*100:.1f}%**  |  "
                            f"Średnia: **{(sum(vals)/len(vals))*100:.1f}%**  |  "
                            f"Liczba par: **{len(vals)}**"
                        )
            else:
                st.info("Brak macierzy podobieństw.")

            st.markdown("**Log działań (ostatnie ~200 linii)**")
            log = st.session_state.run_log[-200:]
            st.code("\n".join(log) if log else "Brak logu.")

        # -------------------------
        # Export tab
        # -------------------------
        with tabs[5]:
            st.markdown("## Eksport CSV (UTF-8, działa w Excelu)")
            if a is not None and not a.empty:
                _download_csv_button(a, "⬇️ articles.csv", "articles.csv")
            if p is not None and not p.empty:
                _download_csv_button(p, "⬇️ similarity_pairs.csv", "similarity_pairs.csv")
            if g is not None and not g.empty:
                _download_csv_button(g, "⬇️ similarity_groups.csv", "similarity_groups.csv")


# -------------------------
# Run pipeline
# -------------------------
if run_btn:
    st.session_state.run_log = []

    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        st.error("Base URL musi zaczynać się od http:// lub https://")
        st.stop()

    status = st.empty()
    bar = st.progress(0.0)
    cb = _progress_callback_factory(status, bar, st.session_state.run_log)

    # 1) scrape
    try:
        if mode == "Auto (sitemap/RSS) – pobierz artykuły ze strony":
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

        elif mode == "Wklej URL-e ręcznie":
            urls = _extract_urls_from_text(manual_urls_text)
            urls = [normalize_url_public(u) for u in urls]
            urls = filter_internal_urls_public(urls, base_url=base_url, same_subdomain_only=bool(same_subdomain_only))
            if not urls:
                st.error("Nie wykryłam żadnych poprawnych URL-i do analizy.")
                st.stop()

            cb(0, 1, f"Start: pobieranie treści dla {len(urls)} URL-i…")
            articles_df = scrape_articles_from_urls(
                base_url=base_url,
                urls=urls,
                max_pages=int(max_pages),
                timeout=int(timeout),
                delay=float(delay),
                same_subdomain_only=bool(same_subdomain_only),
                progress_callback=cb,
            )

        else:
            if uploaded_csv is None:
                st.error("Wgraj plik CSV z URL-ami.")
                st.stop()

            urls = _read_urls_from_uploaded_csv(uploaded_csv)
            urls = [normalize_url_public(u) for u in urls]
            urls = filter_internal_urls_public(urls, base_url=base_url, same_subdomain_only=bool(same_subdomain_only))
            if not urls:
                st.error("W CSV nie znalazłam URL-i pasujących do domeny / filtra.")
                st.stop()

            cb(0, 1, f"Start: pobieranie treści dla {len(urls)} URL-i z CSV…")
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

    if articles_df is None or articles_df.empty:
        st.warning("Nie udało się pobrać żadnych artykułów.")
        st.stop()

    # Fix Polish chars if needed
    articles_df = _fix_dataframe_text(articles_df, ["H1", "title", "treść w Markdown"])

    # 2) similarity
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

    pairs_df = _fix_dataframe_text(pairs_df, ["title_a", "h1_a", "title_b", "h1_b"])
    groups_df = _fix_dataframe_text(groups_df, ["urls"])

    # store in session for right panel
    st.session_state.articles_df = articles_df
    st.session_state.pairs_df = pairs_df
    st.session_state.groups_df = groups_df
    st.session_state.sim_matrix = sim

    bar.progress(1.0)
    status.write("Gotowe ✅ Przejdź do zakładek po prawej stronie (GSC workflow).")
    st.rerun()
