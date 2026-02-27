# SEO Content Similarity & Cannibalization Audit (Streamlit)

Aplikacja w Python/Streamlit do automatycznego:
- pobierania treści artykułów ze wskazanej strony,
- zapisu do CSV (URL, H1, title, treść w Markdown),
- analizy podobieństwa treści i wykrywania potencjalnej kanibalizacji SEO.

Projekt jest modularny:
- `scraper.py` – pobieranie treści i konwersja HTML → Markdown
- `analyzer.py` – liczenie podobieństw, raport par i grup
- `app.py` – interfejs Streamlit

---

## Funkcje

### 1) Pozyskanie URL-i do analizy (3 tryby)
1. **Auto (sitemap/RSS)** – wykrywa URL-e artykułów i pobiera treść
2. **Wklej URL-e ręcznie** – 1 URL na linię
3. **Wgraj CSV z URL-ami** – CSV z kolumną `URL` lub URL w pierwszej kolumnie

### 2) Eksport danych do CSV (dokładnie 4 kolumny)
CSV zawiera:
- `URL`
- `H1`
- `title`
- `treść w Markdown`

### 3) Analiza podobieństwa i kanibalizacji
Obsługiwane metody:
- `word_tfidf` – podobieństwo **tematyczne** (czy artykuły są o tym samym)
- `char_tfidf` – podobieństwo **tekstowo-szablonowe** (czy mają dużo identycznych fragmentów)
- `hybrid` – połączenie obu (zalecane do kanibalizacji SEO)

Dodatkowo:
- automatyczne usuwanie boilerplate (linie/fragmenty powtarzalne w wielu dokumentach)
- raport par podobnych artykułów
- raport grup (klastrów) potencjalnej kanibalizacji

---

## Instalacja

### 1) Klon repo
```bash
git clone <twoje-repo>
cd <twoje-repo>
