# SEO Content Cannibalization Auditor (Streamlit)

Aplikacja w Streamlit do automatycznego audytu treści pod kątem potencjalnej kanibalizacji SEO:
- wykrywa URL-e artykułów na podanej stronie (sitemap → RSS → crawl),
- pobiera każdy artykuł i zapisuje dane do CSV,
- konwertuje treść HTML do Markdown z zachowaniem nagłówków (H1–H6) oraz list,
- liczy podobieństwo tekstowe i wskazuje pary artykułów o wysokim podobieństwie.

## Output CSV (dokładne kolumny)
Plik CSV z artykułami zawiera **dokładnie**:
- `URL`
- `H1`
- `title`
- `treść w Markdown`

## Instalacja

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
### Przykład użycia
1. Uruchom aplikację:
   streamlit run app.py
2. W polu „Adres strony” wpisz np.:
   https://marczak.me
3. Kliknij „Start” i pobierz CSV oraz raport podobieństw.
