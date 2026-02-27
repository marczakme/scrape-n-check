from __future__ import annotations

import time
import re
import urllib.parse
from dataclasses import dataclass
from typing import Callable, Optional, Set, List, Dict, Tuple

import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
from markdownify import markdownify as md

ProgressCallback = Callable[[int, int, str], None]

# Spójne progi (Twoje życzenie: wszędzie ~50)
MIN_WORDS_CONTENT_CONTAINER = 50   # minimalna liczba słów dla kontenera treści
MIN_WORDS_ARTICLE_MARKDOWN = 50    # minimalna liczba słów w markdown, żeby uznać stronę za "artykuł"


@dataclass
class ScrapeConfig:
    base_url: str
    max_pages: int = 200
    timeout: int = 20
    delay: float = 0.6
    max_depth: int = 3
    user_agent: str = "Mozilla/5.0 (compatible; SEOContentAuditor/1.0)"
    same_subdomain_only: bool = True


# -------------------------
# URL utilities
# -------------------------
def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    parsed = urllib.parse.urlsplit(url)
    parsed = parsed._replace(fragment="")  # remove fragments

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    parsed = parsed._replace(path=path)
    return urllib.parse.urlunsplit(parsed)


def _get_domain_parts(url: str) -> Tuple[str, str]:
    """
    Returns (netloc, "root domain" approximation).
    Root domain is approximated as last two labels.
    """
    netloc = urllib.parse.urlsplit(url).netloc.lower()
    parts = [p for p in netloc.split(".") if p]
    root = ".".join(parts[-2:]) if len(parts) >= 2 else netloc
    return netloc, root


def _is_internal(url: str, base_url: str, same_subdomain_only: bool) -> bool:
    u = urllib.parse.urlsplit(url)
    if not u.scheme.startswith("http"):
        return False

    base_netloc, base_root = _get_domain_parts(base_url)
    netloc, root = _get_domain_parts(url)

    if same_subdomain_only:
        return netloc == base_netloc
    return root == base_root


def _likely_article_url(url: str) -> bool:
    """
    Heuristic: prefer URLs that look like content pages.
    Keep it general + language agnostic.
    """
    u = urllib.parse.urlsplit(url)
    path = (u.path or "/").lower()

    if path in ("/", ""):
        return False

    # Exclude typical non-article endpoints
    bad = [
        "/tag/", "/tags/", "/category/", "/kategoria/", "/author/", "/autor/",
        "/search", "/szukaj", "/page/", "/strona/",
        "/wp-admin", "/wp-login",
        "/feed", "/rss", "/atom",
        "/xmlrpc.php",
        ".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".pdf", ".zip",
    ]
    if any(b in path for b in bad):
        return False

    # Positive cues
    good = ["/blog/", "/articles/", "/artykul/", "/artykuly/", "/poradnik/", "/wiedza/", "/news/", "/aktualnosci/"]
    if any(g in path for g in good):
        return True

    # Date-like patterns: /2024/05/slug
    if re.search(r"/20\d{2}/\d{1,2}/", path):
        return True

    # Slug-ish URL: deeper path and contains hyphen
    depth = len([p for p in path.split("/") if p])
    if depth >= 2 and "-" in path:
        return True

    return depth >= 2


# -------------------------
# Networking
# -------------------------
def _safe_get(session: requests.Session, url: str, cfg: ScrapeConfig) -> Optional[requests.Response]:
    try:
        resp = session.get(
            url,
            timeout=cfg.timeout,
            headers={"User-Agent": cfg.user_agent},
            allow_redirects=True,
        )
        if resp.status_code >= 400:
            return None
        return resp
    except requests.RequestException:
        return None


# -------------------------
# HTML -> Markdown
# -------------------------
def _html_to_markdown(html_fragment: str) -> str:
    """
    Convert HTML to Markdown while preserving headings and lists.
    IMPORTANT: markdownify cannot accept both 'strip' and 'convert' in newer versions.
    We keep it simple and rely on prior BeautifulSoup cleanup + markdownify defaults.
    """
    text = md(
        html_fragment,
        heading_style="ATX",
        bullets="-",
    )
    text = re.sub(r"\n{3,}", "\n\n", (text or "")).strip()
    return text


def _text_len(tag: Tag) -> int:
    return len(" ".join(tag.get_text(" ", strip=True).split()))


def _pick_main_content(soup: BeautifulSoup) -> Optional[Tag]:
    """
    Robust content extraction with low, consistent thresholds:
    1) <article>
    2) <main>
    3) semantic hints (itemprop=articleBody, role=main, class/id includes content/post/entry/article)
    4) best <div>/<section> by text length and link ratio
    """
    # 1) <article>
    article = soup.find("article")
    if isinstance(article, Tag) and _text_len(article) >= MIN_WORDS_CONTENT_CONTAINER:
        return article

    # 2) <main>
    main = soup.find("main")
    if isinstance(main, Tag) and _text_len(main) >= MIN_WORDS_CONTENT_CONTAINER:
        return main

    # 3) semantic hints (still generic)
    semantic_candidates: List[Tag] = []
    semantic_selectors = [
        '[itemprop="articleBody"]',
        '[role="main"]',
        '[class*="content"]',
        '[id*="content"]',
        '[class*="entry"]',
        '[id*="entry"]',
        '[class*="post"]',
        '[id*="post"]',
        '[class*="article"]',
        '[id*="article"]',
    ]
    for sel in semantic_selectors:
        for t in soup.select(sel):
            if isinstance(t, Tag) and _text_len(t) >= MIN_WORDS_CONTENT_CONTAINER:
                semantic_candidates.append(t)

    if semantic_candidates:
        return max(semantic_candidates, key=_text_len)

    # 4) generic candidates: div/section
    candidates: List[Tag] = []
    for sel in ["div", "section"]:
        for t in soup.find_all(sel):
            if not isinstance(t, Tag):
                continue

            classes = " ".join(t.get("class", [])).lower()
            tid = (t.get("id") or "").lower()

            bad = ["nav", "menu", "footer", "header", "sidebar", "cookie", "newsletter", "share", "breadcrumb", "comments", "related"]
            if any(b in classes for b in bad) or any(b in tid for b in bad):
                continue

            txt_len = _text_len(t)
            if txt_len < MIN_WORDS_CONTENT_CONTAINER:
                continue
            candidates.append(t)

    if not candidates:
        return None

    def score(t: Tag) -> float:
        text_words = len(t.get_text(" ", strip=True).split())
        links_words = 0
        for a in t.find_all("a"):
            links_words += len(a.get_text(" ", strip=True).split())
        link_ratio = links_words / max(1, text_words)
        # Prefer long text, penalize too many links
        return float(text_words) * (1.0 - min(0.9, link_ratio))

    return max(candidates, key=score)


def _extract_h1(soup: BeautifulSoup, content: Optional[Tag]) -> str:
    if content is not None:
        h1 = content.find("h1")
        if isinstance(h1, Tag):
            return h1.get_text(" ", strip=True)
    h1 = soup.find("h1")
    if isinstance(h1, Tag):
        return h1.get_text(" ", strip=True)
    return ""


def _extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return str(og.get("content")).strip()
    return ""


def _collect_links(soup: BeautifulSoup, page_url: str, base_url: str, same_subdomain_only: bool) -> Set[str]:
    links: Set[str] = set()
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        href = href.strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        abs_url = urllib.parse.urljoin(page_url, href)
        abs_url = _normalize_url(abs_url)
        if not _is_internal(abs_url, base_url, same_subdomain_only):
            continue
        links.add(abs_url)
    return links


# -------------------------
# URL discovery
# -------------------------
def _discover_urls_via_sitemap(session: requests.Session, cfg: ScrapeConfig) -> List[str]:
    base = cfg.base_url.rstrip("/")
    candidates = [
        f"{base}/sitemap.xml",
        f"{base}/sitemap_index.xml",
    ]
    found: List[str] = []

    for sm_url in candidates:
        resp = _safe_get(session, sm_url, cfg)
        if not resp or not resp.text:
            continue

        xml = BeautifulSoup(resp.text, "xml")
        locs = [loc.get_text(strip=True) for loc in xml.find_all("loc")]
        locs = [_normalize_url(u) for u in locs if u]

        nested = [u for u in locs if ("sitemap" in u.lower() and u.lower().endswith(".xml"))]
        if nested:
            for nested_sm in nested[:80]:
                r2 = _safe_get(session, nested_sm, cfg)
                if not r2 or not r2.text:
                    continue
                x2 = BeautifulSoup(r2.text, "xml")
                nested_locs = [l.get_text(strip=True) for l in x2.find_all("loc")]
                for u in nested_locs:
                    u = _normalize_url(u)
                    if _is_internal(u, cfg.base_url, cfg.same_subdomain_only) and _likely_article_url(u):
                        found.append(u)
                if len(found) >= cfg.max_pages:
                    break
        else:
            for u in locs:
                if _is_internal(u, cfg.base_url, cfg.same_subdomain_only) and _likely_article_url(u):
                    found.append(u)

        if found:
            break

    seen = set()
    out = []
    for u in found:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _discover_urls_via_rss(session: requests.Session, cfg: ScrapeConfig) -> List[str]:
    base = cfg.base_url.rstrip("/")
    candidates = [
        f"{base}/feed",
        f"{base}/rss",
        f"{base}/feed.xml",
        f"{base}/rss.xml",
        f"{base}/atom.xml",
    ]
    found: List[str] = []
    for feed_url in candidates:
        resp = _safe_get(session, feed_url, cfg)
        if not resp or resp.status_code >= 400 or not resp.text:
            continue

        xml = BeautifulSoup(resp.text, "xml")
        for item in xml.find_all(["item", "entry"]):
            link = item.find("link")
            if link is None:
                continue
            if link.get("href"):
                u = link.get("href")
            else:
                u = link.get_text(strip=True)

            u = _normalize_url(str(u))
            if _is_internal(u, cfg.base_url, cfg.same_subdomain_only) and _likely_article_url(u):
                found.append(u)

        if found:
            break

    seen = set()
    out = []
    for u in found:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _crawl_discover_article_urls(
    session: requests.Session,
    cfg: ScrapeConfig,
    progress_callback: Optional[ProgressCallback],
) -> List[str]:
    start = _normalize_url(cfg.base_url)
    queue: List[Tuple[str, int]] = [(start, 0)]
    visited: Set[str] = set()
    articles: List[str] = []

    while queue and len(articles) < cfg.max_pages:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if progress_callback:
            progress_callback(len(articles), cfg.max_pages, f"Crawl: {url} (depth={depth})")

        resp = _safe_get(session, url, cfg)
        if resp is None or not getattr(resp, "text", None):
            continue

        soup = BeautifulSoup(resp.text, "lxml")

        if _likely_article_url(url):
            content = _pick_main_content(soup)
            if content is not None and _text_len(content) >= MIN_WORDS_CONTENT_CONTAINER:
                articles.append(url)
                if len(articles) >= cfg.max_pages:
                    break

        if depth >= cfg.max_depth:
            continue

        links = _collect_links(soup, url, cfg.base_url, cfg.same_subdomain_only)
        prioritized = sorted(
            links,
            key=lambda x: (not _likely_article_url(x), len(urllib.parse.urlsplit(x).path)),
        )

        for link in prioritized:
            if link not in visited:
                queue.append((link, depth + 1))

        time.sleep(cfg.delay)

    seen = set()
    out = []
    for u in articles:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def discover_article_urls(
    session: requests.Session,
    cfg: ScrapeConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[str]:
    if progress_callback:
        progress_callback(0, cfg.max_pages, "Próbuję wykryć URL-e przez sitemap…")
    urls = _discover_urls_via_sitemap(session, cfg)
    if urls:
        return urls[: cfg.max_pages]

    if progress_callback:
        progress_callback(0, cfg.max_pages, "Brak sitemap lub pusta. Próbuję RSS/Atom…")
    urls = _discover_urls_via_rss(session, cfg)
    if urls:
        return urls[: cfg.max_pages]

    if progress_callback:
        progress_callback(0, cfg.max_pages, "Brak RSS. Przechodzę do crawl (fallback)…")
    return _crawl_discover_article_urls(session, cfg, progress_callback)[: cfg.max_pages]


# -------------------------
# Article scraping
# -------------------------
def scrape_single_article(session: requests.Session, url: str, cfg: ScrapeConfig) -> Optional[Dict[str, str]]:
    resp = _safe_get(session, url, cfg)
    if resp is None or not getattr(resp, "text", None):
        return None

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    content_tag = _pick_main_content(soup)
    if content_tag is None:
        body = soup.find("body")
        content_tag = body if isinstance(body, Tag) else None

    h1 = _extract_h1(soup, content_tag)
    title = _extract_title(soup)

    if content_tag is not None:
        # Remove obvious junk inside selected container
        for bad_sel in [
            "script", "style", "noscript",
            "nav", "footer", "header", "aside",
            "[class*=cookie]", "[id*=cookie]",
            "[class*=newsletter]", "[id*=newsletter]",
            "[class*=share]", "[class*=social]",
            "[class*=comment]", "[id*=comment]",
            "[class*=related]", "[id*=related]",
            "[class*=breadcrumb]", "[id*=breadcrumb]",
        ]:
            try:
                for t in content_tag.select(bad_sel):
                    t.decompose()
            except Exception:
                pass

        html_fragment = str(content_tag)
    else:
        html_fragment = resp.text

    markdown_text = _html_to_markdown(html_fragment)

    # QUALITY GATE: spójnie 50 słów
    if len(markdown_text.split()) < MIN_WORDS_ARTICLE_MARKDOWN:
        return None

    return {
        "URL": _normalize_url(resp.url),
        "H1": h1 or "",
        "title": title or "",
        "treść w Markdown": markdown_text or "",
    }


def scrape_site_articles(
    base_url: str,
    max_pages: int = 200,
    timeout: int = 20,
    delay: float = 0.6,
    max_depth: int = 3,
    user_agent: str = "Mozilla/5.0 (compatible; SEOContentAuditor/1.0)",
    same_subdomain_only: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    cfg = ScrapeConfig(
        base_url=base_url,
        max_pages=max_pages,
        timeout=timeout,
        delay=delay,
        max_depth=max_depth,
        user_agent=user_agent,
        same_subdomain_only=same_subdomain_only,
    )

    session = requests.Session()
    urls = discover_article_urls(session, cfg, progress_callback=progress_callback)
    if not urls:
        return pd.DataFrame(columns=["URL", "H1", "title", "treść w Markdown"])

    records: List[Dict[str, str]] = []
    total = min(len(urls), cfg.max_pages)

    for i, url in enumerate(urls[:total], start=1):
        if progress_callback:
            progress_callback(i - 1, total, f"Pobieram ({i}/{total}): {url}")

        rec = scrape_single_article(session, url, cfg)
        if rec:
            records.append(rec)

        time.sleep(cfg.delay)

    df = pd.DataFrame(records, columns=["URL", "H1", "title", "treść w Markdown"])
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    return df


def scrape_articles_from_urls(
    base_url: str,
    urls: list[str],
    max_pages: int = 200,
    timeout: int = 20,
    delay: float = 0.6,
    user_agent: str = "Mozilla/5.0 (compatible; SEOContentAuditor/1.0)",
    same_subdomain_only: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    cfg = ScrapeConfig(
        base_url=base_url,
        max_pages=max_pages,
        timeout=timeout,
        delay=delay,
        max_depth=1,
        user_agent=user_agent,
        same_subdomain_only=same_subdomain_only,
    )

    session = requests.Session()

    clean_urls = [u for u in urls if u and isinstance(u, str)]
    clean_urls = [_normalize_url(u) for u in clean_urls]
    clean_urls = [u for u in clean_urls if _is_internal(u, cfg.base_url, cfg.same_subdomain_only)]

    seen = set()
    final_urls = []
    for u in clean_urls:
        if u not in seen:
            seen.add(u)
            final_urls.append(u)
        if len(final_urls) >= cfg.max_pages:
            break

    if not final_urls:
        return pd.DataFrame(columns=["URL", "H1", "title", "treść w Markdown"])

    records: List[Dict[str, str]] = []
    total = len(final_urls)

    for i, url in enumerate(final_urls, start=1):
        if progress_callback:
            progress_callback(i - 1, total, f"Pobieram ({i}/{total}): {url}")

        rec = scrape_single_article(session, url, cfg)
        if rec:
            records.append(rec)

        time.sleep(cfg.delay)

    df = pd.DataFrame(records, columns=["URL", "H1", "title", "treść w Markdown"])
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    return df


# -------------------------
# Public wrappers (for app.py)
# -------------------------
def normalize_url_public(url: str) -> str:
    return _normalize_url(url)


def likely_article_url_public(url: str) -> bool:
    return _likely_article_url(url)


def filter_internal_urls_public(urls: list[str], base_url: str, same_subdomain_only: bool) -> list[str]:
    out = []
    for u in urls:
        if _is_internal(u, base_url, same_subdomain_only):
            out.append(u)

    seen = set()
    final = []
    for u in out:
        if u not in seen:
            seen.add(u)
            final.append(u)
    return final
