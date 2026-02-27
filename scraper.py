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

try:
    import trafilatura
except Exception:
    trafilatura = None

ProgressCallback = Callable[[int, int, str], None]

MIN_WORDS_ARTICLE_MARKDOWN = 120  # realny artykuł zwykle ma dużo więcej; listingi łatwiej odsiać
MAX_REDIRECTS = 3


@dataclass
class ScrapeConfig:
    base_url: str
    max_pages: int = 200
    timeout: int = 20
    delay: float = 0.6
    max_depth: int = 3
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    same_subdomain_only: bool = True


# -------------------------
# URL utilities
# -------------------------
def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    parsed = urllib.parse.urlsplit(url)
    parsed = parsed._replace(fragment="")
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    parsed = parsed._replace(path=path)
    return urllib.parse.urlunsplit(parsed)


def _get_domain_parts(url: str) -> Tuple[str, str]:
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
    u = urllib.parse.urlsplit(url)
    path = (u.path or "/").lower()

    if path in ("/", ""):
        return False

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

    good = ["/blog/", "/articles/", "/artykul/", "/artykuly/", "/poradnik/", "/wiedza/", "/news/", "/aktualnosci/"]
    if any(g in path for g in good):
        return True

    if re.search(r"/20\d{2}/\d{1,2}/", path):
        return True

    depth = len([p for p in path.split("/") if p])
    if depth >= 2 and "-" in path:
        return True

    return depth >= 2


def _is_listing_like(url: str) -> bool:
    """
    Detect typical list pages to treat redirects as "not an article".
    """
    path = (urllib.parse.urlsplit(url).path or "/").lower()
    if path.rstrip("/") in ("/blog", "/blog/"):
        return True
    if any(x in path for x in ["/tag/", "/category/", "/kategoria/", "/page/", "/strona/"]):
        return True
    return False


# -------------------------
# Networking (redirect-aware)
# -------------------------
def _browser_headers(cfg: ScrapeConfig, referer: Optional[str] = None) -> Dict[str, str]:
    h = {
        "User-Agent": cfg.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }
    if referer:
        h["Referer"] = referer
    return h


def _safe_get_no_redirect(session: requests.Session, url: str, cfg: ScrapeConfig, referer: Optional[str] = None) -> Optional[requests.Response]:
    try:
        resp = session.get(
            url,
            timeout=cfg.timeout,
            headers=_browser_headers(cfg, referer=referer),
            allow_redirects=False,  # kluczowe
        )
        return resp
    except requests.RequestException:
        return None


def _follow_redirects_manually(
    session: requests.Session,
    url: str,
    cfg: ScrapeConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> Optional[requests.Response]:
    """
    Fetch URL without auto-redirects; follow up to MAX_REDIRECTS only within the same site.
    If redirect leads to listing-like URL (e.g., /blog/), treat as soft-block and return that response
    (caller may decide to skip).
    """
    current = _normalize_url(url)
    referer = cfg.base_url

    for hop in range(MAX_REDIRECTS + 1):
        resp = _safe_get_no_redirect(session, current, cfg, referer=referer)
        if resp is None:
            return None

        # 2xx OK
        if 200 <= resp.status_code < 300:
            # Attach a helpful attribute: final_url
            resp.final_url = current  # type: ignore[attr-defined]
            return resp

        # redirects
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location") or ""
            nxt = _normalize_url(urllib.parse.urljoin(current, loc))

            if progress_callback:
                progress_callback(0, 0, f"Redirect {resp.status_code}: {current} → {nxt}")

            # block external jumps
            if not _is_internal(nxt, cfg.base_url, cfg.same_subdomain_only):
                resp.final_url = current  # type: ignore[attr-defined]
                return resp

            # if redirected to listing-like page, likely bot/consent soft redirect
            if _is_listing_like(nxt):
                # Return response but mark final_url as the listing target
                resp.final_url = nxt  # type: ignore[attr-defined]
                resp.soft_redirect_to_listing = True  # type: ignore[attr-defined]
                return resp

            referer = current
            current = nxt
            continue

        # other codes (403/503 etc.)
        resp.final_url = current  # type: ignore[attr-defined]
        return resp

    return None


# -------------------------
# Extraction helpers
# -------------------------
def _words(s: str) -> int:
    return len((s or "").split())


def _extract_h1(soup: BeautifulSoup) -> str:
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


def _html_to_markdown(html_fragment: str) -> str:
    text = md(html_fragment, heading_style="ATX", bullets="-")
    text = re.sub(r"\n{3,}", "\n\n", (text or "")).strip()
    return text


def _extract_markdown_trafilatura(html: str, url: str) -> str:
    if trafilatura is None:
        return ""
    try:
        md_text = trafilatura.extract(
            html,
            url=url,
            output_format="markdown",
            include_formatting=True,
            include_links=False,
            include_images=False,
            favor_precision=False,
        )
        return (md_text or "").strip()
    except Exception:
        return ""


def _remove_boilerplate(body: Tag) -> None:
    for tname in ["script", "style", "noscript", "svg"]:
        for t in body.find_all(tname):
            try:
                t.decompose()
            except Exception:
                pass
    for tname in ["nav", "footer", "header", "aside", "form"]:
        for t in body.find_all(tname):
            try:
                t.decompose()
            except Exception:
                pass

    bad_patterns = [
        "cookie", "consent", "gdpr",
        "newsletter", "subscribe",
        "share", "social",
        "breadcrumb",
        "comment", "comments",
        "related",
        "modal", "popup",
        "banner", "ads", "advert",
        "pagination", "pager",
        "sidebar",
    ]
    for t in body.find_all(True):
        cls = " ".join(t.get("class", [])).lower()
        tid = (t.get("id") or "").lower()
        if any(p in cls for p in bad_patterns) or any(p in tid for p in bad_patterns):
            try:
                t.decompose()
            except Exception:
                pass


def _select_best_block(body: Tag) -> Tag:
    candidates: List[Tag] = []
    for sel in ["article", "main", "section", "div"]:
        for t in body.find_all(sel):
            if not isinstance(t, Tag):
                continue
            tw = _words(t.get_text(" ", strip=True))
            if tw >= 120:
                candidates.append(t)
    if not candidates:
        return body

    def link_words(t: Tag) -> int:
        lw = 0
        for a in t.find_all("a"):
            lw += _words(a.get_text(" ", strip=True))
        return lw

    def score(t: Tag) -> float:
        tw = _words(t.get_text(" ", strip=True))
        lw = link_words(t)
        density = tw / (lw + 1.0)
        return density * (1.0 + min(2.0, tw / 800.0))

    return max(candidates, key=score)


# -------------------------
# URL discovery
# -------------------------
def _discover_urls_via_sitemap(session: requests.Session, cfg: ScrapeConfig) -> List[str]:
    base = cfg.base_url.rstrip("/")
    candidates = [f"{base}/sitemap.xml", f"{base}/sitemap_index.xml"]
    found: List[str] = []

    for sm_url in candidates:
        resp = _follow_redirects_manually(session, sm_url, cfg)
        if not resp or not getattr(resp, "text", None):
            continue
        if resp.status_code >= 400:
            continue

        xml = BeautifulSoup(resp.text, "xml")
        locs = [loc.get_text(strip=True) for loc in xml.find_all("loc")]
        locs = [_normalize_url(u) for u in locs if u]

        nested = [u for u in locs if ("sitemap" in u.lower() and u.lower().endswith(".xml"))]
        if nested:
            for nested_sm in nested[:200]:
                r2 = _follow_redirects_manually(session, nested_sm, cfg)
                if not r2 or not getattr(r2, "text", None) or r2.status_code >= 400:
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


def discover_article_urls(
    session: requests.Session,
    cfg: ScrapeConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> List[str]:
    if progress_callback:
        progress_callback(0, cfg.max_pages, "Próbuję wykryć URL-e przez sitemap…")
    urls = _discover_urls_via_sitemap(session, cfg)
    return urls[: cfg.max_pages]


# -------------------------
# Article scraping
# -------------------------
def scrape_single_article(
    session: requests.Session,
    url: str,
    cfg: ScrapeConfig,
    progress_callback: Optional[ProgressCallback] = None,
) -> Optional[Dict[str, str]]:
    input_url = _normalize_url(url)

    resp = _follow_redirects_manually(session, input_url, cfg, progress_callback=progress_callback)
    if resp is None:
        if progress_callback:
            progress_callback(0, 0, f"FAIL: {input_url} (no response)")
        return None

    # Soft redirect to listing is the #1 suspect in Twoich screenach:
    if getattr(resp, "soft_redirect_to_listing", False):
        if progress_callback:
            progress_callback(0, 0, f"SKIP (soft-redirect→listing): {input_url} → {getattr(resp, 'final_url', '')}")
        return None

    if resp.status_code >= 400:
        if progress_callback:
            progress_callback(0, 0, f"SKIP (HTTP {resp.status_code}): {input_url}")
        return None

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
        if progress_callback:
            progress_callback(0, 0, f"SKIP (non-HTML {ctype}): {input_url}")
        return None

    html = resp.text
    soup = BeautifulSoup(html, "lxml")

    h1 = _extract_h1(soup)
    title = _extract_title(soup)

    # 1) Trafiatura (jeśli jest)
    markdown_text = _extract_markdown_trafilatura(html, url=input_url)

    # 2) Fallback
    if _words(markdown_text) < MIN_WORDS_ARTICLE_MARKDOWN:
        body = soup.find("body")
        if not isinstance(body, Tag):
            return None
        _remove_boilerplate(body)
        best = _select_best_block(body)
        markdown_text = _html_to_markdown(str(best))

    wc = _words(markdown_text)
    if wc < MIN_WORDS_ARTICLE_MARKDOWN:
        if progress_callback:
            progress_callback(0, 0, f"SKIP (thin content {wc}w): {input_url}")
        return None

    # KLUCZ: zapisujemy URL wejściowy, nie końcowy (żeby nie „skleiło” wszystkiego w /blog/)
    return {
        "URL": input_url,
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

    for i, u in enumerate(urls[:total], start=1):
        if progress_callback:
            progress_callback(i - 1, total, f"Pobieram ({i}/{total}): {u}")

        rec = scrape_single_article(session, u, cfg, progress_callback=progress_callback)
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

    records: List[Dict[str, str]] = []
    total = len(final_urls)

    for i, u in enumerate(final_urls, start=1):
        if progress_callback:
            progress_callback(i - 1, total, f"Pobieram ({i}/{total}): {u}")

        rec = scrape_single_article(session, u, cfg, progress_callback=progress_callback)
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
