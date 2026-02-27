from __future__ import annotations

import time
import re
import urllib.parse
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Set, List, Dict, Tuple

import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd
from markdownify import markdownify as md

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class ScrapeConfig:
    base_url: str
    max_pages: int = 200
    timeout: int = 20
    delay: float = 0.6
    max_depth: int = 3
    user_agent: str = "Mozilla/5.0 (compatible; SEOContentAuditor/1.0)"
    same_subdomain_only: bool = True


def _normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    parsed = urllib.parse.urlsplit(url)
    # remove fragments
    parsed = parsed._replace(fragment="")
    # normalize path
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    parsed = parsed._replace(path=path)
    return urllib.parse.urlunsplit(parsed)


def _get_domain_parts(url: str) -> Tuple[str, str]:
    """Returns (netloc, registered-ish domain). We approximate the 'root domain' by last 2 labels."""
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


def _safe_get(session: requests.Session, url: str, cfg: ScrapeConfig) -> Optional[requests.Response]:
    try:
        resp = session.get(
            url,
            timeout=cfg.timeout,
            headers={"User-Agent": cfg.user_agent},
            allow_redirects=True,
        )
        # basic content-type gate
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
            return None
        if resp.status_code >= 400:
            return None
        return resp
    except requests.RequestException:
        return None


def _html_to_markdown(html_fragment: str) -> str:
    """
    Convert HTML to Markdown while preserving headings and lists.
    markdownify generally keeps headings/list structure; we set options to be consistent.
    """
    text = md(
        html_fragment,
        heading_style="ATX",
        bullets="-",
        strip=["script", "style", "noscript"],
        convert=["a", "p", "ul", "ol", "li", "strong", "em", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "code", "pre", "br"],
    )
    # cleanup excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _text_len(tag: Tag) -> int:
    return len(" ".join(tag.get_text(" ", strip=True).split()))


def _pick_main_content(soup: BeautifulSoup) -> Optional[Tag]:
    """
    Heuristic content extraction:
    1) <article>
    2) <main>
    3) best candidate among common containers by text length / link ratio
    """
    article = soup.find("article")
    if isinstance(article, Tag) and _text_len(article) > 150:
        return article

    main = soup.find("main")
    if isinstance(main, Tag) and _text_len(main) > 150:
        return main

    candidates: List[Tag] = []
    for sel in ["div", "section"]:
        for t in soup.find_all(sel):
            if not isinstance(t, Tag):
                continue
            # skip obvious nav/footers/sidebars
            classes = " ".join(t.get("class", [])).lower()
            tid = (t.get("id") or "").lower()
            bad = ["nav", "menu", "footer", "header", "sidebar", "cookie", "newsletter", "share", "breadcrumb", "comments", "related"]
            if any(b in classes for b in bad) or any(b in tid for b in bad):
                continue
            txt_len = _text_len(t)
            if txt_len < 250:
                continue
            candidates.append(t)

    if not candidates:
        return None

    def score(t: Tag) -> float:
        text = t.get_text(" ", strip=True)
        text_words = len(text.split())
        links_words = 0
        for a in t.find_all("a"):
            links_words += len(a.get_text(" ", strip=True).split())
        link_ratio = (links_words / max(1, text_words))
        # prefer long text, penalize too many links
        return float(text_words) * (1.0 - min(0.9, link_ratio))

    best = max(candidates, key=score)
    return best


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
    # fallback: og:title
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


def _likely_article_url(url: str) -> bool:
    """
    Heuristic: prefer URLs that look like content pages.
    """
    u = urllib.parse.urlsplit(url)
    path = (u.path or "/").lower()

    # exclude typical non-articles
    bad = [
        "/tag/", "/tags/", "/category/", "/kategoria/", "/author/", "/autor/",
        "/search", "/szukaj", "/page/", "/strona/", "/wp-admin", "/wp-login",
        ".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".pdf", ".zip",
        "/feed", "/xmlrpc.php"
    ]
    if any(b in path for b in bad):
        return False

    # too shallow homepage
    if path in ("/", ""):
        return False

    # common article cues
    good = ["/blog/", "/articles/", "/artykul/", "/artykuly/", "/poradnik/", "/wiedza/", "/news/", "/aktualnosci/"]
    if any(g in path for g in good):
        return True

    # date-like patterns / slug depth
    # e.g. /2024/05/slug
    if re.search(r"/20\d{2}/\d{1,2}/", path):
        return True

    # slug with hyphens and reasonable depth
    depth = len([p for p in path.split("/") if p])
    if depth >= 2 and "-" in path:
        return True

    return depth >= 2


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

        # naive xml parse with BeautifulSoup
        xml = BeautifulSoup(resp.text, "xml")
        locs = [loc.get_text(strip=True) for loc in xml.find_all("loc")]
        locs = [_normalize_url(u) for u in locs if u]
        # if it's an index, pull nested sitemaps quickly (limited)
        if any("sitemap" in u.lower() and u.lower().endswith(".xml") for u in locs[:50]):
            for nested in locs[:50]:
                r2 = _safe_get(session, nested, cfg)
                if not r2 or not r2.text:
                    continue
                x2 = BeautifulSoup(r2.text, "xml")
                nested_locs = [l.get_text(strip=True) for l in x2.find_all("loc")]
                for u in nested_locs:
                    u = _normalize_url(u)
                    if _is_internal(u, cfg.base_url, cfg.same_subdomain_only) and _likely_article_url(u):
                        found.append(u)
        else:
            for u in locs:
                if _is_internal(u, cfg.base_url, cfg.same_subdomain_only) and _likely_article_url(u):
                    found.append(u)

        if found:
            break

    # unique preserve order
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
    ]
    found: List[str] = []
    for feed_url in candidates:
        try:
            resp = session.get(feed_url, timeout=cfg.timeout, headers={"User-Agent": cfg.user_agent})
        except requests.RequestException:
            continue
        if resp.status_code >= 400 or not resp.text:
            continue
        xml = BeautifulSoup(resp.text, "xml")
        # RSS: <item><link> or Atom: <entry><link href="">
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


def _crawl_discover_article_urls(session: requests.Session, cfg: ScrapeConfig, progress_callback: Optional[ProgressCallback]) -> List[str]:
    """
    Fallback: BFS crawl within domain to collect likely article URLs.
    """
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
        if resp is None:
            continue

        soup = BeautifulSoup(resp.text, "lxml")

        # if this page looks like an article, keep it
        if _likely_article_url(url):
            content = _pick_main_content(soup)
            if content is not None and _text_len(content) > 250:
                articles.append(url)
                if len(articles) >= cfg.max_pages:
                    break

        if depth >= cfg.max_depth:
            continue

        links = _collect_links(soup, url, cfg.base_url, cfg.same_subdomain_only)
        # prioritize likely articles
        prioritized = sorted(links, key=lambda x: (not _likely_article_url(x), len(urllib.parse.urlsplit(x).path)))
        for link in prioritized:
            if link not in visited:
                queue.append((link, depth + 1))

        time.sleep(cfg.delay)

    # unique preserve order
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
    """
    Discovery strategy:
    1) sitemap.xml / sitemap_index.xml
    2) RSS/Atom feed
    3) crawl fallback
    """
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
    urls = _crawl_discover_article_urls(session, cfg, progress_callback)
    return urls[: cfg.max_pages]


def scrape_single_article(session: requests.Session, url: str, cfg: ScrapeConfig) -> Optional[Dict[str, str]]:
    resp = _safe_get(session, url, cfg)
    if resp is None or not resp.text:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    content_tag = _pick_main_content(soup)
    if content_tag is None:
        # fallback: whole body
        body = soup.find("body")
        content_tag = body if isinstance(body, Tag) else None

    h1 = _extract_h1(soup, content_tag)
    title = _extract_title(soup)

    # remove common noisy blocks inside content
    if content_tag is not None:
        for bad_sel in [
            "[class*=cookie]", "[id*=cookie]",
            "[class*=newsletter]", "[id*=newsletter]",
            "[class*=share]", "[class*=social]",
            "nav", "footer", "header", "aside",
            "[class*=comment]", "[id*=comment]",
            "[class*=related]", "[id*=related]",
        ]:
            for t in content_tag.select(bad_sel):
                try:
                    t.decompose()
                except Exception:
                    pass

        html_fragment = str(content_tag)
    else:
        html_fragment = resp.text

    markdown_text = _html_to_markdown(html_fragment)

    # minimal quality gate
    if len(markdown_text.split()) < 80:
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
    # drop duplicates by URL
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    return df
