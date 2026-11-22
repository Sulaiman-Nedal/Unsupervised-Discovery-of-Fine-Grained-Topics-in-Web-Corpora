"""url_filters.py — URL-pattern boilerplate/template-page detection.

DLR's site uses conventional CMS URL structure: legal/nav pages live at
predictable paths (/impressum, /datenschutz, /sitemap), and Apache directory
listings carry a recognizable sort-order query string (?C=D;O=A). Matching
on url_path/url_query catches these template pages deterministically and
cheaply, without needing the (already-discarded) raw HTML.
"""
import re

_BOILERPLATE_PATH_PATTERNS = [
    r'/impressum', r'/imprint', r'/datenschutz', r'/privacy',
    r'/sitemap', r'/agb', r'/terms', r'/cookie', r'/cookies',
    r'/search', r'/suche', r'/login', r'/logout', r'/anmeldung',
    # Dedicated German-gov accessibility-statement pages (legally mandated
    # boilerplate; formed their own spurious topic — barrierefreiheit/bitv/bgg).
    r'/barrierefreiheit', r'/accessibility',
]
_BOILERPLATE_RE = re.compile('|'.join(_BOILERPLATE_PATH_PATTERNS), re.IGNORECASE)

# Apache mod_autoindex directory-listing sort-order query params: C=D;O=A etc.
# (url_query is already isolated from the leading "?", so no separator is required)
_DIRLISTING_QUERY_RE = re.compile(r'C=[NMSD];O=[AD]', re.IGNORECASE)

# Auto-generated API/source/coverage documentation (Doxygen, Javadoc, lcov)
# and the SUMO build server's nightly artifacts. These pages are raw source
# code, coverage tables, or build/test reports, not research content — they
# form their own spurious topics (const/void/int, lcov/hit, texttest/busstop).
# `/daily/` is the SUMO build-server path and is exclusive to it in this corpus
# (38,417 docs, zero non-SUMO matches), so it cleanly subsumes its doxygen,
# report, and test_default_*.html artifacts in one pattern.
_CODEDOC_RE = re.compile(
    r'/daily/|/doxygen/|/javadoc/|/lcov/|_source\.html|_8h|_8cpp|_8c\b|class_.*\.html',
    re.IGNORECASE,
)

# Auto-generated per-record database/catalog pages: one near-identical templated
# page per database entry (same field layout, only the values change). Like the
# SUMO build reports, these form their own spurious "topics" (the duplicate-topic
# probe in the Stage 2 analysis traced a cluster of them) and, being templated,
# carry no distinct research content. The DLR instance is EARN (European Asteroid
# Research Node): earn.dlr.de/nea/<designation>.htm — one sheet per near-Earth
# asteroid (orbital elements / albedo / discovery circumstances). 35,583 docs,
# ~18% of the German corpus, and `/nea/` is exclusive to earn.dlr.de here
# (zero non-EARN matches), so the path alone is a safe, deterministic signature.
_CATALOG_RE = re.compile(r'/nea/', re.IGNORECASE)


def is_codedoc_url(url_path: str) -> bool:
    """Return True if the URL is auto-generated source/API/coverage documentation
    or a templated per-record database/catalog page (e.g. EARN asteroid sheets)."""
    return bool(_CODEDOC_RE.search(url_path or "") or _CATALOG_RE.search(url_path or ""))


def is_boilerplate_url(url_path: str, url_query: str = "") -> bool:
    """Return True if the URL is a template/nav/legal page, directory listing,
    or auto-generated code documentation — i.e. not research content."""
    if _BOILERPLATE_RE.search(url_path or ""):
        return True
    if _DIRLISTING_QUERY_RE.search(url_query or ""):
        return True
    if is_codedoc_url(url_path):
        return True
    return False
