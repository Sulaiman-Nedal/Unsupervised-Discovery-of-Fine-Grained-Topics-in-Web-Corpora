"""text_cleaning.py — shared text cleaning for the DLR pipeline.

Removes PDF extraction artifacts, file-listing noise, and other
non-linguistic tokens that pollute topic keywords and lbl2vec vocabulary.
"""
import re

# PDF internal structure markers and font artifacts
_PDF_KEYWORDS = frozenset({
    "obj", "endobj", "stream", "endstream", "xref", "startxref",
    "trailer", "null", "true", "false",
    # PDF operators
    "bt", "et", "tf", "td", "tm", "tj", "cm", "re", "rg",
    "scn", "gs", "do", "bmc", "emc", "bdc",
    # PDF font/descriptor artifacts
    "fontdescriptor", "fontname", "fontbbox", "fontfile",
    "basefont", "encoding", "widths", "firstchar", "lastchar",
    "arial", "helvetica", "timesnewroman", "courier", "calibri",
    "subtype", "type1", "truetype", "cidfont",
    # PDF font-descriptor field names + stream filters (seen in extracted .pdf text)
    "flatedecode", "missingwidth", "avgwidth", "maxwidth", "minwidth",
    "stemh", "stemv", "italicangle", "capheight", "ascent", "descent",
    "xheight", "leading", "acrobat", "eof", "flags", "stembv",
    # PDF producer/writer markers (unambiguous — never a real content word).
    # NB: deliberately NOT adding catalog/kids/font/pages here — those are PDF
    # dict keys too, but each is also a legitimate content word (e.g. "Earth
    # Observation Catalog"), so blocklisting them would delete real content.
    "pdfwriter",
    # File extension noise
    "txt", "pdf", "png", "jpg", "jpeg", "gif", "svg", "html", "htm",
    "php", "aspx", "css", "xml", "json",
})

# Apache directory listing artifacts
_DIRLIST_TOKENS = frozenset({
    "parentdir", "parentdirparent", "iconamelast", "iconname",
    "lastmodified", "icondescription", "img", "dir",
})

# Concatenated PDF-extraction / site-boilerplate junk that arrives as a single
# token (no internal whitespace to split on), so the per-word filters miss it.
_EXTRA_NOISE = frozenset({
    "objstream", "applicationpdf", "pdf14", "objprocset",
    "objprocsetpdftextextgstate", "dlrdeunspecified",
    "datenschutzerklrung", "datenschutzerklaerung", "hn0",
    # PDF encoding + GIS/image metadata tokens (never real content words)
    "winansiencoding", "worldfile",
    # Monitoring/data-catalog page artifacts (nightly build/cycle reports)
    "nightly", "cyclelength",
    # Glued site-boilerplate captured as one token (no whitespace to split on)
    "articleby", "empowerax", "tagnavigation",
    "homesitemapkontaktbarrierefreiheit",
})

# Generic lazy-load placeholder text injected by JS widgets (embedded
# video/image players, infinite-scroll boxes) that the scraper captured
# before the widget finished rendering. Site-agnostic — not a DLR artifact,
# seen e.g. repeated 3x per affected article in the GDELT Ahrtal corpus.
_WIDGET_NOISE = frozenset({"loading"})

# Compiled patterns
_LONG_HEX    = re.compile(r'^[0-9a-f]{6,}$', re.IGNORECASE)   # hex strings
_PURE_INT    = re.compile(r'^\d+$')                             # standalone integers
_FILESIZE    = re.compile(r'^\d+[kmgKMG]$')                     # Apache filesize tokens (18k, 5M)
_TIMESTAMP   = re.compile(r'^\d{1,2}h\d{2}$')                   # time-of-day tokens (0h20, 3h09, 4h29)
_SHORT_WORD  = re.compile(r'^.{1,1}$')                          # single chars
_PDF_ESCAPE  = re.compile(r'\\[0-9]{3}|\\[nrtf\\]')            # PDF escape sequences
_PUNCT_STRIP = re.compile(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$')    # leading/trailing punctuation (e.g. "/FontDescriptor")
_FONT_COMPOUND = re.compile(r'^(arial|helvetica|times|courier|calibri|verdana)[a-z]*$')  # arialbold, arialmt, ...
_VOWEL       = re.compile(r'[aeiouyäöü]')                       # for label-word quality check


def is_noise_token(token: str) -> bool:
    """Return True if the token is likely a PDF artifact or file-listing noise."""
    t = _PUNCT_STRIP.sub('', token.lower().strip())
    if not t:
        return True
    if t in _PDF_KEYWORDS:
        return True
    if t in _DIRLIST_TOKENS:
        return True
    if t in _EXTRA_NOISE:
        return True
    if t in _WIDGET_NOISE:
        return True
    if _TIMESTAMP.match(t):
        return True
    if _FONT_COMPOUND.match(t):
        return True
    if _SHORT_WORD.match(t):
        return True
    if _LONG_HEX.match(t):
        return True
    if _FILESIZE.match(t):
        return True
    # Pure integers: keep only plausible years (1800-2100), drop everything else
    if _PURE_INT.match(t):
        n = int(t)
        if not (1800 <= n <= 2100):
            return True
    return False


def is_good_label_word(token: str) -> bool:
    """True if a token is suitable as a human-readable topic-label NAME.

    Stricter than `not is_noise_token`: a label name should be a real word —
    not noise, length 3-30, and containing a vowel (filters consonant-garbage
    fragments like 'wg', 'rq', 'jb', 'gp', 'hn0' that survive as keywords).
    Used to pick a clean label name from a topic's keyword list instead of
    blindly taking the top keyword (which may be a noise token).
    """
    t = _PUNCT_STRIP.sub('', str(token).lower().strip())
    if is_noise_token(t):
        return False
    if not (3 <= len(t) <= 30):
        return False
    if not _VOWEL.search(t):
        return False
    return True


def clean_text(text: str) -> str:
    """Remove PDF artifacts and file-listing noise from a plain-text string."""
    if not text or not text.strip():
        return ""

    # Remove PDF escape sequences
    text = _PDF_ESCAPE.sub(" ", text)

    # Tokenise on whitespace, filter noise tokens, rejoin
    tokens = text.split()
    clean  = [t for t in tokens if not is_noise_token(t)]

    return " ".join(clean)


def clean_texts(texts: list[str]) -> list[str]:
    """Apply clean_text to a list, keeping alignment (empty → 'unknown')."""
    cleaned = [clean_text(t) for t in texts]
    return [t if t.strip() else "unknown" for t in cleaned]


# Common German function words (standard public-domain stopword list, same
# words used by spaCy/NLTK's German corpus). Needed because the corpus is
# ~30% German and sklearn's stop_words='english' only covers English.
GERMAN_STOPWORDS = frozenset({
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines",
    "einem", "einen", "und", "oder", "aber", "auch", "auf", "aus", "bei",
    "bis", "durch", "für", "gegen", "in", "im", "ist", "ich", "sich", "sie",
    "sind", "mit", "nach", "ohne", "seit", "über", "um", "unter", "von",
    "vor", "während", "wegen", "zu", "zur", "zum", "zwischen", "als", "am",
    "an", "be", "bin", "bist", "da", "damit", "dann", "darf", "dass",
    "dazu", "deine", "deiner", "denn", "deshalb", "diese", "dieser", "dieses",
    "doch", "dort", "du", "er", "es", "etwas", "euch", "eure", "hab", "habe",
    "haben", "hat", "hatte", "hatten", "hier", "ihr", "ihre", "ihren", "ihrer",
    "ihres", "immer", "ja", "jede", "jeder", "jedes", "jener", "jetzt", "kann",
    "kein", "keine", "können", "könnte", "machen", "man", "mehr", "mein",
    "meine", "mich", "mir", "muss", "müssen", "nicht", "nichts", "noch",
    "nun", "nur", "ob", "oben", "schon", "sein", "seine", "selbst", "so",
    "soll", "sollte", "sonst", "uns", "unser", "viel", "vielen", "wann",
    "war", "waren", "warum", "was", "weiter", "weitere", "wenn", "wer",
    "werde", "werden", "wie", "wieder", "will", "wir", "wird", "wirst",
    "wo", "wollen", "wurde", "wurden", "zwar", "zwei",
})

# DLR site-boilerplate / navigation tokens, observed empirically polluting
# topic keyword lists (e.g. dlr/home/news/impressum dominating top-10 words).
DLR_DOMAIN_STOPWORDS = frozenset({
    "dlr", "home", "news", "impressum", "kontakt", "contact", "datenschutz",
    "imprint", "sitemap", "search", "login", "agb", "cookie", "cookies",
    "weiterlesen", "mehr", "menu", "navigation", "footer", "header",
    "veranstaltungen", "termine", "presse", "press", "downloadbereich",
    # German-gov accessibility-statement boilerplate (site-wide footer/legal
    # pages: "Barrierefreiheit", arbitration body/procedure, the BITV/BGG
    # accessibility laws) — boilerplate, not research content.
    "barrierefreiheit", "schlichtungsstelle", "schlichtungsverfahren",
    "bitv", "bgg",
})


def _umlaut_variants(word: str) -> set:
    """All forms of a word that may appear in the corpus's extracted text.

    The WARC extraction left two German text variants behind: umlauts preserved
    (für), and umlauts *deleted* (fr) — not folded. ß behaves the same way.
    Generate both the standard ASCII fold (ü→ue) and the deletion form so the
    stopword set matches every variant.
    """
    deleted = (word.replace("ä", "").replace("ö", "").replace("ü", "")
                   .replace("ß", "").replace("Ä", "").replace("Ö", "")
                   .replace("Ü", ""))
    folded = (word.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
                  .replace("ß", "ss").replace("Ä", "ae").replace("Ö", "oe")
                  .replace("Ü", "ue"))
    return {word, deleted, folded}


def get_combined_stopwords() -> frozenset:
    """English (sklearn) + German (all umlaut variants) + DLR-domain stopwords."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    german_expanded = set()
    for w in GERMAN_STOPWORDS:
        german_expanded |= _umlaut_variants(w)
    domain_expanded = set()
    for w in DLR_DOMAIN_STOPWORDS:
        domain_expanded |= _umlaut_variants(w)
    return frozenset(ENGLISH_STOP_WORDS) | german_expanded | domain_expanded
