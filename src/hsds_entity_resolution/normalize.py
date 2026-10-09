"""Conservative, pure normalisers for HSDS contact and address fields.

Every function takes and returns plain Python values (``str``, ``tuple``) and has no
dependency on the rest of the engine, so a writer or flatten step outside the engine
can adopt them unchanged. Each one is idempotent: normalising an already-normalised
value returns it unchanged.

"Conservative" means a function only removes variation that never changes what a
value refers to (case, surrounding whitespace, invisible formatting characters, a
trailing slash). It never guesses: an unparseable phone keeps its digits, an address
is not expanded or abbreviated, and nothing is looked up.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

__all__ = [
    "normalize_address_component",
    "normalize_email",
    "normalize_phone",
    "normalize_postal_code",
    "normalize_url",
]

# Extension markers seen in HSDS phone fields: "x12", "ext 12", "ext. 12",
# "extension 12", and "x12" written straight after the number. Matched
# case-insensitively at the end of the text.
_EXTENSION_PATTERN = re.compile(
    r"(?:\bext(?:ension)?\.?|(?<![a-z])x)\s*:?\s*(\d+)\s*$", re.IGNORECASE
)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_TRAILING_PUNCTUATION_PATTERN = re.compile(r"[\s.,]+$")
_ZIP_PATTERN = re.compile(r"^(\d{5})(?:-?\d{4})?$")
_NANP_LENGTH = 10


def _strip_invisible(value: str) -> str:
    """Remove format characters (bidi marks, zero-width spaces) and normalise width.

    Some exports wrap values in invisible Unicode, e.g. ``"\\u202a555-010-0199\\u202c"``.
    NFKC folds full-width digits and letters to ASCII; category ``Cf`` removes the
    invisible format characters.
    """
    folded = unicodedata.normalize("NFKC", value)
    return "".join(char for char in folded if unicodedata.category(char) != "Cf")


def normalize_phone(raw: str) -> tuple[str, str | None]:
    """Reduce a phone number to its digits and an optional extension.

    A leading North American country code is dropped when it turns an eleven-digit
    number into a ten-digit one (``"+1 555 010 0199"`` → ``"5550100199"``). Short codes
    such as ``"988"`` or ``"211"`` are kept as they are.

    Args:
        raw: Phone text as stored in the source, e.g. ``"(555) 010-0199 ext. 12"``.

    Returns:
        ``(digits, extension)``. ``digits`` is ``""`` when the text has no digits;
        ``extension`` is ``None`` when no extension marker is present.
    """
    text = _strip_invisible(raw).strip()
    extension: str | None = None
    match = _EXTENSION_PATTERN.search(text)
    if match is not None:
        extension = match.group(1)
        text = text[: match.start()]
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) == _NANP_LENGTH + 1 and digits.startswith("1"):
        digits = digits[1:]
    return digits, extension


def normalize_email(raw: str) -> str:
    """Lowercase and trim an email address, dropping a ``mailto:`` prefix and brackets.

    Args:
        raw: Email text, e.g. ``" <Intake@Example.ORG> "``.

    Returns:
        The normalised address, e.g. ``"intake@example.org"``; ``""`` for blank input.
    """
    text = _strip_invisible(raw).strip().strip("<>").strip()
    if text.lower().startswith("mailto:"):
        text = text[len("mailto:") :]
    return text.strip().lower()


def normalize_url(raw: str) -> str:
    """Normalise a website URL's scheme, host case and trailing slash.

    A missing scheme becomes ``https``. Scheme and host are lowercased; the path,
    query and fragment keep their case because servers may treat them as
    case-sensitive. Trailing slashes are removed from the path.

    Args:
        raw: URL text, e.g. ``"WWW.Example.org/Services/"``.

    Returns:
        The normalised URL, e.g. ``"https://www.example.org/Services"``; ``""`` for blank
        input. Text containing square brackets is returned trimmed but otherwise as-is.
    """
    text = _strip_invisible(raw).strip()
    if not text:
        return ""
    if "://" not in text:
        text = f"https://{text.lstrip('/')}"
    if "[" in text or "]" in text:
        # Brackets are only valid around an IPv6 host, which resource directories do
        # not publish; urlsplit rejects malformed ones. Leave such text as it is.
        return text
    parts = urlsplit(text)
    path = parts.path.rstrip("/")
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, parts.query, parts.fragment)
    )


def normalize_address_component(raw: str) -> str:
    """Normalise one address component for comparison.

    Lowercases, strips invisible characters, collapses runs of whitespace, and trims
    trailing periods and commas. Street-type words are deliberately not expanded or
    abbreviated (``"St"`` stays ``"st"``): that needs a gazetteer to be right.

    Args:
        raw: One component such as an address line, city or state.

    Returns:
        The normalised component; ``""`` for blank input.
    """
    text = _WHITESPACE_PATTERN.sub(" ", _strip_invisible(raw)).strip()
    text = _TRAILING_PUNCTUATION_PATTERN.sub("", text)
    return text.lower()


def normalize_postal_code(raw: str) -> str:
    """Reduce a US ZIP or ZIP+4 to its five-digit ZIP; leave other codes trimmed.

    Args:
        raw: Postal code text, e.g. ``"62701-1234"``.

    Returns:
        ``"62701"`` for a US ZIP or ZIP+4; otherwise the trimmed, lowercased input.
    """
    text = _strip_invisible(raw).strip()
    match = _ZIP_PATTERN.match(text)
    if match is not None:
        return match.group(1)
    return text.lower()
