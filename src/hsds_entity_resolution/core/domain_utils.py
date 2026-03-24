"""Domain parsing and overlap helpers shared across candidate and scoring stages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import tldextract

from hsds_entity_resolution.core.dataframe_utils import clean_string_list, clean_text_scalar

_SCHEME_SEPARATOR = "://"
_WWW_PREFIX_PATTERN = re.compile(r"^www\.")


@dataclass(frozen=True)
class _DomainToken:
    """Parsed URL or email domain carrying the fields needed for graded overlap scoring."""

    normalized_host: str
    """Full hostname after ``www.`` prefix removal, lowercased."""

    registered_domain: str
    """eTLD+1 (registrable domain) via tldextract, e.g. ``unitedway.org``.

    Falls back to ``normalized_host`` when tldextract cannot resolve a public
    suffix (private TLDs, bare IP addresses, ``localhost``, etc.).
    """

    has_path: bool
    """True when the original URL contained a non-empty path component (e.g. ``/services``).

    Always ``False`` for email-derived tokens because emails carry no path.
    """


def extract_domain(value: object) -> str | None:
    """Extract normalized domain from an email address or URL-like website value."""
    if not isinstance(value, str):
        return None
    normalized = clean_text_scalar(value)
    if not normalized:
        return None
    if "@" in normalized:
        return _extract_email_domain(normalized)
    return _extract_website_domain(normalized)


def extract_contact_domains(*, emails_value: object, websites_value: object) -> set[str]:
    """Extract unique normalized domains from email and website collections."""
    domains: set[str] = set()
    for email in clean_string_list(emails_value):
        domain = extract_domain(email)
        if domain:
            domains.add(domain)
    for website in clean_string_list(websites_value):
        domain = extract_domain(website)
        if domain:
            domains.add(domain)
    return domains


def domain_overlap_score(
    *,
    left_emails: object,
    left_websites: object,
    right_emails: object,
    right_websites: object,
) -> float:
    """Return a graded domain overlap score in ``[0.0, 1.0]``.

    Computes the highest match score found across all pairs of contact URLs
    and emails from the two entities.

    Score tiers:

    ``1.0``
        Same hostname (after ``www.`` strip) *and* neither URL carries a path
        component — a clean root match
        (e.g. ``https://www.abc.com`` vs ``https://abc.com``).

    ``0.8``
        Same hostname but at least one URL has a path slug
        (e.g. ``https://abc.com/programs`` vs ``https://abc.com``).

    ``0.4``
        Same registrable domain (eTLD+1) but different subdomains — signals
        the same umbrella organization hosting multiple services under distinct
        sub-domains (e.g. ``services.unitedway.org`` vs
        ``outreach.unitedway.org``).

    ``0.0``
        No domain overlap, or either side has no parseable contact domains.
    """
    left_tokens = _collect_domain_tokens(emails_value=left_emails, websites_value=left_websites)
    right_tokens = _collect_domain_tokens(emails_value=right_emails, websites_value=right_websites)
    if not left_tokens or not right_tokens:
        return 0.0
    best = 0.0
    for lt in left_tokens:
        for rt in right_tokens:
            score = _pair_domain_score(lt, rt)
            if score >= 1.0:
                return 1.0
            if score > best:
                best = score
    return best


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_email_domain(value: str) -> str | None:
    """Extract domain from a normalized email-like value."""
    email_value = value
    if email_value.startswith("mailto:"):
        email_value = email_value.removeprefix("mailto:")
    local_part, separator, domain_part = email_value.rpartition("@")
    if not separator or not local_part or not domain_part:
        return None
    return _normalize_domain(domain_part)


def _extract_website_domain(value: str) -> str | None:
    """Extract host domain from a normalized URL or bare domain value."""
    candidate = value if _SCHEME_SEPARATOR in value else f"http://{value}"
    parsed = urlsplit(candidate)
    host = parsed.netloc
    if not host:
        return None
    if "@" in host:
        host = host.rsplit("@", maxsplit=1)[1]
    if ":" in host:
        host = host.split(":", maxsplit=1)[0]
    return _normalize_domain(host)


def _normalize_domain(domain: str) -> str | None:
    """Normalize domain for deterministic comparison."""
    normalized = clean_text_scalar(domain).strip(".")
    if not normalized:
        return None
    normalized = _WWW_PREFIX_PATTERN.sub("", normalized)
    return normalized or None


def _parse_domain_token(value: str, *, is_email: bool) -> _DomainToken | None:
    """Parse one URL or email string into a ``_DomainToken`` for graded matching.

    Emails never carry a path; websites record whether the URL has a non-root
    path component.  Falls back to ``normalized_host`` as the registered domain
    when tldextract cannot resolve a public suffix.
    """
    normalized = clean_text_scalar(value)
    if not normalized:
        return None

    if is_email:
        if normalized.startswith("mailto:"):
            normalized = normalized.removeprefix("mailto:")
        _, sep, host_raw = normalized.rpartition("@")
        if not sep or not host_raw:
            return None
        host = clean_text_scalar(host_raw).strip(".")
        has_path = False
    else:
        candidate = normalized if _SCHEME_SEPARATOR in normalized else f"http://{normalized}"
        parsed = urlsplit(candidate)
        host_with_port = parsed.netloc
        if not host_with_port:
            return None
        if "@" in host_with_port:
            host_with_port = host_with_port.rsplit("@", maxsplit=1)[1]
        if ":" in host_with_port:
            host_with_port = host_with_port.split(":", maxsplit=1)[0]
        host = clean_text_scalar(host_with_port).strip(".")
        has_path = bool(parsed.path.strip("/"))

    if not host:
        return None

    normalized_host = _WWW_PREFIX_PATTERN.sub("", host)
    if not normalized_host:
        return None

    extracted = tldextract.extract(normalized_host)
    if extracted.domain and extracted.suffix:
        registered = f"{extracted.domain}.{extracted.suffix}"
    else:
        registered = normalized_host

    return _DomainToken(
        normalized_host=normalized_host,
        registered_domain=registered,
        has_path=has_path,
    )


def _collect_domain_tokens(*, emails_value: object, websites_value: object) -> list[_DomainToken]:
    """Collect ``_DomainToken`` objects from email and website collections."""
    tokens: list[_DomainToken] = []
    for email in clean_string_list(emails_value):
        token = _parse_domain_token(email, is_email=True)
        if token is not None:
            tokens.append(token)
    for website in clean_string_list(websites_value):
        token = _parse_domain_token(website, is_email=False)
        if token is not None:
            tokens.append(token)
    return tokens


def _pair_domain_score(left: _DomainToken, right: _DomainToken) -> float:
    """Score one left-right domain token pair.

    Returns:
        1.0 — same hostname, no path on either URL (clean root match).
        0.8 — same hostname, path present on at least one URL.
        0.4 — same registrable domain (eTLD+1), different subdomains.
        0.0 — no shared domain.
    """
    if left.registered_domain != right.registered_domain:
        return 0.0
    if left.normalized_host != right.normalized_host:
        return 0.4
    if not left.has_path and not right.has_path:
        return 1.0
    return 0.8
