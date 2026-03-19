"""Domain parsing and overlap helpers shared across candidate and scoring stages."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from hsds_entity_resolution.core.dataframe_utils import clean_string_list, clean_text_scalar

_SCHEME_SEPARATOR = "://"
_WWW_PREFIX_PATTERN = re.compile(r"^www\.")


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
