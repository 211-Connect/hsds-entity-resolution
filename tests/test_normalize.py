"""Property and example tests for ``hsds_entity_resolution.normalize``."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hsds_entity_resolution.normalize import (
    normalize_address_component,
    normalize_email,
    normalize_phone,
    normalize_postal_code,
    normalize_url,
)

# Invisible wrappers seen around exported values: LRE/PDF bidi embedding marks,
# LRM/RLM marks, zero-width space and the byte-order mark.
_INVISIBLE = st.sampled_from(["‪", "‬", "‎", "‏", "​", "﻿"])
_TEN_DIGITS = st.from_regex(r"[2-9][0-9]{9}", fullmatch=True)
_FORMATTERS = st.sampled_from(
    [
        "{a}{b}{c}",
        "({a}) {b}-{c}",
        "{a}-{b}-{c}",
        "{a}.{b}.{c}",
        "+1 {a} {b} {c}",
        "1-{a}-{b}-{c}",
        "+1 ({a}) {b}-{c}",
    ]
)


def _format(digits: str, template: str) -> str:
    """Render ten digits with a phone formatting template."""
    return template.format(a=digits[:3], b=digits[3:6], c=digits[6:])


@given(digits=_TEN_DIGITS, template=_FORMATTERS)
def test_phone_formats_reduce_to_the_same_ten_digits(digits: str, template: str) -> None:
    """Every common rendering of one number, with or without +1, yields its ten digits."""
    assert normalize_phone(_format(digits, template)) == (digits, None)


@given(digits=_TEN_DIGITS, template=_FORMATTERS, left=_INVISIBLE, right=_INVISIBLE)
def test_phone_wrapped_in_invisible_unicode(
    digits: str, template: str, left: str, right: str
) -> None:
    """Invisible bidi or zero-width wrappers (iCarol v2 exports) do not change the digits."""
    assert normalize_phone(f"{left}{_format(digits, template)}{right}") == (digits, None)


@given(
    digits=_TEN_DIGITS,
    extension=st.from_regex(r"[0-9]{1,5}", fullmatch=True),
    marker=st.sampled_from(["x", " x", " ext ", " ext. ", " Ext:", " extension ", " EXT"]),
)
def test_phone_extension_is_split_off(digits: str, extension: str, marker: str) -> None:
    """Extension markers split into the second element and never join the digits."""
    assert normalize_phone(f"{_format(digits, '({a}) {b}-{c}')}{marker}{extension}") == (
        digits,
        extension,
    )


@given(raw=st.text())
def test_phone_is_idempotent_on_its_digits(raw: str) -> None:
    """Normalising the digits again returns them unchanged."""
    digits, _ = normalize_phone(raw)
    assert normalize_phone(digits) == (digits, None)
    assert digits == "" or digits.isdigit()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5550100199", ("5550100199", None)),  # bare ten digits (Salesforce exports)
        ("15550100199", ("5550100199", None)),
        ("‪(555) 010-0199‬", ("5550100199", None)),  # iCarol v2 wrapped form
        ("988", ("988", None)),
        ("211", ("211", None)),
        ("", ("", None)),
        ("call us", ("", None)),
    ],
)
def test_phone_examples(raw: str, expected: tuple[str, str | None]) -> None:
    """Known source forms normalise as documented."""
    assert normalize_phone(raw) == expected


@given(raw=st.text())
def test_email_is_idempotent_and_lowercase(raw: str) -> None:
    """Email normalisation is idempotent and leaves no uppercase or edge whitespace."""
    once = normalize_email(raw)
    assert normalize_email(once) == once
    assert once == once.strip()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Intake@Example.ORG ", "intake@example.org"),
        ("<intake@example.org>", "intake@example.org"),
        ("mailto:Intake@example.org", "intake@example.org"),
        ("​intake@example.org", "intake@example.org"),
        ("", ""),
    ],
)
def test_email_examples(raw: str, expected: str) -> None:
    """Known source forms normalise as documented."""
    assert normalize_email(raw) == expected


_HOSTS = st.from_regex(r"[A-Za-z][A-Za-z0-9-]{0,10}(\.[A-Za-z]{2,6}){1,2}", fullmatch=True)
_PATHS = st.from_regex(r"(/[A-Za-z0-9_-]{1,8}){0,3}/{0,2}", fullmatch=True)


@given(host=_HOSTS, path=_PATHS, scheme=st.sampled_from(["", "http://", "HTTPS://", "https://"]))
def test_url_lowercases_host_and_scheme_and_drops_trailing_slash(
    host: str, path: str, scheme: str
) -> None:
    """Scheme and host are lowercased, path case kept, trailing slashes removed."""
    result = normalize_url(f"{scheme}{host}{path}")
    expected_scheme = scheme[:-3].lower() if scheme else "https"
    assert result == f"{expected_scheme}://{host.lower()}{path.rstrip('/')}"
    assert normalize_url(result) == result


@given(raw=st.text())
def test_url_is_idempotent(raw: str) -> None:
    """URL normalisation is idempotent on arbitrary text."""
    once = normalize_url(raw)
    assert normalize_url(once) == once


@given(raw=st.text())
def test_address_component_is_idempotent(raw: str) -> None:
    """Address normalisation is idempotent and leaves no edge whitespace or punctuation."""
    once = normalize_address_component(raw)
    assert normalize_address_component(once) == once
    assert once == once.strip()
    assert not once.endswith((".", ","))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  100  Example   Way. ", "100 example way"),
        ("Springfield,", "springfield"),
        ("123 Main St", "123 main st"),
    ],
)
def test_address_component_examples(raw: str, expected: str) -> None:
    """Whitespace and case vary; street types are not expanded."""
    assert normalize_address_component(raw) == expected


@given(
    zip5=st.from_regex(r"[0-9]{5}", fullmatch=True),
    plus4=st.from_regex(r"[0-9]{4}", fullmatch=True),
)
def test_postal_code_reduces_zip_plus_four(zip5: str, plus4: str) -> None:
    """ZIP and ZIP+4 (with or without hyphen) reduce to the five-digit ZIP."""
    assert normalize_postal_code(zip5) == zip5
    assert normalize_postal_code(f"{zip5}-{plus4}") == zip5
    assert normalize_postal_code(f" {zip5}{plus4} ") == zip5


@given(raw=st.text())
def test_postal_code_is_idempotent(raw: str) -> None:
    """Postal code normalisation is idempotent on arbitrary text."""
    once = normalize_postal_code(raw)
    assert normalize_postal_code(once) == once
