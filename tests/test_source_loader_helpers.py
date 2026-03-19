"""Unit tests for source loader SQL helper functions."""

from __future__ import annotations

from consumer.consumer_adapter.source_loader import _literal_list, _quote_ident


def test_quote_ident_escapes_double_quotes() -> None:
    """Identifier quoting should preserve embedded quote characters."""
    assert _quote_ident('COMMON"X') == '"COMMON""X"'


def test_literal_list_escapes_single_quotes() -> None:
    """Literal list renderer should escape single-quote characters."""
    rendered = _literal_list(["A", "B'C"])
    assert rendered == "'A', 'B''C'"
