"""Tests for EmbeddingAdapter provider fallback and error handling.

Key design notes from reading the implementation:

1. PRIMARY_INGESTION_EMBEDDING_RESOURCE env var controls provider selection.
   The `primary_provider` __init__ parameter is shadowed by the env var read,
   so monkeypatch.setenv must be called BEFORE constructing the adapter.

2. _should_fallback checks error *message strings* for HTTP status codes
   (e.g. "503", "service unavailable"), not exception types.  Auth errors
   (401, 403) are not in the fallback indicator list and raise immediately.

3. All-providers-fail raises RuntimeError (not a named domain exception).
   This is a known gap; see todo.md for potential improvement.

4. The adapter does NOT validate returned embedding dimensions.  If a provider
   returns vectors of wrong dimensionality, the adapter silently passes them
   through.  This is a known gap documented as a Tier C contract below.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from consumer.embeddings.embedding_adapter import EmbeddingAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    embeddings: list[list[float]] | None = None, error: Exception | None = None
) -> Mock:
    """Build a mock embedding provider resource.

    The adapter calls resource.get_client() then client.embed_batch_sync().
    """
    client = Mock()
    if error is not None:
        client.embed_batch_sync.side_effect = error
        client.embed_many_async.side_effect = error
    else:
        result = embeddings or [[0.1, 0.2, 0.3]]
        client.embed_batch_sync.return_value = result
        client.embed_many_async.return_value = result
    resource = Mock()
    resource.get_client.return_value = client
    return resource


def _adapter(
    *,
    runpod: Mock | None = None,
    modal: Mock | None = None,
    huggingface: Mock | None = None,
    max_retries: int = 1,
    monkeypatch: pytest.MonkeyPatch | None = None,
    primary: str = "runpod",
) -> EmbeddingAdapter:
    """Construct an EmbeddingAdapter with mocked providers.

    monkeypatch must be provided to control the PRIMARY_INGESTION_EMBEDDING_RESOURCE
    env var, which is read during __init__.
    """
    if monkeypatch is not None:
        monkeypatch.setenv("PRIMARY_INGESTION_EMBEDDING_RESOURCE", primary)
    return EmbeddingAdapter(
        runpod_resource=runpod,
        modal_resource=modal,
        huggingface_resource=huggingface,
        max_retries_per_provider=max_retries,
    )


# ---------------------------------------------------------------------------
# Provider fallback on server errors
# ---------------------------------------------------------------------------


def test_fallback_to_modal_when_runpod_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primary (RunPod) returning a 503 error must trigger fallback to Modal.

    HTTP 503 / 'service unavailable' is the standard transient availability
    signal; the pipeline must not hard-fail on a single provider outage.
    """
    expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    runpod = _make_provider(error=RuntimeError("503 service unavailable"))
    modal = _make_provider(embeddings=expected)
    adapter = _adapter(runpod=runpod, modal=modal, monkeypatch=monkeypatch, primary="runpod")

    result = adapter.embed_batch_sync(["text1", "text2"])

    assert result == expected


def test_fallback_to_modal_when_runpod_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    """502 (Bad Gateway / GPU provisioning failure) must also trigger fallback."""
    expected = [[0.1, 0.2]]
    runpod = _make_provider(error=RuntimeError("502 bad gateway"))
    modal = _make_provider(embeddings=expected)
    adapter = _adapter(runpod=runpod, modal=modal, monkeypatch=monkeypatch, primary="runpod")

    result = adapter.embed_batch_sync(["text1"])

    assert result == expected


def test_fallback_to_modal_when_runpod_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """500 (Internal Server Error) must trigger fallback."""
    expected = [[0.9, 0.8]]
    runpod = _make_provider(error=RuntimeError("500 internal server error"))
    modal = _make_provider(embeddings=expected)
    adapter = _adapter(runpod=runpod, modal=modal, monkeypatch=monkeypatch, primary="runpod")

    result = adapter.embed_batch_sync(["text1"])

    assert result == expected


def test_fallback_to_modal_on_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """TimeoutError on primary must trigger fallback."""
    expected = [[0.5, 0.5]]
    runpod = _make_provider(error=TimeoutError("job timeout after 30s"))
    modal = _make_provider(embeddings=expected)
    adapter = _adapter(runpod=runpod, modal=modal, monkeypatch=monkeypatch, primary="runpod")

    result = adapter.embed_batch_sync(["text1"])

    assert result == expected


def test_fallback_uses_same_texts_as_original_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback provider must receive the same texts as the original call."""
    runpod = _make_provider(error=RuntimeError("503 service unavailable"))
    modal = _make_provider(embeddings=[[0.1], [0.2], [0.3]])
    adapter = _adapter(runpod=runpod, modal=modal, monkeypatch=monkeypatch, primary="runpod")
    texts = ["alpha", "beta", "gamma"]

    adapter.embed_batch_sync(texts)

    modal_client = modal.get_client()
    modal_client.embed_batch_sync.assert_called_once()
    call_args = modal_client.embed_batch_sync.call_args
    assert call_args[0][0] == texts


# ---------------------------------------------------------------------------
# Auth errors are fatal (no fallback)
# ---------------------------------------------------------------------------


def test_auth_error_401_raises_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 Unauthorized error must raise immediately without attempting fallback.

    Authentication failures indicate a configuration problem, not a transient
    outage.  Silently falling back to Modal with the same broken credentials
    would just produce a second auth failure and obscure the root cause.
    """
    runpod = _make_provider(error=RuntimeError("401 unauthorized"))
    modal = _make_provider(embeddings=[[0.1, 0.2]])
    adapter = _adapter(runpod=runpod, modal=modal, monkeypatch=monkeypatch, primary="runpod")

    with pytest.raises(RuntimeError, match="401"):
        adapter.embed_batch_sync(["text1"])

    # Modal must never have been called
    modal_client = modal.get_client()
    modal_client.embed_batch_sync.assert_not_called()


def test_auth_error_403_raises_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 403 Forbidden error must raise immediately without attempting fallback."""
    runpod = _make_provider(error=RuntimeError("403 forbidden"))
    modal = _make_provider(embeddings=[[0.1, 0.2]])
    adapter = _adapter(runpod=runpod, modal=modal, monkeypatch=monkeypatch, primary="runpod")

    with pytest.raises(RuntimeError, match="403"):
        adapter.embed_batch_sync(["text1"])

    modal_client = modal.get_client()
    modal_client.embed_batch_sync.assert_not_called()


# ---------------------------------------------------------------------------
# All providers fail
# ---------------------------------------------------------------------------


def test_all_providers_fail_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError must be raised when every provider in the chain fails."""
    runpod = _make_provider(error=RuntimeError("503 service unavailable"))
    modal = _make_provider(error=RuntimeError("503 service unavailable"))
    adapter = _adapter(runpod=runpod, modal=modal, monkeypatch=monkeypatch, primary="runpod")

    with pytest.raises(RuntimeError, match="All embedding providers failed"):
        adapter.embed_batch_sync(["text1"])


def test_all_providers_fail_error_chains_last_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raised RuntimeError must chain the last provider's error as its cause."""
    runpod = _make_provider(error=RuntimeError("503 runpod down"))
    modal = _make_provider(error=RuntimeError("504 modal timeout"))
    adapter = _adapter(runpod=runpod, modal=modal, monkeypatch=monkeypatch, primary="runpod")

    with pytest.raises(RuntimeError) as exc_info:
        adapter.embed_batch_sync(["text1"])

    assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# Output shape invariants
# ---------------------------------------------------------------------------


def test_output_count_matches_input_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The number of returned embeddings must equal the number of input texts."""
    texts = ["alpha", "beta", "gamma"]
    embeddings = [[float(i)] for i in range(len(texts))]
    runpod = _make_provider(embeddings=embeddings)
    adapter = _adapter(runpod=runpod, monkeypatch=monkeypatch, primary="runpod")

    result = adapter.embed_batch_sync(texts)

    assert len(result) == len(texts)


def test_embed_single_returns_one_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_single must return a single flat list of floats, not a nested list."""
    runpod = _make_provider(embeddings=[[0.1, 0.2, 0.3]])
    adapter = _adapter(runpod=runpod, monkeypatch=monkeypatch, primary="runpod")

    result = adapter.embed_single("hello world")

    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Dimension validation — Tier C known gap
# ---------------------------------------------------------------------------


def test_wrong_dimension_response_is_passed_through_silently() -> None:
    """Tier C: the adapter does NOT validate embedding dimensions.

    If a provider returns vectors of wrong dimensionality, the adapter
    returns them as-is without raising an error.  This is a known gap:
    incorrect dimensions will only surface downstream when the pipeline
    attempts to compute cosine similarity or write to a typed Snowflake
    VECTOR column.

    This test documents the current (undesirable) behavior.  A future
    improvement should add dimension validation in _validate_output so
    that dimension mismatches are caught immediately at the adapter boundary.
    """
    monkeypatch_instance = pytest.MonkeyPatch()
    monkeypatch_instance.setenv("PRIMARY_INGESTION_EMBEDDING_RESOURCE", "runpod")
    wrong_dim_embeddings = [[0.1, 0.2]]  # expected 1024-dim, got 2-dim
    runpod = _make_provider(embeddings=wrong_dim_embeddings)
    adapter = _adapter(
        runpod=runpod,
        monkeypatch=None,  # already set above
        primary="runpod",
    )
    # The adapter does not raise; it returns whatever the provider gave it
    result = adapter.embed_batch_sync(["text"])
    assert result == wrong_dim_embeddings, (
        "Tier C known gap: adapter silently passes through wrong-dimension embeddings. "
        "See todo.md for proposed dimension validation improvement."
    )
    monkeypatch_instance.undo()


# ---------------------------------------------------------------------------
# Provider priority configuration
# ---------------------------------------------------------------------------


def test_modal_primary_uses_modal_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """When PRIMARY_INGESTION_EMBEDDING_RESOURCE=modal, Modal is tried first."""
    expected = [[0.7, 0.8]]
    runpod = _make_provider(embeddings=[[0.1, 0.2]])
    modal = _make_provider(embeddings=expected)
    adapter = _adapter(runpod=runpod, modal=modal, monkeypatch=monkeypatch, primary="modal")

    result = adapter.embed_batch_sync(["text"])

    assert result == expected
    modal_client = modal.get_client()
    modal_client.embed_batch_sync.assert_called_once()
    runpod_client = runpod.get_client()
    runpod_client.embed_batch_sync.assert_not_called()


def test_invalid_primary_provider_env_defaults_to_runpod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid PRIMARY_INGESTION_EMBEDDING_RESOURCE value defaults to RunPod."""
    monkeypatch.setenv("PRIMARY_INGESTION_EMBEDDING_RESOURCE", "banana")
    expected = [[0.3, 0.4]]
    runpod = _make_provider(embeddings=expected)
    modal = _make_provider(embeddings=[[0.9]])
    adapter = EmbeddingAdapter(
        runpod_resource=runpod,
        modal_resource=modal,
        max_retries_per_provider=1,
    )

    result = adapter.embed_batch_sync(["text"])

    assert result == expected
    runpod_client = runpod.get_client()
    runpod_client.embed_batch_sync.assert_called_once()
