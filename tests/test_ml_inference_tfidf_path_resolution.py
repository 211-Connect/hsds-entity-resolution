"""Tests for TF-IDF vectorizer path resolution in ML inference."""

from __future__ import annotations

from pathlib import Path

from hsds_entity_resolution.core import ml_inference


def test_resolve_tfidf_vectorizer_path_prefers_env_override(monkeypatch) -> None:
    """Env override directory should take precedence when multiple paths exist."""
    filename = "tfidf_vectorizer_organization.joblib"
    override_dir = Path("/tmp/custom-tfidf")
    override_path = override_dir / filename
    package_path = Path(ml_inference.__file__).resolve().parents[1] / "tf_idf_models" / filename
    monkeypatch.setenv("AI_UTILS_TFIDF_MODEL_DIR", str(override_dir))

    def fake_exists(path: Path) -> bool:
        return path in {override_path, package_path}

    monkeypatch.setattr(Path, "exists", fake_exists)

    resolved = ml_inference._resolve_tfidf_vectorizer_path(entity_type="organization")

    assert resolved == override_path


def test_resolve_tfidf_vectorizer_path_uses_package_local_models(monkeypatch) -> None:
    """Package-local TF-IDF path should be used when override is not set."""
    filename = "tfidf_vectorizer_service.joblib"
    package_path = Path(ml_inference.__file__).resolve().parents[1] / "tf_idf_models" / filename
    monkeypatch.delenv("AI_UTILS_TFIDF_MODEL_DIR", raising=False)

    def fake_exists(path: Path) -> bool:
        return path == package_path

    monkeypatch.setattr(Path, "exists", fake_exists)

    resolved = ml_inference._resolve_tfidf_vectorizer_path(entity_type="service")

    assert resolved == package_path


def test_resolve_tfidf_vectorizer_path_returns_none_when_missing(monkeypatch) -> None:
    """Resolution should return None when no known TF-IDF model path exists."""
    monkeypatch.delenv("AI_UTILS_TFIDF_MODEL_DIR", raising=False)
    monkeypatch.setattr(Path, "exists", lambda _path: False)

    resolved = ml_inference._resolve_tfidf_vectorizer_path(entity_type="organization")

    assert resolved is None
