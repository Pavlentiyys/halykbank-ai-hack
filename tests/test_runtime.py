from urllib.error import URLError

import cli.runtime as runtime
from cli.runtime import _is_local, _offline_settings, _with_model, ensure_gemma_runtime
from model import Settings


def test_local_runtime_detection() -> None:
    assert _is_local("http://localhost:11434")
    assert _is_local("http://127.0.0.1:11434")
    assert not _is_local("https://models.example.com")


def test_runtime_can_switch_to_offline_without_mutating_original() -> None:
    original = Settings()
    offline = _offline_settings(original)

    assert original.ensemble.gemma.enabled is True
    assert offline.ensemble.gemma.enabled is False
    assert offline.model_name == "offline numeric"


def test_runtime_can_select_another_model() -> None:
    selected = _with_model(Settings(), "gemma3:4b")

    assert selected.ensemble.gemma.model_id == "gemma3:4b"
    assert selected.model_name == "gemma3:4b + numeric"


def test_missing_ollama_can_switch_to_offline(monkeypatch) -> None:
    def unavailable(_endpoint: str):
        raise URLError("not running")

    answers = iter(["o"])
    monkeypatch.setattr(runtime, "_installed_models", unavailable)
    monkeypatch.setattr(runtime, "_find_ollama", lambda: None)

    prepared = ensure_gemma_runtime(
        Settings(),
        input_fn=lambda _prompt: next(answers),
        interactive=True,
    )

    assert prepared is not None
    assert prepared.ensemble.gemma.enabled is False


def test_missing_model_can_select_installed_model(monkeypatch) -> None:
    answers = iter(["m", "gemma3:4b"])
    monkeypatch.setattr(runtime, "_installed_models", lambda _endpoint: {"gemma3:4b"})
    monkeypatch.setattr(runtime, "_find_ollama", lambda: "/usr/local/bin/ollama")

    prepared = ensure_gemma_runtime(
        Settings(),
        input_fn=lambda _prompt: next(answers),
        interactive=True,
    )

    assert prepared is not None
    assert prepared.ensemble.gemma.model_id == "gemma3:4b"
