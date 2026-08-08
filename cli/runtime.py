"""Interactive lifecycle helper for the local Ollama runtime."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Set
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from model import Settings

OLLAMA_INSTALL_URL = "https://ollama.com/download"
OLLAMA_INSTALL_SCRIPT = "curl -fsSL https://ollama.com/install.sh | sh"


def _installed_models(endpoint: str) -> Set[str]:
    url = endpoint.rstrip("/") + "/api/tags"
    with urlopen(url, timeout=2) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        str(model.get("name") or model.get("model"))
        for model in payload.get("models", [])
        if model.get("name") or model.get("model")
    }


def _is_local(endpoint: str) -> bool:
    host = urlparse(endpoint).hostname
    return host in {"localhost", "127.0.0.1", "::1"}


def _find_ollama() -> Optional[str]:
    discovered = shutil.which("ollama")
    if discovered:
        return discovered
    candidates = [
        Path("/Applications/Ollama.app/Contents/Resources/ollama"),
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
    ]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe")
    return next((str(path) for path in candidates if path.is_file()), None)


def _offline_settings(settings: Settings) -> Settings:
    gemma = replace(settings.ensemble.gemma, enabled=False)
    ensemble = replace(settings.ensemble, gemma=gemma)
    return replace(settings, ensemble=ensemble, model_name="offline numeric")


def _with_model(settings: Settings, model_id: str) -> Settings:
    gemma = replace(settings.ensemble.gemma, model_id=model_id)
    ensemble = replace(settings.ensemble, gemma=gemma)
    return replace(settings, ensemble=ensemble, model_name="{} + numeric".format(model_id))


def _choice(
    prompt: str,
    allowed: Set[str],
    default: str,
    input_fn: Callable[[str], str],
) -> str:
    while True:
        answer = input_fn(prompt).strip().casefold() or default
        if answer in allowed:
            return answer
        print("Неизвестный вариант: {}".format(answer))


def _install_ollama(input_fn: Callable[[str], str]) -> bool:
    system = platform.system()
    if system in {"Darwin", "Linux"} and shutil.which("curl"):
        print("Запускаю официальный установщик: {}".format(OLLAMA_INSTALL_SCRIPT))
        result = subprocess.run(["sh", "-c", OLLAMA_INSTALL_SCRIPT], check=False)
        return result.returncode == 0 and _find_ollama() is not None

    if system == "Windows":
        print("Открываю официальный установщик Ollama для Windows...")
        webbrowser.open(OLLAMA_INSTALL_URL + "/windows")
        input_fn("Установите Ollama, затем нажмите Enter для продолжения: ")
        return _find_ollama() is not None

    print("Автоматическая установка для {} недоступна.".format(system or "этой ОС"))
    print("Установите Ollama с {} и повторите запуск.".format(OLLAMA_INSTALL_URL))
    return False


def _start_server(executable: str, endpoint: str) -> Set[str]:
    print("Ollama не запущен — запускаю локальный сервер...")
    subprocess.Popen(
        [executable, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 30
    while True:
        try:
            return _installed_models(endpoint)
        except (OSError, URLError, TimeoutError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError("Ollama не запустилась за 30 секунд") from exc
            time.sleep(0.5)


def _offer_offline_or_cancel(
    settings: Settings,
    input_fn: Callable[[str], str],
) -> Optional[Settings]:
    choice = _choice(
        "[o] Запустить offline  [q] Отмена: ",
        {"o", "q"},
        "o",
        input_fn,
    )
    return _offline_settings(settings) if choice == "o" else None


def ensure_gemma_runtime(
    settings: Settings,
    *,
    input_fn: Optional[Callable[[str], str]] = None,
    interactive: Optional[bool] = None,
) -> Optional[Settings]:
    """Prepare Ollama/model, offer another model, or switch to offline mode."""
    gemma = settings.ensemble.gemma
    if not gemma.enabled:
        return settings

    input_fn = input_fn or input
    if interactive is None:
        interactive = sys.stdin.isatty()

    try:
        models = _installed_models(gemma.endpoint)
        executable = _find_ollama()
    except (OSError, URLError, TimeoutError):
        if not _is_local(gemma.endpoint):
            raise RuntimeError(
                "Endpoint {} недоступен. Проверьте --gemma-endpoint или используйте --offline.".format(
                    gemma.endpoint
                )
            )
        executable = _find_ollama()
        if executable is None:
            if not interactive:
                raise RuntimeError(
                    "Ollama не установлена. Установите её с {} или запустите с --offline.".format(
                        OLLAMA_INSTALL_URL
                    )
                )
            print("Ollama не найдена на этом компьютере.")
            choice = _choice(
                "[i] Установить Ollama  [o] Offline  [q] Отмена: ",
                {"i", "o", "q"},
                "i",
                input_fn,
            )
            if choice == "o":
                return _offline_settings(settings)
            if choice == "q":
                return None
            if not _install_ollama(input_fn):
                return _offer_offline_or_cancel(settings, input_fn)
            executable = _find_ollama()
            if executable is None:
                return _offer_offline_or_cancel(settings, input_fn)
        models = _start_server(executable, gemma.endpoint)

    if gemma.model_id in models:
        return settings
    if not interactive:
        raise RuntimeError(
            "Модель {} не установлена. Выполните `ollama pull {}` или используйте --offline.".format(
                gemma.model_id,
                gemma.model_id,
            )
        )

    print("Модель {} не найдена. Установлены: {}".format(
        gemma.model_id,
        ", ".join(sorted(models)) or "нет",
    ))
    can_pull = _is_local(gemma.endpoint) and executable is not None
    if can_pull:
        choice = _choice(
            "[d] Скачать выбранную  [m] Указать другую  [o] Offline  [q] Отмена: ",
            {"d", "m", "o", "q"},
            "d",
            input_fn,
        )
    else:
        print("Скачать модель на удалённый endpoint из этого CLI нельзя.")
        choice = _choice(
            "[m] Указать установленную модель  [o] Offline  [q] Отмена: ",
            {"m", "o", "q"},
            "m",
            input_fn,
        )
    if choice == "o":
        return _offline_settings(settings)
    if choice == "q":
        return None
    if choice == "m":
        model_id = input_fn("Имя модели Ollama: ").strip()
        if not model_id:
            return _offer_offline_or_cancel(settings, input_fn)
        settings = _with_model(settings, model_id)
        gemma = settings.ensemble.gemma
        if model_id in models:
            return settings

    executable = executable or _find_ollama()
    if executable is None:
        raise RuntimeError("Ollama установлена, но команда ollama не найдена в PATH")
    print("Скачиваю модель {}...".format(gemma.model_id))
    result = subprocess.run([executable, "pull", gemma.model_id], check=False)
    if result.returncode != 0:
        print("Не удалось скачать модель {}.".format(gemma.model_id))
        return _offer_offline_or_cancel(settings, input_fn)
    return settings
