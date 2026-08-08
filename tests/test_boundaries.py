import re
import shutil
import subprocess
import sys
from pathlib import Path


def _source(root: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))


def test_model_does_not_depend_on_cli() -> None:
    source = _source(Path("model"))
    for banned in ("import cli", "from cli", "argparse", "load_dotenv", "sys.argv"):
        assert banned not in source


def test_model_imports_standalone(tmp_path: Path) -> None:
    shutil.copytree("model", tmp_path / "model")
    subprocess.run(
        [sys.executable, "-c", "import model; assert model.Settings"],
        cwd=tmp_path,
        check=True,
    )


def test_no_terminal_output_in_model() -> None:
    assert not re.search(r"(?<!\.)\bprint\(", _source(Path("model")))


def test_services_and_ensemble_do_not_select_adapters() -> None:
    source = _source(Path("model/services")) + _source(Path("model/ensemble"))
    assert "model.adapters" not in source

