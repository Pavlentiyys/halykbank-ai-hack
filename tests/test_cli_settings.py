from argparse import Namespace
from pathlib import Path

from cli.settings import CONTACT_EMAIL, TEAM_NAME, load_settings


def test_submission_identity_is_embedded_in_cli_config(tmp_path: Path) -> None:
    args = Namespace(
        env_file=tmp_path / "missing.env",
        gemma_model=None,
        gemma_endpoint=None,
        llm_mode=None,
        workers=None,
        fx_eur_usd=None,
        team=None,
        contact_email=None,
        offline=True,
        no_cache=True,
    )
    settings = load_settings(args)
    assert settings.team == TEAM_NAME == "Astrea"
    assert settings.contact_email == CONTACT_EMAIL == "pashpichug@mail.ru"
    assert settings.ensemble.llm_mode == "gaps-only"
