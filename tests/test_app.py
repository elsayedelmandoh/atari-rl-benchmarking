import app
from src.config.config import settings


def test_main_returns_zero_and_prints_project_info(capsys) -> None:
    exit_code = app.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert settings.PROJECT_NAME in output
    assert settings.ENV in output
