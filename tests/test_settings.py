from src.config.config import Settings


def test_create_required_directories_creates_all_paths(tmp_path) -> None:
    settings = Settings(BASE_DIR=tmp_path)

    settings.create_required_directories()

    expected = [
        settings.DATA_DIR,
        settings.LOGS_DIR,
        settings.FIGURES_DIR,
    ]
    for path in expected:
        assert (tmp_path / path).exists(), f"{path} not created"
