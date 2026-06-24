"""propose: strongly-typed project settings via pydantic-settings.
input: .env file, environment variables, defaults.
output: singleton settings instance with computed absolute paths."""

# ruff: noqa: N802 -- uppercase computed field names follow pydantic-settings convention

from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: I001

BASE_DIR: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    BASE_DIR: Path = BASE_DIR

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
        case_sensitive=False,
    )

    PROJECT_NAME: str = "atari-rl-benchmarking"
    ENV: str = "development"

    # -- private relative targets --
    _DATA_REL: Path = Path("data")
    _ARTIFACTS_REL: Path = Path("artifacts")
    _LOGS_REL: Path = Path("logs")
    _FIGURES_REL: Path = Path("evals/figures")
    _CONFIGS_REL: Path = Path("configs")
    _PROMPTS_REL: Path = Path("prompts")

    # -- public absolute paths (computed) --
    @computed_field
    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / self._DATA_REL

    @computed_field
    @property
    def ARTIFACTS_DIR(self) -> Path:
        return self.BASE_DIR / self._ARTIFACTS_REL

    @computed_field
    @property
    def LOGS_DIR(self) -> Path:
        return self.BASE_DIR / self._LOGS_REL

    @computed_field
    @property
    def FIGURES_DIR(self) -> Path:
        return self.BASE_DIR / self._FIGURES_REL

    @computed_field
    @property
    def CONFIGS_DIR(self) -> Path:
        return self.BASE_DIR / self._CONFIGS_REL

    @computed_field
    @property
    def PROMPTS_DIR(self) -> Path:
        return self.BASE_DIR / self._PROMPTS_REL

    # -- data subdirectories --
    @computed_field
    @property
    def RAW_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @computed_field
    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @computed_field
    @property
    def TRAIN_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "train"

    @computed_field
    @property
    def VAL_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "val"

    @computed_field
    @property
    def TEST_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "test"

    @computed_field
    @property
    def SAMPLES_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "samples"

    @computed_field
    @property
    def CURATED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "curated"

    @computed_field
    @property
    def INSTRUCTIONS_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "instructions"

    # -- artifacts subdirectories --
    @computed_field
    @property
    def PREPARING_DIR(self) -> Path:
        return self.ARTIFACTS_DIR / "preparing"

    @computed_field
    @property
    def TRAINING_DIR(self) -> Path:
        return self.ARTIFACTS_DIR / "training"

    @computed_field
    @property
    def EVALUATION_DIR(self) -> Path:
        return self.ARTIFACTS_DIR / "evaluation"

    @computed_field
    @property
    def EXPORTS_DIR(self) -> Path:
        return self.ARTIFACTS_DIR / "exports"

    def create_required_directories(self) -> None:
        for dir_path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.TRAIN_DATA_DIR,
            self.VAL_DATA_DIR,
            self.TEST_DATA_DIR,
            self.SAMPLES_DATA_DIR,
            self.CURATED_DATA_DIR,
            self.INSTRUCTIONS_DATA_DIR,
            self.ARTIFACTS_DIR,
            self.PREPARING_DIR,
            self.TRAINING_DIR,
            self.EVALUATION_DIR,
            self.EXPORTS_DIR,
            self.LOGS_DIR,
            self.FIGURES_DIR,
            self.CONFIGS_DIR,
            self.PROMPTS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.create_required_directories()
