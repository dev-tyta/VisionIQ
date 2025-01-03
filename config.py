from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    PROJECT_NAME: str = "VisionIQ"

    model_config = SettingsConfigDict(gcase_sensitive=True)


def get_settings():
    return Settings()


settings = Settings()