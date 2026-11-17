"""Environment-based OpenAI-compatible API configuration."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI


class ConfigurationError(ValueError):
    """Raised when required, non-secret environment configuration is missing."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings; repr=False prevents accidental API-key disclosure."""

    api_key: str = field(repr=False)
    base_url: str
    model: str


def load_settings() -> Settings:
    """Load .env without overriding variables already set in the environment."""
    load_dotenv(override=False)

    api_key = os.getenv("AI_API_KEY", "").strip()
    base_url = os.getenv("AI_BASE_URL", "").strip().rstrip("/")
    model = os.getenv("AI_MODEL", "").strip()

    missing = [
        name
        for name, value in (
            ("AI_API_KEY", api_key),
            ("AI_BASE_URL", base_url),
            ("AI_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise ConfigurationError(
            "Missing required setting(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in your provider values."
        )

    # Never log or print this object; although the key is hidden from repr,
    # avoiding configuration logging is a safer default.
    return Settings(api_key=api_key, base_url=base_url, model=model)


def create_client(settings: Settings) -> OpenAI:
    """Create an SDK client for the configured OpenAI-compatible endpoint."""
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)
