"""Configuration utilities for the TicTacToe bot."""
from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ConfigError(f"Missing required environment variable: {name}")
    return value.strip()


@dataclass(frozen=True)
class Settings:
    bot_token: str
    mongo_url: str
    owner_id: int
    api_id: int
    api_hash: str
    join_timeout_sec: int = 120
    turn_timeout_sec: int = 90
    callback_throttle_sec: float = 1.2

    @property
    def owner_ids(self) -> set[int]:
        return {self.owner_id}



def load_settings() -> Settings:
    """Load and validate all required environment variables."""
    bot_token = _get_env("BOT_TOKEN")
    mongo_url = _get_env("MONGO_URL")
    owner_id = int(_get_env("OWNER_ID"))
    api_id = int(_get_env("API_ID"))
    api_hash = _get_env("API_HASH")

    join_timeout = int(os.getenv("JOIN_TIMEOUT_SEC", "120"))
    turn_timeout = int(os.getenv("TURN_TIMEOUT_SEC", "90"))
    callback_throttle = float(os.getenv("CALLBACK_THROTTLE_SEC", "1.2"))

    return Settings(
        bot_token=bot_token,
        mongo_url=mongo_url,
        owner_id=owner_id,
        api_id=api_id,
        api_hash=api_hash,
        join_timeout_sec=join_timeout,
        turn_timeout_sec=turn_timeout,
        callback_throttle_sec=callback_throttle,
    )


def is_owner(user_id: int, settings: Settings) -> bool:
    """Check if the given user id belongs to the bot owner."""
    return user_id in settings.owner_ids

