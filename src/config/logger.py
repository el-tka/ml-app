"""
This module contains the configuration settings for the logging.
It defines the LoggerSettings class, which uses Pydantic to manage
configuration settings, allowing settings to be read from environment
variables and .env file.
"""

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggerSettings(BaseSettings):
    """
    Configuration settings for the logging.

    Attributes:
        model_config (SettingsConfigDict): Configuration for reading settings
        from a .env file.
        log_level (str): Logging level for the application.
    """

    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str


def configure_logging(log_level: str) -> None:
    """
    Configures the logging settings for the application.

    Args:
        log_level (str): The logging level to be set for the application.

    Returns:
        None
    """
    logger.remove()  # Remove the default logger to avoid duplicate logs
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="2 days",
        compression="zip",
        level=log_level,
    )


# Configure logging
configure_logging(LoggerSettings().log_level)
