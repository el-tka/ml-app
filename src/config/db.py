"""
This module contains the configuration settings for the database.
It defines the Settings class, which uses Pydantic to manage
configuration settings, allowing settings to be read from environment
variables and .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine


class DbSettings(BaseSettings):
    """
    Configuration settings for the database.

    Attributes:
        model_config (SettingsConfigDict): Configuration for reading settings
        from a .env file.
        rent_apartments_table_name (str): Name of the database table
        containing rent apartment data.
        connection_string (str): Database connection string for SQLAlchemy
        engine.
    """

    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields in the .env file
    )

    rent_apartments_table_name: str
    connection_string: str
    

# Initialize settings
db_settings = DbSettings()

# Create database engine using the connection string from settings
engine = create_engine(db_settings.connection_string)
