"""
This module contains the configuration settings for the ML model.
It defines the ModelSettings class, which uses Pydantic to manage
configuration settings, allowing settings to be read from environment
variables and .env file.
"""

from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """
    Configuration settings for the application.

    Attributes:
        model_config (SettingsConfigDict): Configuration for reading settings
        from a .env file.
        model_output_path (DirectoryPath): Path to the directory where the
        trained model will be saved.
        model_results_path (str): Path to the directory where model results
        will be saved.
        model_name (str): Name of the trained model file.
        model_results_name (str): Name of the file where model results will
        be saved.
        random_state (int): Random state for reproducibility of results.
    """

    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_output_path: DirectoryPath
    model_results_path: str
    model_name: str
    model_results_name: str
    random_state: int


# Initialize settings and configure logging
model_settings = ModelSettings()
