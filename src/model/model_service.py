"""
This module provides functionality for managing the ML model,
including loading the model and making predictions.

It contains the ModelService class, which handles loading and
using the pre-trained model. The class offers methods
to load the model from a file, building it if it doesn't exist,
and making predictions using the loaded model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelService:
    """
    A service class for managing the ML model, including loading and
    making predictions.

    This class provides functionality to load a pre-trained model from a
    specified path, build the model if it doesn't exist, and make predictions
    using the loaded model.

    Attributes:
        model (object): The loaded ML model. Initially set to None until the
            model is loaded.
        model_path (Path): The path to the model file.
        features (list): The list of feature names used for predictions.
        model_name (str): The name of the loaded model.

    Methods:
        __init__(self, model_path): Initializes the ModelService instance with
            the specified model path.
        load_model(self): Loads the model from the specified path, building it
            if it doesn't exist.
        predict(self, input_variables): Makes predictions using the loaded
            model based on the provided input variables.
    """

    def __init__(
        self,
        model_path: str = (
            f'{model_settings.model_output_path}/{model_settings.model_name}'
        ),
    ) -> None:
        """Initializes the ModelService instance."""
        self.model = None
        self.model_path = Path(model_path)
        self.features = None
        self.model_name = None

    def load_model(self) -> None:
        """Loads the model and metadata, building it if it doesn't exist."""
        logger.info(
            f'Checking if model exists at {self.model_path}',
            )
                     
        # 1. Check if model exists
        if not self.model_path.exists():
            logger.warning(
                f'Model not found at {self.model_path}.'
                f'Starting training {model_settings.model_name}',
                )
            build_model()

        # 2. Load artifact
        with open(self.model_path, 'rb') as model_file:
            logger.info(
                f'Loading model from {self.model_path}',
                )
            artifact = joblib.load(model_file)

        # 3. Extract trained model
        self.model = artifact['model']
        self.features = artifact['features']
        self.model_name = artifact['model_name']

        logger.info(
            f'Model loaded: {self.model_name}',
            )

    def predict(self,
                input_variables: list,) -> np.ndarray:
        """Makes predictions using the loaded model based on the provided
        input variables.
         
        Takes a list of input variables, converts them into a DataFrame, and
        uses the loaded model to make predictions.
        Returns the predictions as a NumPy array.

        Args:
        input_variables (list): A list of input variables for making
        predictions.

        Returns:
        numpy.ndarray: An array of predictions made by the model.
        """
        logger.info(
            f'Making predictions using the model: {self.model_name}',
        )

        if self.model is None:
            logger.error(
                'Model is not loaded. Call load_model() first.',
            )
            raise ValueError('Model is not loaded. Call load_model() first.')

        # Convert input to DataFrame
        x = pd.DataFrame(
            [input_variables],
            columns=self.features,
        )

        predictions = self.model.predict(x)

        return predictions

