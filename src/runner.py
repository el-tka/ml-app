"""
Runner script for the application.

This script serves as the entry point for the application. 
It initializes the necessary components, loads the trained model, 
performs predictions on sample input data and logs the results.
"""
from loguru import logger

from model.model_service import ModelService


@logger.catch
def main() -> None:
    """
    Main function to run the application.

    It initializes the ModelService, loads the trained model,
    performs predictions on sample input data, and logs the results.

    Returns:
        None
    """
    logger.info("Running the application...")
    ml_svc = ModelService()
    ml_svc.load_model()

    input_data = {
        'area': 100,
        'constraction_year': 2000,
        'bedrooms': 3,
        'garden': 0,
        'balcony_yes': 0,
        'parking_yes': 0,
        'furnished_yes': 0,
        'garage_yes': 0,
        'storage_yes': 0,
    }
    predictions = ml_svc.predict(list(input_data.values()))
    logger.info(f'Predicted rent price: {predictions[0]}')


if __name__ == "__main__":
    main()
