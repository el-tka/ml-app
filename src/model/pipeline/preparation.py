"""
This module provides functionality to prepare data for further analysis and 
modeling.

It includes functions to load data from a database, encode categorical columns, 
parse specific columns, and check for missing columns.
"""

import pandas as pd
from loguru import logger

from model.pipeline.collection import load_data_from_db

feature_cols = ['area',
                'constraction_year', 
                'bedrooms', 
                'garden',
                'balcony_yes', 
                'parking_yes', 
                'furnished_yes', 
                'garage_yes', 
                'storage_yes']


def prepare_data() -> pd.DataFrame:
    """
    Prepares the data for further analysis and modeling.

    It performs the following steps:
    1. Loads the data from the database using the `load_data_from_db` function.
    2. Encodes categorical columns using one-hot encoding.
    3. Parses the 'garden' column to extract numeric values.
    4. Checks for missing columns and adds them with default values if necessary.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared data.
    """
    logger.info('Starting preprocessing the data')
    dataframe = load_data_from_db()
    encoded_data = _encode_cat_cols(dataframe)
    parsed_data = _parse_garden_col(encoded_data)
    return _col_check(parsed_data)


def _encode_cat_cols(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns in the DataFrame using one-hot encoding.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing categorical columns.
    
    Returns:
        pd.DataFrame: A DataFrame with categorical columns encoded as one-hot vectors.
    """
    cat_cols = ['balcony', 
                'parking', 
                'furnished', 
                'garage', 
                'storage',
                ]
    logger.info(f'Encoding categorical columns: {cat_cols}')
    return pd.get_dummies(dataframe.copy(), 
                           columns=cat_cols, 
                           drop_first=True,
    )   


def _parse_garden_col(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the 'garden' column in the DataFrame to extract numeric values.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the 'garden' column.

    Returns:
        pd.DataFrame: A DataFrame with the 'garden' column parsed to contain numeric values.
    """
    logger.info('Parsing garden column')
    dataframe['garden'] = (dataframe['garden'].astype(str).str.extract(r'(\d+)', 
                                                     expand=False).fillna('0').astype(float))
    return dataframe


def _col_check(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for missing columns in the DataFrame and adds them with default 
    values if necessary.

    Args:
        dataframe (pd.DataFrame): The input DataFrame to check for missing columns.

    Returns:
        pd.DataFrame: A DataFrame with missing columns added and filled with 
        default values.
    """
    logger.info('Checking for missing columns')

    for col in feature_cols:
        if col not in dataframe.columns:
            dataframe[col] = 0
    return dataframe
