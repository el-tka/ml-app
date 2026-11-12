"""
This modeule provides functionality to load data from a database table.

It includes a function to load data from the database using SQLAlchemy 
and return it as a pandas DataFrame.
The module uses the SQLAlchemy ORM to query the database and retrieve data 
from the specified table. It also uses pandas to convert the query results 
into a DataFrame for further analysis and processing. 
"""

import pandas as pd
from loguru import logger
from sqlalchemy import select

from config import engine
from db.db_model import RentApartment


def load_data_from_db() -> pd.DataFrame:
    """
    Loads data from the database table RentApartment.

    Returns:
        pd.DataFrame: A DataFrame containing the data loaded from the database table.
    """
    logger.info('Loading data from database table')
    query = select(RentApartment)
    return pd.read_sql(query, engine)
