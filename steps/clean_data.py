from src.logger import get_logger
import pandas as pd 
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitterStrategy    
from typing_extensions import Annotated
from typing import Tuple, Union
logger = get_logger("data_cleaning_step")
@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    clean the data and split it into training and testing sets.
    Args:
        data (pd.DataFrame): The input data to be cleaned and split.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The cleaned and split data.
    """
    logger.info("Starting data cleaning...")
    preprocess_strategy = DataPreProcessStrategy()
    data_cleaner = DataCleaning(data, preprocess_strategy)
    processed_data = data_cleaner.handle_data()

    if not isinstance(processed_data, pd.DataFrame):
        raise TypeError("Expected processed_data to be a pandas DataFrame")

    splitter_strategy = DataSplitterStrategy()
    data_splitter = DataCleaning(processed_data, splitter_strategy)
    split_result = data_splitter.handle_data()

    if not isinstance(split_result, tuple):
        raise TypeError("Expected split_result to be a Tuple")
    X_train, X_test, y_train, y_test = split_result
    logger.info("Data cleaning completed successfully.")
    return X_train, X_test, y_train, y_test