import logging
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from typing import Tuple, Union

class DataStrategy(ABC):
    """
    Abstract base class for defining data strategies.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Concrete strategy for data preprocessing.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Example preprocessing steps
        try:
            logging.info("Starting data preprocessing...")
            # Handle missing values
            data = data.fillna(data.mean())
            # Encode categorical variables
            data = pd.get_dummies(data)

            # remove outliers
            data['monetary'] = data['monetary'].clip(
    upper=data['monetary'].quantile(0.99)
)
            logging.info("Data preprocessing completed successfully.")
            return data
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise

class DataSplitterStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            logging.info("Starting data splitting...")
            X = data.drop('churn_status', axis=1)
            y = data['churn_status']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info("Data splitting completed successfully.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error during data splitting: {e}")
            raise
        
class DataCleaning:
    def __init__(self,data: pd.DataFrame,strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error during data cleaning: {e}")
            raise

if __name__ == "__main__":
    data = pd.read_csv("D:\\sourabh project\\olist-churn-prediction\\data\\raw\\data.csv")
    
    # Data Preprocessing
    preprocess_strategy = DataPreProcessStrategy()
    data_cleaner = DataCleaning(data, preprocess_strategy)
    data_cleaner.handle_data()
    
