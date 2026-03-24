import logging
from abc import ABC, abstractmethod
from typing import Any 
from sklearn.linear_model import LogisticRegression
import pandas as pd

class Model(ABC):
    """Abstract base class for machine learning models."""
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Any:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for the model
            
        Returns: 
            Any: The trained model instance
        """
        pass

class LogisticRegressionModel(Model):
    """
    Logistic Regression model
    
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> LogisticRegression:
        """Train the logistic regression model."""
        try:
            logging.info("Starting model training...")
            
            reg  = LogisticRegression(**kwargs)
            reg.fit(X_train, y_train)
            
            logging.info("Model training completed successfully.")
            return reg
        
        except Exception as e:
            logging.error(f"Error occurred while training the model: {e}")
            raise