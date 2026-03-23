from src.logger import get_logger
import pandas as pd 
from zenml import step

@step
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates the given model on the test data and returns evaluation metrics.

    Args:
        model: The trained machine learning model to be evaluated.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The true labels for the test data. """
    pass