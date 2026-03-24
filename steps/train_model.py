import pandas as pd 
from src.logger import get_logger
from zenml import step
from src.model_dev import LogisticRegressionModel
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig
logger = get_logger("model_training_step")
@step
def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    model_config: ModelNameConfig
    ) -> ClassifierMixin: #type: ignore
    
    """
    Train the model on the ingest data.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training labels.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing labels.
        model_config (ModelNameConfig): Configuration for the model.
    
    Returns:
        ClassifierMixin: The trained model instance.
    """
    try:
        logger.info("Starting model training...")
        
        model = None
        if model_config.model_name == "LogisticRegression":
            model = LogisticRegressionModel()
            trained_model = model.train(X_train, y_train)
            logger.info("Model training completed successfully.")
            return trained_model
        else:
            raise ValueError(f"Unsupported model name: {model_config.model_name}")
    except Exception as e:
        logger.error(f"Error occurred while training the model: {e}")
        raise
    