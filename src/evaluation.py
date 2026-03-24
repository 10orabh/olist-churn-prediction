import logging
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Union
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, log_loss, roc_auc_score
)
class Evaluation(ABC):
    '''
    Abstract base class for defining evaluation strategies.
    
    '''
    @abstractmethod
    def calculate_scores(self, y_true: Union[np.ndarray,pd.Series], y_pred:Union[np.ndarray,pd.Series]) -> Union[float,np.ndarray ]  :
        ''' 
        Abstract method to calculate evaluation scores.
        
        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        
        Returns:
            None 
        '''
        pass

class accuracy(Evaluation):
    def calculate_scores(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
        try:
            logging.info("Calculating accuracy score...")
            acc = accuracy_score(y_true, y_pred)
            logging.info(f"Accuracy score calculated successfully: {acc}")
            return float(acc)
        except Exception as e:
            logging.error(f"Error calculating accuracy score: {e}")
            raise

class precision(Evaluation):
    def calculate_scores(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
        try:
            logging.info("Calculating precision score...")
            prec = precision_score(y_true, y_pred)
            logging.info(f"Precision score calculated successfully: {prec}")
            return float(prec)
        except Exception as e:
            logging.error(f"Error calculating precision score: {e}")
            raise

class recall(Evaluation):
    def calculate_scores(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
        try:
            logging.info("Calculating recall score...")
            rec = recall_score(y_true, y_pred)
            logging.info(f"Recall score calculated successfully: {rec}")
            return float(rec)
        except Exception as e:
            logging.error(f"Error calculating recall score: {e}")
            raise

class f1(Evaluation):
    def calculate_scores(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
        try:
            logging.info("Calculating F1 score...")
            f1_sc = f1_score(y_true, y_pred)
            logging.info(f"F1 score calculated successfully: {f1_sc}")
            return float(f1_sc)
        except Exception as e:
            logging.error(f"Error calculating F1 score: {e}")
            raise

class confusion(Evaluation):
    def calculate_scores(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> np.ndarray:
        try:
            logging.info("Calculating confusion matrix...")
            conf_matrix = confusion_matrix(y_true, y_pred)
            logging.info(f"Confusion matrix calculated successfully:\n{conf_matrix}")
            return conf_matrix
        except Exception as e:
            logging.error(f"Error calculating confusion matrix: {e}")
            raise
class logloss(Evaluation):
    def calculate_scores(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
        try:
            logging.info("Calculating log loss...")
            log_loss_score = log_loss(y_true, y_pred)
            logging.info(f"Log loss calculated successfully: {log_loss_score}")
            return float(log_loss_score)
        except Exception as e:
            logging.error(f"Error calculating log loss: {e}")
            raise

class roc_auc(Evaluation):
    def calculate_scores(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
        try:
            logging.info("Calculating ROC AUC score...")
            roc_auc_score_value = roc_auc_score(y_true, y_pred)
            logging.info(f"ROC AUC score calculated successfully: {roc_auc_score_value}")
            return float(roc_auc_score_value)
        except Exception as e:
            logging.error(f"Error calculating ROC AUC score: {e}")
            raise