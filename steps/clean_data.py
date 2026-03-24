from src.logger import get_logger
import pandas as pd 
from zenml import step

@step
def clean_df(data: pd.DataFrame) -> None:
    pass