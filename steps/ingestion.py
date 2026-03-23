
import pandas as pd 
from zenml import step
from src.logger import get_logger

logger = get_logger('data_ingestion')
class IngestData:
    def __init__(self,data_path:str):
        self.data_path = data_path

    def get_data(self):
        logger.info(f"Reading data from path: {self.data_path}")
        data = pd.read_csv(self.data_path)

        return data

@step

def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingests data from the specified path using the IngestData class.

    Args:
        data_path (str): The file path where the raw data is stored.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        Exception: If the data ingestion process fails.
    """
    try:
        logger.info(f"Starting data ingestion from path: {data_path}")
        
        ingestor = IngestData(data_path)
        data = ingestor.get_data()
        
        logger.info(f"Data ingestion successful! Data shape: {data.shape}")
        return data

    except Exception as e:
        logger.error(f"Error occurred during data ingestion: {e}", exc_info=True)
        raise e