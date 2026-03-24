from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """
    Model config.
    """
    model_name : str = "LogisticRegression"