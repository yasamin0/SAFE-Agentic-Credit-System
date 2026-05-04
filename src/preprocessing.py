# src/preprocessing.py

def preprocessing_note():
    """
    Return a short note about the current preprocessing location.

    The real preprocessing logic currently lives in:
    src/data_loader.py -> data_preprocessing_tool
    """
    return (
        "Preprocessing is currently handled inside data_preprocessing_tool "
        "in src/data_loader.py. It includes train/test split, one-hot encoding, "
        "scaling, and sensitive-feature handling for fairness analysis."
    )