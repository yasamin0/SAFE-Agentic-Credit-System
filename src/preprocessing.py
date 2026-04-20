# src/preprocessing.py

def preprocessing_note():
    """
    Return a short explanation of where preprocessing currently happens.

    At the moment, preprocessing is not implemented here as separate reusable
    functions. Instead, the actual preprocessing logic lives inside
    data_preprocessing_tool in data_loader.py.

    That preprocessing includes:
    - train/test split
    - one-hot encoding of categorical features
    - scaling of numerical features
    - handling of the sensitive feature for fairness analysis
    """
    return (
        "Preprocessing is currently handled inside data_preprocessing_tool "
        "using train/test split, one-hot encoding, scaling, and sensitive-feature handling."
    )