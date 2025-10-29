import joblib
# import numpy as np
# import logging

def predict_data(features):
    """
    Predict using the saved model.
    Args:
        features (list): Input features for prediction.
    Returns:
        np.ndarray: Predicted class.
    """
    model = joblib.load("../model/wine_model.pkl")
    prediction = model.predict(features)
    return prediction
