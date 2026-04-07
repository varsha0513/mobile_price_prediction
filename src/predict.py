import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Load model
try:
    model = joblib.load("models/model.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Try loading scaler (if exists)
try:
    scaler = joblib.load("models/scaler.pkl")
    logger.info("Scaler loaded successfully")
except Exception as e:
    logger.warning(f"Scaler not found or failed to load: {str(e)}")
    scaler = None


def predict(data: list):
    """
    Predict mobile phone price range
    
    Args:
        data: list of feature values (length = 20)
    
    Returns:
        int: predicted price range (0-3)
    """
    try:
        data = np.array(data).reshape(1, -1)

        if scaler is not None:
            data = scaler.transform(data)

        prediction = model.predict(data)
        logger.debug(f"Prediction computed: {prediction[0]}")
        
        return int(prediction[0])
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise