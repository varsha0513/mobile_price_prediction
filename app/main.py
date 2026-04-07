import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.predict import predict

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mobile Price Prediction API",
    version="1.0.0",
    description="Predicts mobile phone price ranges based on features"
)

# Input schema
class MobileFeatures(BaseModel):
    features: list


@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Mobile Price Prediction API Running"}


@app.get("/health")
def health_check():
    """Health check endpoint for Azure monitoring"""
    logger.debug("Health check performed")
    return {
        "status": "healthy",
        "service": "mobile-price-prediction"
    }


@app.post("/predict")
def get_prediction(data: MobileFeatures):
    """
    Predict mobile phone price range
    
    Args:
        data: MobileFeatures containing list of 20 feature values
    
    Returns:
        Dictionary with predicted price_range
    """
    try:
        logger.info(f"Prediction request received with {len(data.features)} features")
        
        if len(data.features) != 20:
            logger.warning(f"Invalid feature count: expected 20, got {len(data.features)}")
            raise HTTPException(status_code=400, detail="Expected 20 features")
        
        result = predict(data.features)
        logger.info(f"Prediction successful: {result}")
        return {"price_range": result}
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


if __name__ == "__main__":
    logger.info("Starting Mobile Price Prediction API")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)