import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.predict import predict

# Load environment variables
load_dotenv()

# Request counter for monitoring
request_count = 0

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
    """Health check endpoint for monitoring (student-friendly)"""
    logger.info("Health check performed")
    return {
        "status": "API is running",
        "service": "mobile-price-prediction",
        "requests_served": request_count
    }


@app.post("/predict")
def get_prediction(data: MobileFeatures):
    """
    Predict mobile phone price range
    
    Args:
        data: MobileFeatures containing list of 20 feature values
    
    Returns:
        Dictionary with predicted price_range and total requests served
    """
    global request_count
    request_count += 1
    
    try:
        logger.info(f"Request #{request_count}: Prediction request received with {len(data.features)} features")
        logger.info(f"Input features: {data.features}")
        
        if len(data.features) != 20:
            logger.warning(f"Invalid feature count: expected 20, got {len(data.features)}")
            raise HTTPException(status_code=400, detail="Expected 20 features")
        
        result = predict(data.features)
        logger.info(f"Request #{request_count}: Prediction successful - Price range: {result}")
        price_range = int(result)
        return {
            "price_range": price_range,
            "label": ["Low", "Medium", "High", "Very High"][price_range],
            "total_requests": request_count
        }
    
    except Exception as e:
        logger.error(f"Request #{request_count}: Prediction failed - Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


if __name__ == "__main__":
    logger.info("Starting Mobile Price Prediction API")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)