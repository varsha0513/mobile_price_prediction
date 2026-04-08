from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_prediction():
    response = client.post("/predict", json={
        "features": [842,0,2,0,1,0,7,0,2,0,0,1,188,2,2,20,756,2549,9,7]
    })
    assert response.status_code == 200
    assert "price_range" in response.json()