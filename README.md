# 📱 Mobile Price Prediction API

A machine learning-based REST API that predicts mobile phone price ranges using scikit-learn models. Built with FastAPI and deployed on Render.

---

## 🚀 Features

- **ML-Powered Predictions:** Predicts mobile phone price ranges (0-3) based on 20 device features
- **REST API:** FastAPI-based API with automatic documentation (Swagger UI)
- **Health Checks:** Built-in health monitoring for production environments
- **Docker Ready:** Containerized application for easy deployment
- **Streamlit Frontend:** Interactive web interface for predictions
- **Error Handling:** Comprehensive error handling and logging
- **Lightweight Models:** Ultra-compact model files (< 3KB total)

---

## 📊 Project Structure

```
mobile_price_prediction/
├── app/
│   └── main.py              # FastAPI application
├── src/
│   ├── predict.py           # Prediction logic
│   └── train.py             # Model training script
├── models/
│   ├── model.pkl            # Trained scikit-learn model
│   └── scaler.pkl           # Feature scaler
├── data/
│   ├── train.csv            # Training data
│   └── test.csv             # Test data
├── notebooks/
│   └── test_loads.py        # Testing utilities
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── streamlit_app.py         # Web UI (Streamlit)
├── .dockerignore            # Docker ignore rules
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

---

## 🔧 Installation & Setup

### 1. **Prerequisites**
- Python 3.9+
- Docker (optional, for containerization)
- Git

### 2. **Clone Repository**
```bash
git clone <repository-url>
cd mobile_price_prediction
```

### 3. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 5. **Download Models**
- Place `model.pkl` and `scaler.pkl` in the `models/` directory
- Models should be trained using `src/train.py`

---

## 💻 Usage

### **Option 1: Run FastAPI Server Locally**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Server will be available at: `http://localhost:8000`

#### **API Endpoints:**
- **`GET /`** - Home endpoint
- **`GET /health`** - Health check
- **`GET /docs`** - Swagger UI documentation
- **`POST /predict`** - Make predictions

### **Option 2: Run Streamlit Frontend**
```bash
streamlit run streamlit_app.py
```

Interactive UI will open at: `http://localhost:8501`

### **Option 3: Docker**
```bash
# Build image
docker build -t mobile-price-api .

# Run container
docker run -p 8000:8000 mobile-price-api
```

---

## 📡 API Documentation

### **Prediction Endpoint**

**Request:**
```bash
curl -X POST "https://mobile-price-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1500, 1, 2.0, 1, 5, 1, 32, 0.5, 150, 4, 12, 1080, 1440, 2000, 12, 7, 10, 1, 1, 1]
  }'
```

**Response:**
```json
{
  "price_range": 1
}
```

### **Feature Schema (20 Features)**

| Index | Feature | Range | Unit |
|-------|---------|-------|------|
| 0 | battery_power | 500-5000 | mAh |
| 1 | blue | 0-1 | Binary |
| 2 | clock_speed | 0.5-3.5 | GHz |
| 3 | dual_sim | 0-1 | Binary |
| 4 | fc | 0-20 | Megapixels |
| 5 | four_g | 0-1 | Binary |
| 6 | int_memory | 2-512 | GB |
| 7 | m_dep | 0.1-1.0 | cm |
| 8 | mobile_wt | 80-250 | grams |
| 9 | n_cores | 2-8 | Count |
| 10 | pc | 5-20 | Megapixels |
| 11 | px_height | 0-2000 | pixels |
| 12 | px_width | 0-3000 | pixels |
| 13 | ram | 256-8000 | MB |
| 14 | sc_h | 5-20 | cm |
| 15 | sc_w | 5-15 | cm |
| 16 | talk_time | 2-25 | hours |
| 17 | three_g | 0-1 | Binary |
| 18 | touch_screen | 0-1 | Binary |
| 19 | wifi | 0-1 | Binary |

### **Price Ranges**

| Range | Category | Price |
|-------|----------|-------|
| 0 | Budget | 0-15k |
| 1 | Mid-range | 15-30k |
| 2 | Premium | 30-50k |
| 3 | Ultra-premium | 50k+ |

---

## 🌐 Deployment

### **Deployed on Render**
- **API URL:** `https://mobile-price-api.onrender.com`
- **Swagger Docs:** `https://mobile-price-api.onrender.com/docs`
- **Health Check:** `https://mobile-price-api.onrender.com/health`

### **Deployment Steps**

1. **Build Docker Image**
   ```bash
   docker build -t mobile-price-api .
   ```

2. **Push to Docker Registry**
   ```bash
   docker login
   docker tag mobile-price-api your-registry/mobile-price-api
   docker push your-registry/mobile-price-api
   ```

3. **Deploy on Render**
   - Go to https://render.com
   - Create new Web Service
   - Connect Docker image
   - Set `Port: 8000`
   - Deploy

---

## 📦 Dependencies

```
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.0
pandas==2.0.0
numpy==1.24.0
joblib==1.3.0
python-dotenv==1.0.0
requests==2.31.0
streamlit==1.28.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔍 Development

### **Train Model**
```bash
python src/train.py
```

### **Run Tests**
```bash
python notebooks/test_loads.py
```

### **View Logs**
```bash
# For FastAPI
# Check console output

# For Streamlit
# Built-in logging in sidebar
```

---

## 🐛 Troubleshooting

### **Issue: Models not found**
- **Solution:** Ensure `model.pkl` and `scaler.pkl` exist in `models/` directory
- Run `python src/train.py` to generate models

### **Issue: Port already in use**
```bash
# Change port
uvicorn app.main:app --port 8001
```

### **Issue: API returns 500 error**
- Check logs: `Get-Content <log-file>`
- Verify all dependencies installed: `pip install -r requirements.txt`
- Test locally first before deploying

### **Issue: Streamlit can't connect to API**
- Verify API is running
- Check internet connection
- Confirm API URL in `streamlit_app.py`

---

## 📝 Configuration

### **Environment Variables**
Create `.env` file:
```env
API_URL=https://mobile-price-api.onrender.com
LOG_LEVEL=INFO
WEBSITES_PORT=8000
```

Or set in deployment platform settings.

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m 'Add YourFeature'`
4. Push to branch: `git push origin feature/YourFeature`
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

**Varsh**
- GitHub: [github.com/yourprofile](https://github.com)
- Email: your.email@example.com

---

## 📞 Support

For issues, questions, or suggestions:
1. Check existing issues on GitHub
2. Create a new issue with detailed description
3. Contact via email

---

## 🔗 Quick Links

- **API Docs:** https://mobile-price-api.onrender.com/docs
- **Health Status:** https://mobile-price-api.onrender.com/health
- **GitHub:** [Repository Link]
- **Render Dashboard:** https://dashboard.render.com

---

**Last Updated:** April 8, 2026  
**Status:** ✅ Active & Production Ready
