# 🤖 Mobile Price Prediction - ML Pipeline & Concepts

Comprehensive guide to the machine learning pipeline, concepts, and workflows used in the mobile price prediction system.

---

## 📊 ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PREPROCESSING                          │
│  Raw Data → Cleaning → Feature Engineering → Normalization      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌─────────────────────────▼────────────────────────────────────────┐
│                    TRAIN/TEST SPLIT                              │
│  80% Training Data     →     20% Testing Data                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌─────────────────────────▼────────────────────────────────────────┐
│                  FEATURE SCALING (StandardScaler)                │
│  Normalize features to mean=0, std=1                             │
│  Prevents numerical features from dominating                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌─────────────────────────▼────────────────────────────────────────┐
│              MODEL TRAINING (Classification)                     │
│  Scikit-learn Classifier (Random Forest / Gradient Boosting)     │
│  Learns patterns from 20 features → Price Range (0-3)            │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌─────────────────────────▼────────────────────────────────────────┐
│                  MODEL EVALUATION                                │
│  Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Mat  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌─────────────────────────▼────────────────────────────────────────┐
│                  MODEL SERIALIZATION                             │
│  Save: model.pkl (model) + scaler.pkl (transformer)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌─────────────────────────▼────────────────────────────────────────┐
│                  DEPLOYMENT (FastAPI)                            │
│  Production API accepts 20 features → Returns price range        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagram

```
┌──────────────────┐
│   User Input     │  20 features from mobile device specs
│  (Streamlit)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  Feature Array (List)    │
│ [battery, ram, px, ...]  │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Load Scaler (scaler.pkl)        │
│  Transform/Normalize Features    │
└────────┬───────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Load Model (model.pkl)          │
│  Run Prediction                  │
└────────┬───────────────────────────┘
         │
         ▼
┌──────────────────────┐
│   Price Range (0-3)  │
│  Budget / Mid / ...  │
└──────────────────────┘
```

---

## 📈 ML Concepts & Techniques Used

### **1. Supervised Learning - Classification**

**Type:** Multi-class Classification (4 classes: 0, 1, 2, 3)

**What it does:**
- Learns from labeled data (features + price range)
- Maps input features → output class
- Predicts which price range a phone belongs to

**Formula:**
```
f(x) = price_range
where x = [battery_power, ram, px_height, px_width, ...]
     price_range ∈ {0, 1, 2, 3}
```

---

### **2. Feature Scaling (Standardization)**

**Why needed:** Different features have different ranges
- Battery: 500-5000 (large range)
- Blue: 0-1 (small range)

**Solution: StandardScaler**
```
x_scaled = (x - mean) / std_dev

Example:
battery = 2000
mean = 2500
std = 800
battery_scaled = (2000 - 2500) / 800 = -0.625
```

**Benefits:**
- Speeds up training
- Improves model accuracy
- Prevents large-range features from dominating

---

### **3. Train/Test Split**

**Purpose:** Evaluate model on unseen data

```
Original Data (100%)
    ├─ Training Set 80% → Used for learning
    └─ Testing Set 20% → Used for evaluation
```

**Why important:**
- Training accuracy ≠ Real-world accuracy
- Tests if model generalizes well
- Detects overfitting

---

### **4. Algorithm: Random Forest / Gradient Boosting**

**Random Forest:**
```
Random Forest
├─ Decision Tree 1 ─┐
├─ Decision Tree 2  ├─→ Majority Vote → Final Prediction
├─ Decision Tree 3  │
└─ Decision Tree N ─┘

Each tree sees random subset of features & samples
More robust, reduces overfitting
```

**How it works:**
1. Creates multiple decision trees
2. Each tree votes on the prediction
3. Final answer = majority vote

**Advantages:**
- Less prone to overfitting than single tree
- Handles both numerical & categorical data
- Fast prediction time
- Feature importance ranking

---

### **5. Model Evaluation Metrics**

**Accuracy**
```
Accuracy = Correct Predictions / Total Predictions
Range: 0-1 (0-100%)
What it measures: Overall correctness
```

**Precision**
```
Precision = TP / (TP + FP)
What it measures: Of predicted positives, how many were correct?
Use when: False positives are costly
```

**Recall**
```
Recall = TP / (TP + FN)
What it measures: Of actual positives, how many were found?
Use when: False negatives are costly
```

**F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
What it measures: Balanced average of precision & recall
Range: 0-1 (higher is better)
```

**Confusion Matrix**
```
                 Predicted TRUE  Predicted FALSE
Actual TRUE      TP             FN
Actual FALSE     FP             TN

Shows correct vs incorrect predictions per class
```

---

## 🔧 Feature Engineering

### **20 Input Features**

| # | Feature | Type | Purpose | Range |
|---|---------|------|---------|-------|
| 0 | battery_power | Numerical | Device power capacity | 500-5000 mAh |
| 1 | blue | Binary | Bluetooth support | 0/1 |
| 2 | clock_speed | Numerical | Processor speed | 0.5-3.5 GHz |
| 3 | dual_sim | Binary | Dual SIM support | 0/1 |
| 4 | fc | Numerical | Front camera pixels | 0-20 MP |
| 5 | four_g | Binary | 4G support | 0/1 |
| 6 | int_memory | Numerical | Internal storage | 2-512 GB |
| 7 | m_dep | Numerical | Mobile depth | 0.1-1.0 cm |
| 8 | mobile_wt | Numerical | Mobile weight | 80-250 g |
| 9 | n_cores | Numerical | Processor cores | 2-8 |
| 10 | pc | Numerical | Rear camera pixels | 5-20 MP |
| 11 | px_height | Numerical | Screen pixels (height) | 0-2000 |
| 12 | px_width | Numerical | Screen pixels (width) | 0-3000 |
| 13 | ram | Numerical | RAM memory | 256-8000 MB |
| 14 | sc_h | Numerical | Screen height | 5-20 cm |
| 15 | sc_w | Numerical | Screen width | 5-15 cm |
| 16 | talk_time | Numerical | Battery talk time | 2-25 hours |
| 17 | three_g | Binary | 3G support | 0/1 |
| 18 | touch_screen | Binary | Touchscreen support | 0/1 |
| 19 | wifi | Binary | WiFi support | 0/1 |

### **Feature Categories**

**Performance Features:**
- clock_speed (processor)
- n_cores (processor)
- ram (memory)
- int_memory (storage)

**Design Features:**
- mobile_wt (weight)
- m_dep (depth)
- px_height, px_width (screen resolution)
- sc_h, sc_w (screen size)

**Camera Features:**
- fc (front camera)
- pc (rear camera)

**Connectivity Features:**
- blue (bluetooth)
- four_g (4G)
- three_g (3G)
- wifi (WiFi)

**Battery Features:**
- battery_power (capacity)
- talk_time (endurance)

---

## 📊 Target Variable: Price Range

**Output Classes:**
```
Class 0: Budget ($0-15k)
  → Low specs, affordable
  → Features: Low RAM, Low storage, No 4G
  
Class 1: Mid-range ($15-30k)
  → Balanced specs
  → Features: Medium RAM, Medium storage, Basic 4G
  
Class 2: Premium ($30-50k)
  → High specs
  → Features: High RAM, High storage, Great camera
  
Class 3: Ultra-premium ($50k+)
  → Flagship specs
  → Features: Highest specs all around
```

---

## 🧠 How the Model Learns

### **Training Phase**

1. **Input:** 6000+ samples with features + price range
2. **Process:**
   - For each sample, find patterns
   - Learn which features correlate with which price range
   - Adjust weights to minimize prediction error

3. **Example Learning Pattern:**
   ```
   IF (ram > 6GB) AND (processor_cores > 6):
       Likely Premium or Ultra-premium
   
   IF (4G enabled) AND (clock_speed > 2.5 GHz):
       Likely Mid-range or better
   
   IF (NO 4G) AND (RAM < 2GB):
       Likely Budget
   ```

### **Prediction Phase**

1. **Input:** New phone with 20 features
2. **Process:**
   - Load trained model & scaler
   - Normalize features using learned scaler
   - Pass through model
   - Model applies learned patterns
   - Returns probability for each class

3. **Example Prediction:**
   ```
   Input: ram=6000, clock_speed=2.8, cores=8, 4g=1
   
   Model calculates:
   P(Budget) = 0.05
   P(Mid-range) = 0.25
   P(Premium) = 0.60  ← Highest
   P(Ultra-premium) = 0.10
   
   Output: Class 2 (Premium)
   ```

---

## 💾 Model Serialization

### **Why Save Models?**

Models must be saved to disk for:
- ✅ Production deployment
- ✅ Reusing without retraining
- ✅ Version control
- ✅ Consistent predictions

### **Saved Artifacts**

**1. model.pkl** (1.49 KB)
- Contains: Trained classifier
- Created: During `src/train.py`
- Used: For making predictions

**2. scaler.pkl** (1.48 KB)
- Contains: StandardScaler fit parameters
- Stores: Mean & std deviation of training features
- Used: To normalize new features before prediction

### **How Serialization Works**

```
Training:
data → fit scaler → fit model → save scaler.pkl + model.pkl

Production:
load scaler.pkl → load model.pkl → normalize input → predict
```

---

## 🔬 Training Workflow

### **Step 1: Load Data**
```python
import pandas as pd
train_data = pd.read_csv('data/train.csv')
# Shape: (6000, 21) - 6000 samples, 20 features + 1 target
```

### **Step 2: Separate Features & Target**
```python
X = train_data.drop('price_range', axis=1)  # 20 features
y = train_data['price_range']                # Target (0-3)
```

### **Step 3: Train/Test Split**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Training: 4800 samples
# Testing: 1200 samples
```

### **Step 4: Scale Features**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# fit_transform: Learn mean/std from training data
# transform: Apply to test data
```

### **Step 5: Train Model**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
# Learns patterns from 4800 scaled training samples
```

### **Step 6: Evaluate**
```python
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")  # ~90-95% typical
print(classification_report(y_test, y_pred))
```

### **Step 7: Save Models**
```python
import joblib
joblib.dump(scaler, 'models/scaler.pkl')  # Save scaler
joblib.dump(model, 'models/model.pkl')     # Save model
```

---

## 🎯 Prediction Workflow

### **Production Prediction Flow**

```python
# 1. Load saved objects
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/model.pkl')

# 2. Get new phone features (as list)
new_phone = [2000, 1, 2.5, 1, 8, 1, 64, 0.7, 160, 6, 16, 1920, 1080, 4000, 15, 9, 20, 1, 1, 1]

# 3. Convert to array & reshape
import numpy as np
X_new = np.array(new_phone).reshape(1, -1)  # Shape: (1, 20)

# 4. Scale using saved scaler
X_new_scaled = scaler.transform(X_new)

# 5. Make prediction
prediction = model.predict(X_new_scaled)  # Returns [2]
price_range = int(prediction[0])          # Range: 0-3

# 6. Map to category
categories = {0: "Budget", 1: "Mid-range", 2: "Premium", 3: "Ultra-premium"}
result = categories[price_range]  # "Premium"
```

---

## 📉 Common ML Problems & Solutions

### **Problem 1: Overfitting**
**What:** Model memorizes training data, fails on new data
**Symptoms:** 99% training accuracy, 70% test accuracy
**Solutions:**
- Use train/test split
- Reduce model complexity
- Add regularization
- Use ensemble methods (Random Forest)

### **Problem 2: Underfitting**
**What:** Model too simple, can't learn patterns
**Symptoms:** 60% training accuracy, 60% test accuracy
**Solutions:**
- Increase model complexity
- Add more features
- Train longer
- Use more powerful algorithms

### **Problem 3: Class Imbalance**
**What:** Some price ranges more common than others
**Symptoms:** Model always predicts majority class
**Solutions:**
- Use stratified split
- Adjust class weights
- Oversample minority classes
- Use F1-score instead of accuracy

### **Problem 4: Feature Scaling Issues**
**What:** Forgot to scale before prediction
**Symptoms:** Wrong predictions after deployment
**Solutions:**
- Always scale test/production data with training scaler
- Save scaler alongside model
- Document preprocessing steps

---

## 🏗️ Model Architecture Summary

```
Input Layer (20 features)
    │
    ▼
Preprocessing (Feature Scaling)
    │
    ├─ Scaler.pkl (StandardScaler)
    │
    ▼
Random Forest Classifier
    │
    ├─ Tree 1 ──┐
    ├─ Tree 2   ├─→ Voting (Majority)
    ├─ Tree 3   │
    └─ Tree N ──┘
    │
    ▼
Output Layer (4 classes: 0, 1, 2, 3)
```

---

## 📚 Mathematical Formulas

### **Standard Scaler Formula**
```
z = (x - μ) / σ

where:
z = scaled value
x = original value
μ = mean of training data
σ = standard deviation of training data
```

### **Decision Boundary (Simplified)**
```
For each feature combination:
f(x₁, x₂, ..., x₂₀) → [p₀, p₁, p₂, p₃]
where p_i = probability of class i

Final prediction = argmax(p₀, p₁, p₂, p₃)
```

---

## 🔗 Related Files

- **Training:** [src/train.py](src/train.py) - Run to train model
- **Prediction:** [src/predict.py](src/predict.py) - Load & use model
- **API:** [app/main.py](app/main.py) - REST API wrapper
- **UI:** [streamlit_app.py](streamlit_app.py) - Interactive frontend
- **Data:** [data/train.csv](data/train.csv) - Training dataset

---

## 🚀 Next Steps for ML Enhancement

1. **Hyperparameter Tuning**
   - Experiment with Random Forest parameters
   - Use GridSearchCV for optimization

2. **Feature Importance Analysis**
   - Identify which features matter most
   - Remove low-importance features

3. **Cross-Validation**
   - Use k-fold CV for robust evaluation
   - Reduce variance in accuracy scores

4. **Ensemble Methods**
   - Combine multiple algorithms
   - Stack models for better predictions

5. **Class Imbalance Handling**
   - Use SMOTE for oversampling
   - Adjust class weights

---

**Created:** April 8, 2026  
**Status:** Production Ready  
**Model Accuracy:** ~92% (typical)
