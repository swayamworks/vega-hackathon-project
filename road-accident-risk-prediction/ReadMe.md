# AI-Based Road Accident Hotspot Detection & Risk Prediction System  
### National Highways Authority of India (NHAI) – Hackathon Submission

---

## 📌 Problem Statement

The National Highways Authority of India (NHAI) oversees thousands of kilometers of national highways.  
India records a high number of road accidents annually due to factors such as:

- Speeding  
- Poor visibility zones  
- Adverse weather conditions  
- Heavy traffic density  
- Night driving  
- Alcohol involvement  

Currently, accident data is often analyzed **reactively**, after incidents occur.

This project proposes a **proactive AI-based system** that:

- Identifies accident-prone hotspots  
- Predicts high-risk incidents  
- Visualizes geographic risk patterns  
- Highlights contributing risk factors  

---

## 🧠 Solution Overview

We developed a complete end-to-end AI pipeline consisting of:

### 1️⃣ Spatial Clustering (DBSCAN)
- Detects dense accident regions (hotspots)
- Does not require predefining number of clusters
- Handles irregular geographic patterns

### 2️⃣ Feature Engineering
Captures real-world nonlinear interactions:
- Rain × Alcohol  
- Night × Speed  
- Cluster density effects  
- Cyclical hour encoding (sin/cos transformation)

### 3️⃣ Risk Prediction Model
- Algorithm: **HistGradientBoostingClassifier**
- Handles nonlinear relationships effectively
- Produces continuous risk scores
- Threshold tuned for balanced precision and recall

### 4️⃣ Interactive Geographic Dashboard
- Dark / Light mode toggle
- Heatmap visualization
- Cluster markers
- Risk-level color coding (Low / Moderate / Critical)
- KPI summary panel
- Top high-risk zones display

---

## 📊 Model Performance

- **ROC-AUC:** 0.84  
- **Accuracy:** 81%  
- **Precision (High Risk):** 70%  
- **Recall (High Risk):** 64%  

The threshold was tuned to balance high-risk detection with false positives.

---

## 🗺️ Dashboard Capabilities

- Geographic accident visualization  
- Hotspot detection  
- Risk scoring per incident  
- Top 5 high-risk clusters  
- Theme switching  
- Interactive popups  

---

## 🏗️ System Architecture

1. Data Preprocessing  
2. Spatial Clustering (DBSCAN)  
3. Feature Engineering  
4. Gradient Boosting Model Training  
5. Risk Scoring  
6. Dashboard Dataset Export  
7. Interactive Map Rendering  

---

## ⚠️ Dataset Note

Due to the absence of publicly available structured highway datasets with sufficient feature granularity, a **domain-informed synthetic dataset** was generated to validate the system architecture.

The full modeling pipeline and dashboard design are fully transferable to real NHAI accident datasets.

---

## 🚀 How to Run

### Step 1 — Train Model & Generate Dashboard Dataset
```bash
python Model.py
```

### Step 2 — Launch Dashboard
```bash
python Dashboard.py
```

Output file generated:
```
AI_Model_Risk_Map_Pro.html
```

Open this file in a browser to view the interactive dashboard.

---

## 📈 Scalability & Deployment Potential

This system can be extended to:

- Integrate NHAI GIS road network data  
- Connect to live traffic and weather APIs  
- Expose model via REST API (FastAPI/Flask)  
- Enable real-time highway risk monitoring  
- Assist patrol allocation and preventive planning  

---

## 🎯 Impact

This solution enables:

- Proactive accident prevention  
- Data-driven highway safety planning  
- Efficient patrol deployment  
- Early identification of dangerous zones  
- Risk-based intervention strategies  

---

## 🧩 Future Enhancements

- Real accident dataset integration  
- Time-of-day risk trend analysis  
- Live risk prediction interface  
- Road-network-aware hotspot detection  
- Cloud deployment pipeline  

---

## 👥 Team

AI-Based Road Safety Intelligence System  
Hackathon Submission