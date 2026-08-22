# 🛡️ Zero day online payment fraud detection using Deep learning and XAI

> An advanced anomaly detection system utilizing Deep Learning (Autoencoders) and Explainable AI (SHAP & LIME) to identify unseen fraudulent behaviors in online transactions.

**Institution:** Maulana Mukhtar Ahmed Nadvi Technical Campus  
**Project Guide:** Dr. Salman Baig  

---

## 📖 Project Overview
Traditional fraud detection systems rely on rule-based logic or supervised models that memorize past frauds. However, they fail to detect **"Zero-Day Frauds"**—completely new and unseen fraudulent patterns. 

This project aims to solve this critical research gap by modeling the **behavior** of genuine users (spending patterns, geographical footprint, device history, and transaction failures). Using an **Autoencoder Neural Network**, any transaction that significantly deviates from this learned normal behavior is flagged as an anomaly. To ensure transparency, **Explainable AI (XAI)** frameworks like SHAP and LIME are integrated to explain exactly *why* a transaction was blocked.

---

## 👥 The Team
This project is developed collaboratively by:

* **Momin Shoaib Akhter Rafeeque Ahmed** (Lead - Deep Learning & XAI Local)

* **Mohammad Aun** (Deep Learning & XAI Local)

* **Mohammad Waseem** (Preprocessing & XAI Global)

* **Abdurrahman** (Preprocessing & XAI Global)

---

## 🚀 Master Project Roadmap

### ✅ Phase 0: Exploratory Data Analysis (EDA) - [COMPLETED]
* Conducted data integrity checks (missing values, duplicates).
* Identified severe class imbalance (~1.6% actual fraud cases).
* Validated behavioral features:
  * Analyzed geographical risk (`is_international`).
  * Investigated device usage patterns (`device_type`).
  * Mapped consecutive failed transaction history (`failed_txn_count_24h`).
* Generated Correlation Heatmaps for feature independence.

### ⏳ Phase 1: Data Preprocessing (In Progress)
* **Feature Selection:** Removing non-behavioral identifiers (e.g., `transaction_id`).
* **Feature Scaling:** Standardizing numerical variance using `StandardScaler`.
* **Encoding:** Converting categorical parameters using One-Hot Encoding.
* **Class Balancing:** Applying SMOTE to handle the 1.6% anomaly distribution.

### 📅 Phase 2: Deep Learning Architecture
* Developing an **Autoencoder** framework using TensorFlow/Keras.
* Training the model exclusively on genuine user behavior.
* Evaluating performance using Precision, Recall, and F1-Score to prioritize zero-day detection over raw accuracy.

### 📅 Phase 3: Explainable AI (XAI) Integration
* Implementing **LIME** for Local Interpretability (explaining individual flagged transactions).
* Implementing **SHAP** for Global Interpretability (identifying the most weighted features across the entire model).

### 📅 Phase 4: Web Application & Deployment
* Building a backend **Flask API** to serve the trained model.
* Designing a user-friendly frontend to simulate real-time transaction processing.
* Compiling the final research thesis and documentation.

---

## 💻 Getting Started (For Team Members)

To set up this project locally on your machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/sohu9/zero-day-fraud-detection.git](https://github.com/sohu9/zero-day-fraud-detection.git)
```

**2. Navigate to the project directory:**
```bash
cd zero-day-fraud-detection
```

**3. Create and activate a virtual environment:**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

**4. Install required dependencies:**
```bash
pip install -r requirements.txt
```

**5. Create your working branch:**
```bash
git checkout -b your-name-feature
```

---
*Developed with ❤️ by the final year project team.*
