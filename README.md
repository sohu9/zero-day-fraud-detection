# 🛡️ Zero Day Online Payment Fraud Detection Using Deep Learning & XAI.

> An advanced anomaly detection system utilizing Deep Learning (Autoencoders) and Explainable AI (SHAP & LIME) to identify unseen fraudulent behaviors in online transactions.

**Institution:** Maulana Mukhtar Ahmed Nadvi Technical Campus  
**Project Guide:** Dr. Salman Baig  

---

## 📖 Project Overview
This project aims to build an intelligent, behavior-driven security system to protect online transactions. By modeling the normal behavior of genuine users (spending patterns, geographical footprint, device history), the system uses an **Autoencoder Neural Network** to flag any transaction that significantly deviates from the norm. To ensure transparency, **Explainable AI (XAI)** frameworks are integrated to explain exactly *why* a transaction was blocked.

---

## 🔬 The Research Gap (Why This Project?)
* **The Problem with Existing Systems:** Traditional fraud detection models (like Random Forests or standard Neural Networks) rely heavily on historical data. They are trained to memorize *known* fraud patterns. 
* **The Attacker's Advantage (Zero-Day Frauds):** Cybercriminals constantly evolve and use completely new, unseen methods to bypass security. Since existing systems have never seen these new patterns, they fail to detect them. 
* **Our Solution (Autoencoders):** Instead of teaching the AI *what fraud looks like*, our Autoencoder model learns *what a normal user looks like*. When an attacker uses a new method, it inherently breaks the user's normal behavioral pattern, allowing the AI to catch zero-day attacks.
* **The "Black Box" Problem & System Trust (Why XAI?):** Traditional Deep Learning models can block transactions but cannot explain *why* they did it. This creates a "black box" that makes it difficult for banks and users to trust the AI's decisions. By integrating **Explainable AI (XAI)**, our system provides genuine, transparent reasons for every suspicious transaction (e.g., "Marked suspicious due to an unusual transaction amount and unexpected device change"). This ensures that both the bank and the customer can confidently trust the model's decision-making process.

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
* Validated behavioral features (Location, Device usage, Failed transaction history).
* Generated Correlation Heatmaps for feature independence.

### ⏳ Phase 1: Data Preprocessing (In Progress)
* **Feature Selection:** Removing non-behavioral identifiers (e.g., `transaction_id`).
* **Feature Scaling:** Standardizing numerical variance using `StandardScaler`.
* **Encoding:** Converting categorical parameters using One-Hot Encoding.
* **Class Balancing:** Applying SMOTE to handle the 1.6% anomaly distribution.

### 📅 Phase 2: Deep Learning Architecture
* Developing an **Autoencoder** framework using TensorFlow/Keras.
* Training the model exclusively on genuine user behavior to spot anomalies.
* Evaluating performance using Precision, Recall, and F1-Score.

### 📅 Phase 3: Explainable AI (XAI) Integration
* Implementing **LIME** for Local Interpretability (explaining individual flagged transactions).
* Implementing **SHAP** for Global Interpretability (identifying the most weighted features).

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

---

## 🔬 The Research Gap (Why This Project?)
* **The Problem with Existing Systems:** Traditional fraud detection models (like Random Forests or standard Neural Networks) rely heavily on historical data. They are trained to memorize *known* fraud patterns. 
* **The Attacker's Advantage:** Cybercriminals constantly evolve and use completely new, unseen methods to bypass security. Since existing systems have never seen these new patterns, they fail to detect them. This is known as a **"Zero-Day Fraud"**.
* **Our Solution:** Instead of teaching the AI *what fraud looks like*, our Autoencoder model learns *what a normal user looks like*. When an attacker uses a new method, it inherently breaks the user's normal behavioral pattern. The AI detects this behavioral shift as an anomaly, successfully catching zero-day attacks that traditional systems miss.

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
* Validated behavioral features (Location, Device usage, Failed transaction history).
* Generated Correlation Heatmaps for feature independence.

### ⏳ Phase 1: Data Preprocessing (In Progress)
* **Feature Selection:** Removing non-behavioral identifiers (e.g., `transaction_id`).
* **Feature Scaling:** Standardizing numerical variance using `StandardScaler`.
* **Encoding:** Converting categorical parameters using One-Hot Encoding.
* **Class Balancing:** Applying SMOTE to handle the 1.6% anomaly distribution.

### 📅 Phase 2: Deep Learning Architecture
* Developing an **Autoencoder** framework using TensorFlow/Keras.
* Training the model exclusively on genuine user behavior to spot anomalies.
* Evaluating performance using Precision, Recall, and F1-Score.

### 📅 Phase 3: Explainable AI (XAI) Integration
* Implementing **LIME** for Local Interpretability (explaining individual flagged transactions).
* Implementing **SHAP** for Global Interpretability (identifying the most weighted features).

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
