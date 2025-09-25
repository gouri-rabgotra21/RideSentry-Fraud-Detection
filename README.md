# RideSentry - Mobility Fraud Detection Engine

## Overview

This project presents an end-to-end data science solution aimed at detecting and preventing payment fraud within a simulated ride-sharing (mobility) platform, inspired by challenges faced by companies like Uber. It encompasses data generation, in-depth exploratory data analysis, advanced feature engineering, machine learning model development (LightGBM), and a critical business impact assessment. The project also outlines a robust A/B testing framework for real-world validation.

## Problem Statement

Fraudulent activities can lead to significant financial losses, erode user trust, and strain operational resources (e.g., customer support). The goal is to build an intelligent system that accurately identifies high-risk transactions while minimizing false positives to maintain a positive user experience.

## Key Features & Methodology

### 1. Data Generation & Exploration

A synthetic dataset was generated to mimic real-world ride-sharing transactions, including user information, trip details, and payment methods, with embedded fraud patterns.

#### **Fraud Rate & Initial Insights:**

The dataset was generated with a 2% fraud rate. Initial exploratory data analysis (EDA) revealed key differences between fraudulent and legitimate trips.

**Fraud vs. Non-Fraud Fare Distribution:**
_This boxplot illustrates that fraudulent trips often involve significantly higher fare amounts, indicating a potential 'high-value' target for fraudsters._
![Fare Distribution for Fraud vs. Non-Fraud Trips](path/to/your/fare_distribution.png)
*(Replace `path/to/your/fare_distribution.png` with the actual path to your image on GitHub)*

**Fraud Rate by Hour of Day:**
_This chart shows that fraud risk isn't uniform throughout the day, with certain hours exhibiting higher relative fraud rates, which could inform real-time monitoring strategies._
![Fraud Rate by Hour of Day](path/to/your/fraud_by_hour.png)
*(Replace `path/to/your/fraud_by_hour.png` with the actual path to your image on GitHub)*

### 2. Advanced Feature Engineering

Beyond raw transaction data, several advanced features were engineered to capture behavioral patterns, velocity, and relational anomalies crucial for fraud detection:

* **User History:** `seconds_since_signup` (user tenure), `user_trip_count` (number of previous trips).
* **Velocity Features:** `user_trips_last_1h` (trip frequency within a short window).
* **Relational Features:** `num_users_on_device` (indicating shared devices, a common fraud ring indicator).
* **(Potential Upgrade):** Graph-based features (e.g., connected components, degree centrality) could further enhance detection of fraud rings.

### 3. Machine Learning Model Development (LightGBM)

A LightGBM classifier was chosen for its efficiency and strong performance on tabular data, especially with imbalanced datasets (due to its `scale_pos_weight` parameter).

#### **Model Performance (Precision-Recall Curve):**

_The Precision-Recall curve demonstrates the trade-off between identifying actual fraud (Recall) and minimizing false alarms (Precision). An Average Precision (AP) score of 0.59 significantly outperforms a random baseline, indicating robust detection capabilities._
![Precision-Recall Curve](path/to/your/precision_recall_curve.png)
*(Replace `path/to/your/precision_recall_curve.png` with the actual path to your image on GitHub)*

### 4. Business Impact Analysis & Optimal Thresholding

To translate model predictions into actionable business decisions, a cost-benefit analysis was performed.

* **Cost of False Positive (FP):** \$10 (e.g., customer support, user churn).
* **Cost of False Negative (FN):** \$150 (average fraud loss).

#### **Cost vs. Prediction Threshold:**

_This plot identifies the optimal prediction threshold that minimizes the total cost to the business, balancing the losses from missed fraud against the costs of incorrectly blocking legitimate users._
![Cost vs. Prediction Threshold](path/to/your/cost_threshold.png)
*(Replace `path/to/your/cost_threshold.png` with the actual path to your image on GitHub)*

### 5. Model Explainability (SHAP)

Understanding *why* a model makes a prediction is crucial for trust, investigation, and continuous improvement. SHAP values were used to interpret the LightGBM model.

#### **SHAP Summary Plot (Feature Importance):**

_This plot summarizes the most influential features. `fare` and `user_trip_count` (or lack thereof for new users) emerged as the strongest indicators for fraud, pushing the prediction towards higher risk._
![SHAP Summary Plot](path/to/your/shap_summary.png)
*(Replace `path/to/your/shap_summary.png` with the actual path to your image on GitHub)*

#### **SHAP Force Plot (Individual Prediction Explanation):**

_A Force Plot for a specific fraudulent transaction would visually break down how each feature contributes to pushing the model's output towards a fraudulent classification. (Screenshot omitted due to interactive nature; can be viewed in the notebook)._
*(If you can generate a static image, replace `path/to/your/shap_force.png`)*

### 6. A/B Testing Framework for Deployment

A detailed A/B test plan was designed to rigorously validate the model's real-world impact before full-scale deployment:

* **Objective:** Increase Net Financial Savings.
* **Hypothesis:** Treatment group (new model) shows a statistically significant increase in Net Financial Savings compared to Control (existing system).
* **Metrics:** Primary: Net Financial Savings. Guardrail: Trip Completion Rate, Customer Support Contact Rate, New User Retention.
* **Methodology:** Random user assignment (Control vs. Treatment), statistical testing (e.g., t-test) for significance.

## Tech Stack

* **Python:** Pandas, NumPy, Scikit-learn, LightGBM, SHAP, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook
* **Version Control:** Git & GitHub

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/gouri-rabgotra21/RideSentry-Fraud-Detection.git](https://github.com/gouri-rabgotra21/RideSentry-Fraud-Detection.git)
    cd RideSentry-Fraud-Detection
    ```
2.  **Install dependencies:** (It's good practice to have a `requirements.txt` file)
    ```bash
    pip install -r requirements.txt
    ```
    *(You can create `requirements.txt` by running `pip freeze > requirements.txt` in your project folder.)*
3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
4.  Open and run the `fraud_detection_notebook.ipynb` file.

## Future Enhancements

* Implement **graph-based features** using NetworkX for fraud ring detection.
* Simulate a **real-time feature store** and streaming analytics.
* Integrate **model monitoring** for concept drift detection.
* Experiment with **deep learning sequence models (LSTMs)** to capture temporal user behavior.

---
