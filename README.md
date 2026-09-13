# 📊 Superstore Profit & Loss Predictor & Training Pipeline (`training-summer`)

[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Running-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](http://localhost:8501)
[![Python Version](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

> 🚀 **Quick Access URL**: **[http://localhost:8501](http://localhost:8501)**

An end-to-end Machine Learning project and interactive Streamlit web dashboard designed to analyze retail order data and predict whether an order will result in **Profit** or **Loss**.

---

## 🔗 Quick Links & URLs

- 🌐 **Local Web Application URL**: `http://localhost:8501`
- 📁 **GitHub Project Repository**: [https://github.com/karansharma947/training-summer](https://github.com/karansharma947/training-summer)

---

## 🌟 Application Features

1. 🔮 **Profit / Loss Predictor** — Real-time ML predictions with confidence percentage and discount risk indicators.
2. 📋 **Data View** — Interactive data explorer with filtering, column sorting, and CSV exports.
3. 📈 **Visualizations** — Multi-dimensional breakdowns by category, region, customer segment, and discount impacts.
4. 📝 **Findings & Conclusion** — Comparative model performance analysis (Accuracy, Precision, Recall, F1-Score) and strategic business insights.

---

## ⚡ Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Machine Learning Training Pipeline
Train Logistic Regression, Decision Tree, and Random Forest models, evaluate performance, and export the `.pkl` model bundle:
```bash
python train_model.py
```

### 3. Launch the Interactive Web Dashboard
```bash
python -m streamlit run app.py
```
Open your browser at: **[http://localhost:8501](http://localhost:8501)**

---

## 🤖 Machine Learning Pipeline (`train_model.py`)

The standalone model training pipeline evaluates three algorithms on the cleaned dataset (`superstore_cleaned.csv`):

| Model Algorithm | Accuracy | Loss Precision | Loss Recall | Loss F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** *(Selected Best)* | **94.17%** | **91.10%** | **77.34%** | **0.8366** |
| **Decision Tree** | 94.27% | 94.41% | 74.74% | 0.8343 |
| **Logistic Regression** | 93.87% | 90.18% | 76.56% | 0.8282 |

- Models are trained using multi-core parallelization (`n_jobs=-1`).
- The best model (ranked by Loss F1-Score) is exported to `superstore_profit_loss_model.pkl` for fast prediction loading.

---

## 📂 Repository Structure

```
training-2/
├── app.py                            # Streamlit web application
├── train_model.py                    # Standalone ML training pipeline script
├── superstore_cleaned.csv            # Cleaned Superstore dataset
├── superstore_profit_loss_model.pkl  # Pre-trained model bundle
├── requirements.txt                  # Python dependency list
├── README.md                         # Documentation & project guide
├── superstore_analysis.ipynb         # Data analysis notebook
├── superstore_visulization.ipynb     # Visualizations notebook
└── Superstore_model.ipynb            # Model experimentation notebook
```
