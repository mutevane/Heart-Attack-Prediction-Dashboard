
<h1 align="center">❤️ Heart Attack Risk Predictor</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Built%20With-Streamlit-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Model-XGBoost-brightgreen?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Status-Production--Ready-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />
</p>

<p align="center">
🧠 A complete ML-powered tool to predict heart attack risk using real clinical data.<br>
Built with ❤️ using XGBoost and Streamlit, featuring real-time insights, SHAP explainability,<br>
and an interactive, modern UI.
</p>

---

## 🚀 Live Demo

👉 Try the App (if hosted): [**Launch on Streamlit Cloud**](https://share.streamlit.io/your-username/heart-attack-predictor)

---

## 🧾 Full PDF Report

📘 **[Download Full Report: Heart Attack Prediction.pdf](./Heart%20Attack%20Prediction.pdf)**

Includes:
- Data cleaning & EDA
- Visualizations (KDEs, Boxplots, Heatmaps)
- Feature analysis & correlation
- Model training (10+ models)
- Hyperparameter tuning
- Evaluation metrics (accuracy, F1, AUC)
- SHAP explainability & visualizations

---

## 🧬 Features Used

| Feature               | Description                              |
|-----------------------|------------------------------------------|
| Age                  | Patient age                              |
| Gender               | 0 = Female, 1 = Male                      |
| Heart Rate           | Beats per minute                         |
| Systolic BP          | Upper blood pressure                     |
| Diastolic BP         | Lower blood pressure                     |
| Blood Sugar          | mmol/L unit                              |
| CK-MB                | Enzyme level for cardiac injury          |
| Troponin             | Protein marker indicating heart damage   |

---

## 📈 Model Performance

| Model                   | Accuracy | F1 Score | Notes                        |
|-------------------------|----------|----------|------------------------------|
| **✅ XGBoost (Deployed)** | 98.5%    | 0.984    | Final model used in app      |
| Gradient Boosting       | 98.5%    | 0.984    | Strong ensemble              |
| AdaBoost                | 98.5%    | 0.984    | Performs well with tuning    |
| Random Forest           | 98.1%    | 0.980    | Highly interpretable         |
| Stacking Classifier     | 98.5%    | 0.984    | Ensemble of top learners     |
| Logistic Regression     | 71.2%    | 0.66     | Baseline model               |
| KNN                     | 68.9%    | 0.67     | Sensitive to distance metric |

---

## 🧠 Explainability with SHAP

SHAP was used for feature importance and local interpretability.

📌 **Top Predictors**:
- Troponin
- CK-MB
- Heart Rate
- Age

📊 SHAP summary and bar plots included in PDF.

---

## 🌐 App Pages

| Section        | Description                                  |
|----------------|----------------------------------------------|
| 🏠 Welcome      | Introduction and visitor count               |
| 🧮 Prediction   | Input clinical data for real-time risk score |
| 🤖 Recommendations | Lifestyle guidance and alerts         |
| ❓ FAQ          | Explains biomarkers, model, usage            |
| ⚠ Disclaimer   | Medical warning                              |
| 📊 Analytics    | View visitor timestamps                      |

---

## 🛠 Run Locally

```bash
git clone https://github.com/your-username/heart-attack-predictor.git
cd heart-attack-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
📦 Heart-Attack-Predictor
├── app.py
├── heart_attack_xgb_model.sav
├── heart_visitors.pkl
├── requirements.txt
├── README.md
└── Heart Attack Prediction.pdf
```

---

## ⚙️ Tech Stack

- Streamlit
- XGBoost
- Scikit-Learn
- SHAP
- Pandas / Seaborn / Matplotlib

---

## 👨‍💻 Author

**Your Name**  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)

---

## 📝 License

This project is licensed under the **MIT License**.

---

> *For educational and informational purposes only. Not a substitute for professional medical advice.*
