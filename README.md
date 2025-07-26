
<h1 align="center">❤️ Heart Attack Risk Predictor</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Built%20With-Streamlit-red?style=flat-square&logo=streamlit">
  <img src="https://img.shields.io/badge/Model-XGBoost-green?style=flat-square&logo=scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square">
  <img src="https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square">
</p>

<p align="center">
🧠 A complete machine learning pipeline that predicts heart attack risk using clinical features. <br>
Built with data science best practices, optimized XGBoost model, and deployed via Streamlit.
</p>

---

## 📍 Overview

Predict the **probability of a heart attack** using a smart web app powered by machine learning.  
This tool helps raise awareness and supports early detection — especially for those at high risk.

📌 **Key Highlights**:
- 🧬 Based on clinical features like age, heart rate, blood pressure, troponin, and more
- 🧠 Trained on 1,300+ real-world samples
- 🏆 Uses XGBoost, one of the top-performing ML algorithms
- 🎯 Achieves ~98.5% accuracy
- 🔎 Enhanced interpretability via SHAP explainability
- 🌐 Web deployment using Streamlit

---

## 🗂️ Contents

- 📄 Full Project Report (PDF)
- 📊 Exploratory Data Analysis
- ⚙️ Model Building
- 📈 Performance Comparison
- 🧠 SHAP Explainability
- 🌐 Streamlit Web App
- 🛠 How to Run Locally
- 📁 Project Structure

---

## 📄 Project Report

🧾 All technical development, data analysis, and modeling decisions are documented here:

📥 **Download: Heart Attack Prediction.pdf**

**Report Covers:**
- Dataset profiling and cleaning
- Visual EDA and outlier detection
- Feature engineering & transformations
- Statistical testing (t-test, chi-square)
- Model tuning & evaluation
- SHAP-based explainability
- Final model export

---

## 📊 Exploratory Data Analysis

✅ Over 1,300 rows  
✅ No missing values  
✅ Visualized with `seaborn`, `matplotlib`, and `SHAP`

| Feature           | Distribution           | Notes                              |
|------------------|------------------------|------------------------------------|
| Age              | Normal (centered ~58)  | Risk ↑ after 50                    |
| Heart Rate       | Outliers > 200 bpm     | Detected and managed               |
| CK-MB & Troponin | Highly skewed          | Strong predictors of cardiac damage |
| Gender           | Binary encoded         | 1 = Male, 0 = Female               |

📌 Also includes:
- Histograms & KDEs
- Boxplots for outliers
- Correlation heatmap
- Target distribution

---

## ⚙️ Model Building

A robust set of ML models was trained and fine-tuned using:

- `GridSearchCV`, `RandomizedSearchCV`
- Stratified 10-fold CV
- `f1_macro`, `precision`, `recall` as scoring metrics

📦 Models evaluated:
- ✅ Logistic Regression
- ✅ Decision Tree
- ✅ Random Forest
- ✅ XGBoost
- ✅ Support Vector Machines (SVM)
- ✅ K-Nearest Neighbors (KNN)
- ✅ Gradient Boosting
- ✅ AdaBoost
- ✅ Voting Classifier (soft)
- ✅ Stacking Classifier
- ✅ SGD Classifier

---

## 🏁 Model Comparison

| 🔢 Model              | 🎯 Accuracy | 🧠 F1 Score | ⚡ Notes                     |
|----------------------|-------------|-------------|-----------------------------|
| XGBoost (Best)       | 98.5%       | 0.984       | Deployed                    |
| Gradient Boosting    | 98.5%       | 0.984       | Highly stable               |
| AdaBoost             | 98.5%       | 0.984       | Great with shallow trees    |
| Random Forest        | 98.1%       | 0.98        | Ensemble approach           |
| Decision Tree        | 98.1%       | 0.98        | Tuned via GridSearch        |
| Stacking Classifier  | 98.5%       | 0.984       | Best of all models          |
| SVM                  | 78.8%       | 0.78        | Tuned poly kernel           |
| Logistic Regression  | 71.2%       | 0.66        | Baseline model              |
| KNN                  | 68.9%       | 0.67        | Weak generalization         |
| SGD Classifier       | 81.4%       | 0.80        | Competitive linear baseline |

✅ XGBoost was saved with `pickle` for use in the web application.

---

## 🧠 Model Explainability

### 🔍 SHAP (SHapley Additive Explanations)

SHAP was used to:
- Visualize feature contributions
- Provide global and local explanations
- Identify key clinical drivers

🌟 **Top Features:**
- Troponin Level
- CK-MB Level
- Heart Rate
- Age
- Blood Pressure

📈 Generated with:
```python
shap.Explainer(model).shap_values(X_test)
```

---

## 🌐 Web Application

🔧 Built with: **Streamlit**

| Section        | Description                                      |
|----------------|--------------------------------------------------|
| 🏠 Welcome      | App intro, visitor count                        |
| 🧮 Prediction   | Input clinical values and get instant risk score|
| 🤖 Recommendations | Personalized tips based on input           |
| ❓ FAQ          | Explains clinical terms (e.g., CK-MB, Troponin) |
| ⚠ Disclaimer   | Not a substitute for professional diagnosis     |
| 📈 Analytics    | Shows session-based visitor count and timestamps|

💡 Supports `model.predict_proba()`  
✔️ Categorizes risk: Low / Moderate / High

---

## 📁 Project Structure

```
📦 Heart-Attack-Predictor
├── app.py                        # Streamlit frontend
├── heart_attack_xgb_model.sav   # Final XGBoost model (pickle)
├── heart_visitors.pkl           # Visitor counter (local)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── Heart Attack Prediction.pdf  # Full project report
```

---

## 🛠 Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/heart-attack-predictor.git
cd heart-attack-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

---

## ✅ Requirements

```
streamlit
pandas
numpy
scikit-learn
xgboost
seaborn
matplotlib
shap
streamlit-option-menu
```

---

## 🔐 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 👤 Author

**Your Name**  
📬 Email: your.email@example.com  
🌐 [GitHub](https://github.com/your-username) • [LinkedIn](https://linkedin.com/in/your-profile)

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)
- Open-source heart health dataset
