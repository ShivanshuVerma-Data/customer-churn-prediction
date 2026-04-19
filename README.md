# Customer Churn Prediction

End-to-end machine learning project to predict customer churn using classification models, with precision-focused optimization and an interactive Streamlit UI.

---

# Problem Statement

Customer churn is a major challenge for subscription-based businesses.  
This project predicts whether a customer is likely to churn, enabling proactive retention strategies.

---

# Approach

- Data preprocessing and cleaning  
- Feature engineering (behavior-based features)  
- Handling class imbalance using class weights  
- Model training using:
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Model selection using F1-score  
- Threshold tuning to optimize **precision** (reduce false positives)  

---

# Results

- F1 Score: ~0.62  
- Accuracy: ~77%  
- ROC-AUC: ~0.84  
- Precision improved through threshold tuning  
- Balanced trade-off between precision and recall  

---

# Key Insights

- Low tenure customers are more likely to churn  
- Higher monthly charges increase churn risk  
- Contract type strongly influences retention  
- New customers are at higher risk  

---

# Features

- Modular ML pipeline (preprocessing → training → evaluation)  
- Multiple model comparison  
- Precision-focused threshold optimization  
- Feature importance analysis  
- Streamlit UI for real-time predictions  

---

# Project Structure

```
Customer_churn/
│
├── app/ # Streamlit UI
├── src/ # ML pipeline (preprocess, train, evaluate)
├── main.py # Training entry point
├── requirements.txt # Dependencies
└── README.md
```

---

# Installation

```bash
pip install -r requirements.txt
How to Run
1. Train the model
python main.py
2. Run the Streamlit app
streamlit run app/app.py
```
Then open:
```
http://localhost:8501
```

---

## Demo

# The app allows you to:

```
Input customer details
Predict churn probability
View risk level
See contributing features
```
link : https://customer-churn-prediction-rnnmwgmy26qzsyzsgmtziu.streamlit.app/

---

# Screenshots 

## Inputs : 

<img width="805" height="743" alt="image" src="https://github.com/user-attachments/assets/5d575002-787f-4c32-ab0c-efeaf2588ec8" />

## Outputs : 

<img width="560" height="496" alt="image" src="https://github.com/user-attachments/assets/34cfc9c0-4267-40ff-acb9-9fbf835ae08a" />


# Tech Stack
```
Python
Pandas, NumPy
Scikit-learn
XGBoost
Matplotlib
Streamlit
```

---

## Future Improvements
```
Add SHAP for better explainability
Deploy the app online
Improve feature engineering
Add real-time data support
```
---

# Author
## Shivanshu Verma
