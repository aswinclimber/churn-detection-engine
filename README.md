#  Telecom Customer Churn Prediction

**End-to-End Machine Learning Project**

This project predicts whether a telecom customer is likely to churn using machine learning.
It includes data visualization, model training, and a Streamlit web app for real-time prediction.

---

##  Project Structure

```
├── ada_bundle.pkl                # Final trained ML model (AdaBoost bundle)
├── telecom-customer-churn.ipynb  # Exploratory Data Analysis & Visualization
├── Telco-Customer-Churn.csv      # Dataset
├── app.py                        # Streamlit App for prediction
└── README.md                     # Project Documentation
```

---

##  Objective

Telecom companies lose revenue when customers leave ("churn").
This project helps **predict churn** early so companies can take action.

---

##  Features

###  Exploratory Data Analysis (EDA)

* Customer demographics
* Contract & service usage
* Payment patterns
* Churn distribution
* Visualizations using Matplotlib & Seaborn

###  Machine Learning

* Model: **AdaBoost Classifier**
* Preprocessing: Label Encoding, Scaling
* Evaluation: Accuracy, Confusion Matrix, Precision & Recall

###  Deployment

* Interactive **Streamlit web app**
* User-friendly UI
* Real-time predictions using the trained `.pkl` model

---

##  Model

The final AdaBoost model is saved as:
**`ada_bundle.pkl`**

Loaded in `app.py` for prediction.

---

##  How to Run Locally

###  Install dependencies

```
pip install -r requirements.txt
```

###  Run the Streamlit App

```
streamlit run app.py
```

---

##  App Screenshot
<img width="1919" height="962" alt="image" src="https://github.com/user-attachments/assets/360feab0-5251-4a1d-8cf3-07eb2b7b861b" />


##  Dataset

Dataset used: **Telco-Customer-Churn.csv**
Contains customer attributes like:

* Tenure
* Monthly charges
* Contract type
* Internet services
* Payment method
* Churn status

---

##  Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit

---

## Future Improvements

* Hyperparameter tuning
* Add more ML models (XGBoost, Catboost)
* Add SHAP explainability
* API deployment (FastAPI / Flask)

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

