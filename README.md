
# Travel Product Purchase Prediction

## 📌 Project Overview

Travel Product Purchase Prediction is a machine learning project that predicts whether a customer is likely to purchase a travel product based on their demographic details, travel behavior, and interaction history. The project uses a trained **Random Forest Classifier** wrapped inside a **scikit-learn Pipeline** and is deployed using **FastAPI**.

---

## 🎯 Problem Statement

Travel companies often struggle to identify which customers are most likely to purchase a travel product. This project helps solve that problem by analyzing customer data and predicting the probability of product purchase (`ProdTaken`).

---

## 🧠 Machine Learning Approach

* **Model**: Random Forest Classifier
* **Preprocessing**:

  * Missing value handling
  * Categorical encoding using OneHotEncoder
  * Feature scaling using StandardScaler
  * Feature engineering (e.g., `TotalVisits`)
* **Pipeline**: End-to-end pipeline combining preprocessing and model

---

## 📂 Dataset

The dataset contains customer information such as:

* Age, Gender, Marital Status
* Type of Contact
* Occupation, Designation
* Number of Trips
* Monthly Income
* Travel-related interaction details

**Target Variable:**

* `ProdTaken` (1 = Product Purchased, 0 = Not Purchased)

---

## 🏗️ Project Structure

```
Travel-Product-Purchase-Prediction/
│── main.py              # FastAPI backend
│── model.pkl            # Trained ML pipeline
│── Travel.csv           # Dataset (optional)
│── requirements.txt     # Project dependencies
│── README.md            # Project documentation
│── .gitignore           # Ignored files (venv, cache, etc.)
```

---

## 🚀 Backend API (FastAPI)

### Run the application

```bash
uvicorn main:app --reload
```

### Swagger UI

Access the API documentation at:

```
http://127.0.0.1:8000/docs
```

### Prediction Endpoint

**POST** `/predict`

The API accepts customer details in JSON format and returns whether the travel product is likely to be purchased.

---

## 📊 Model Output

* **ProdTaken = 1** → Customer is likely to purchase the travel product
* **ProdTaken = 0** → Customer is unlikely to purchase the travel product

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib

---

## 📦 Installation & Setup

```bash
# Create virtual environment
python -m venv venv

# Activate venv (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ✅ Key Highlights

* End-to-end ML pipeline (training → saving → inference)
* Proper preprocessing consistency between training and prediction
* Production-ready FastAPI backend
* Swagger UI for easy testing

---

## 👨‍💻 Author

**Subhajit Chakraborty**

---

## 📌 Future Improvements

* Add probability-based predictions
* Deploy on cloud platforms (Render / Railway / AWS)
* Add frontend UI
* Perform feature importance analysis

---

⭐ If you find this project useful, give it a star on GitHub!
