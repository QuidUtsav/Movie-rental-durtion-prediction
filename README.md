# Movie-rental-durtion-prediction
Machine Learning Regression Project  This project builds a machine learning model to predict the number of days a customer rents a DVD based on metadata from a rental database. The task involves end-to-end data preprocessing, feature engineering, model selection, and evaluation using regression techniques.
📁 Dataset

The dataset comes from rental_info.csv and contains information on DVD rentals, including:

Rental and return dates

Customer and film information

Special features

Category, replacement costs, and more

The target variable is:

🎯 rental_length_days

Calculated as:

return_date - rental_date


and represents how long (in days) a DVD was rented.

🛠️ Data Preprocessing & Feature Engineering

Key preprocessing steps include:

✔️ 1. Loading Data

Using pandas to read rental_info.csv.

✔️ 2. Creating the Target Column

rental_length_days is computed from the date columns using Pandas datetime operations.

✔️ 3. Dummy Variable Engineering

The column special_features was transformed into two binary features:

deleted_scenes → 1 if feature includes "Deleted Scenes"

behind_the_scenes → 1 if feature includes "Behind the Scenes"

✔️ 4. Feature Selection

A DataFrame X was constructed with only non-leaking, predictive features.
The target y is the newly created rental_length_days.

🔀 Train/Test Split

The dataset was split into:

X_train, X_test

y_train, y_test

with:

20% test size

random_state = 9 (for reproducibility)

🤖 Model Selection & Tuning

Multiple regression models were tested.
A randomized/grid search tuning process was run locally to avoid platform limitations.

The final recommended model is:

⭐ best_model

A regression model tuned with the best-performing hyperparameters, achieving:

📉 Test MSE: 1.97

Well below the project requirement of MSE < 3.

This model is exported as:

best_model
best_mse

📦 Tech Stack

Python

Pandas

NumPy

Scikit-learn

🧠 What This Project Demonstrates

End-to-end ML workflow

Feature engineering from raw real-world data

Avoiding data leakage

Model tuning & validation

Reproducible experiment setup

Interpretation and evaluation of regression models
