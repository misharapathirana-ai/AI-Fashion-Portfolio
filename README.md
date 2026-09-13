# Fashion AI - Fashion Product Category Classification

## Project Overview

Fashion AI is a machine learning project that analyzes fashion product data and predicts the category of a fashion product based on its product title, brand, and price.

The project combines exploratory data analysis, machine learning, and an interactive Streamlit web application.

---

## Project Objectives

The main objectives of this project are:

- Analyze a real-world fashion product dataset
- Explore product prices, ratings, brands, and categories
- Identify patterns and relationships within fashion products
- Develop a machine learning model for product category classification
- Evaluate the performance of different machine learning approaches
- Build an interactive web application for real-time predictions

---

## Live Demo

Try the deployed Fashion AI application:

https://ai-fashion-category-classifier.streamlit.app/

---

## Dataset

The dataset contains information about fashion products, including:

- Product ID
- Brand
- Product Title
- Price
- Category
- Rating
- Product Image URL
- Product URL

The original dataset contains more than 13,000 product records across 11 fashion-related categories.

---

## Exploratory Data Analysis

The dataset was analyzed to investigate:

- Missing values
- Duplicate products
- Product categories
- Brand popularity
- Price distributions
- Ratings
- Average price by category
- Average rating by category
- Relationship between price and rating
- Brand-level performance

### Key Findings

Some notable findings from the analysis include:

- Watches had the highest average product price.
- Made for Amazon products had the lowest average price.
- Most-Loved Fashion products had the highest average rating.
- Product price and rating showed only a weak-to-moderate relationship.
- Brand information provided a significant contribution to category prediction.
- Marketing-oriented categories such as Most-Loved Fashion, New season, and Outlet were more difficult for the model to distinguish.

---

## Machine Learning

A machine learning classification model was developed to predict the category of a fashion product.

### Input Features

The model uses:

- Product Title
- Brand
- Price

### Preprocessing

Different preprocessing techniques were applied to the input features:

- TF-IDF vectorization for product titles
- One-Hot Encoding for brand
- Standardization for price

### Model

The final model uses:

**Logistic Regression**

with class balancing and hyperparameter tuning.

---

## Model Performance

The final model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 78.83% |
| Macro F1 Score | 0.7882 |

The model was evaluated using a test set containing 2,064 products.

---

## Supported Categories

The model predicts products across 11 categories:

1. Accessories
2. Handbags
3. Luggage
4. Made for Amazon
5. Most-Loved Fashion
6. New season
7. Outlet
8. Shoes
9. Sportswear
10. Wallets
11. Watches

---

## Streamlit Application

The trained model is integrated into an interactive Streamlit web application.

Users can enter:

- Product title
- Brand
- Price

The application then predicts the most likely fashion product category and displays the model's prediction confidence.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib
- Jupyter Notebook

---

## Project Structure

```text
01-fashion-data-analytics/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── products.csv
│   │
│   └── processed/
│
├── notebooks/
│   ├── fashion_data_analysis.ipynb
│   └── fashion_category_classifier.pkl
│
├── .gitignore
├── requirements.txt
└── README.md
