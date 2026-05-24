# Restaurant Rating Prediction

Machine Learning project for predicting restaurant ratings based on restaurant information, customer behavior, pricing, delivery details, and other features.

---

# Project Overview

The goal of this project is to build a supervised machine learning model that predicts restaurant ratings.

This project includes:

- Data preprocessing
- Feature engineering
- Exploratory data analysis (EDA)
- Model training
- Model evaluation
- Error handling and logging
- Modular programming structure

---

# Dataset Features

Example features used in the project:

- Restaurant name
- City
- Address
- Locality
- Locality Verbose
- Longitude
- Latitude
- Cuisines
- Average Cost for two
- Has Table booking
- Has Online delivery
- Is delivering now
- Switch to order menu
- Price range
- Rating color
- Rating text
- Votes

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Shap

---

# Project Structure

```bash
project/
│
├── data/
│   ├── raw/
│   └── processed/
│ 			├── baseline/
├── logs/
│		├── data_loader.log
│		├── data_saver.log
│		├── feature_engineer.log
│		├── feature_selector.log
│		├── feasture_transform.log
│		├── preprocessor.log
│		└── trainer.log
├── models/
│   ├── baseline/
│   └── improved/
│
├── myenv/
│
├── notebooks/
│		├── baseline_model.ipynb
│		├── data-loader.ipynb
│		├── eda.ipynb
│		├── improvement_model.ipynb
│		└── preprocessing.ipynb
│
├── results/
│   ├── baseline/
│   └── improved/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_saver.py
│   ├── feature_engineering.py
│   ├── feature_selector.py
│   ├── feature_transform.py
│   ├── logger.py
│   ├── preprocess.py
│   └── trainer.py
│
├── .gitignore
│
├── README.md
│
├── requirements.txt
│
└── setup.py
```

---

# Machine Learning Workflow

1. Data Cleaning
2. Train-Test Split
3. Handling Missing Values
4. Encoding Categorical Features
5. Feature Engineering
6. Feature transforming
7. Feature Selection
8. Model Training
9. Model Evaluation
10. Shap value

---

# Installation

Clone the repository:

```bash
git clone https://github.com/agus787-dev/restaurant_rating.git
```

Go to project directory:

```bash
cd restaurant_rating
```

Create virtual environment:

```bash
python3 -m venv myenv
```

Activate virtual environment:

Mac/Linux:

```bash
source myenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

1. Download dataset from Google Drive : https://drive.google.com/drive/folders/1PHbGMq1Se9EsHm2UBCOIYi1o1ngYnTxP
2. `data/raw/`
3. Fayl name: `restaurant_dataset.csv`

# Model Evaluation Metrics

Example metrics:

- R2
- RMSE
- MAE

---

# Author

agus787-dev
