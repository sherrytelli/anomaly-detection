# Anomaly Detection

An anomaly detection system built using the Isolation Forest algorithm from scikit-learn. This project generates a synthetic classification dataset, preprocesses it, trains an Isolation Forest model, and evaluates its performance in detecting anomalies.

## Overview

This project demonstrates a complete anomaly detection pipeline:

1. **Dataset Generation** — Creates a synthetic imbalanced dataset (950 normal samples, 50 anomalies) with 6 features using `sklearn.datasets.make_blobs`.
2. **Data Preprocessing** — Splits data into train/test sets (80/20) and applies `StandardScaler` normalization.
3. **Model Training** — Trains an `IsolationForest` model with `contamination=0.05` and `random_state=24`.
4. **Model Persistence** — Saves the trained model and scaler as pickle files (`iso_forest.pkl`, `scaler.pkl`).
5. **Prediction & Evaluation** — Loads the saved model, makes predictions on the test set, and evaluates using a classification report.

### Model Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Not Anomaly) | 0.98 | 0.99 | 0.99 | 189 |
| 1 (Anomaly) | 0.89 | 0.73 | 0.80 | 11 |
| **Accuracy** | | | **0.98** | 200 |

## Project Structure

```
.
├── development.ipynb    # Jupyter notebook with the full pipeline
├── iso_forest.pkl       # Saved trained Isolation Forest model
├── scaler.pkl           # Saved fitted StandardScaler
└── README.md            # Project documentation
```

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sherrytelli/anomaly-detection
cd anomaly-detection
```

### 2. Create a Virtual Environment

**Linux / macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist, install the packages manually:

```bash
pip install scikit-learn matplotlib jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `development.ipynb` to run the full pipeline interactively.

## How It Works

### Step 1: Generate Dataset

A synthetic dataset with 1000 samples and 6 features is generated using `make_blobs`:

- **Normal samples (label 0):** 950
- **Anomaly samples (label 1):** 50
- **Features:** 6
- **Random state:** 24 (for reproducibility)

### Step 2: Preprocess Data

- Split into training (800 samples) and testing (200 samples) sets using an 80/20 split.
- Scale the training features using `StandardScaler` and save the fitted scaler.

### Step 3: Train Isolation Forest

- Train an `IsolationForest` with 100 estimators, `contamination=0.05`, and `random_state=24`.
- Save the trained model as `iso_forest.pkl`.

### Step 4: Load Model & Predict

- Load the saved scaler and model from pickle files.
- Transform test data using the scaler.
- Make predictions: `-1` indicates anomaly, `1` indicates normal.
- Convert predictions to `0` (normal) and `1` (anomaly) format.

### Step 5: Evaluate

- Generate a classification report with precision, recall, and F1-score for both classes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
