# 💳 Credit Card Fraud Detection Project

## 📋 Project Overview

This project implements a comprehensive machine learning pipeline to detect fraudulent credit card transactions. The system analyzes transaction patterns, customer behavior, and geographic data to identify potentially fraudulent activities with high accuracy.

## 🎯 Business Problem

Credit card fraud causes billions of dollars in losses annually and undermines customer trust in financial institutions. Traditional rule-based systems often have high false positive rates, leading to legitimate transactions being declined. This project aims to build an intelligent fraud detection system that:

- Accurately identifies fraudulent transactions
- Minimizes false positives (legitimate transactions flagged as fraud)
- Processes transactions in real-time
- Provides interpretable results for fraud analysts

## 📊 Dataset Description

**Source:** Synthetic credit card transactions dataset  
**Size:** 1852394 transactions  
**Time Period:** Transaction data spanning multiple months  
**Target Variable:** `is_fraud` (binary: 0 = legitimate, 1 = fraudulent)

### Key Features:
- **Transaction Details:** Amount, date/time, transaction number
- **Card Information:** Credit card number (anonymized)
- **Customer Demographics:** Name, age, gender, location
- **Merchant Information:** Merchant name, category, location
- **Geographic Data:** Customer and merchant coordinates

### Class Distribution:
- **Legitimate Transactions:** ~99.35%
- **Fraudulent Transactions:** ~0.65%
- **Challenge:** Highly imbalanced dataset requiring specialized techniques

## 🔧 Technical Implementation

### 1. Data Preprocessing Pipeline

#### **Missing Values Handling**
```python
# Geographic data: filled with median values
# Categorical data: filled with mode (most frequent)
# Target variable: missing values assumed as non-fraud
```

#### **Outlier Treatment**
```python
# Geographic outliers: Capped using IQR method
# Amount outliers: Removed only invalid values (negative/zero)
# Fraud cases: Preserved (they're the target, not outliers!)
```

#### **Data Type Corrections**
```python
# DateTime: trans_date_trans_time, dob
# Numeric: amt, lat, long, city_pop
# Categorical: merchant, category, state, etc.
```

### 2. Feature Engineering

#### **Time-Based Features**
- `hour`: Transaction hour (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `is_weekend`: Weekend transactions (higher risk)
- `is_night_transaction`: Transactions 10PM-6AM (higher risk)
- `is_business_hours`: Business hours transactions (9AM-5PM)
- `is_high_risk_hours`: Transactions 12AM-3AM (highest risk)

#### **Customer Demographics**
- `customer_age`: Calculated from date of birth
- `age_risk_category`: Age groups with different risk levels

#### **Geographic Features**
- `distance_km`: Distance between customer and merchant (Haversine formula)
- `is_far_transaction`: Transactions >100km from customer location
- `is_very_far_transaction`: Transactions >500km (very suspicious)

#### **Transaction Amount Features**
- `log_amount`: Log-transformed amount (handles skewness)
- `is_high_amount`: High-value transactions (>95th percentile)
- `is_low_amount`: Low-value transactions (<5th percentile)
- `is_round_amount`: Round amounts (e.g., $100.00)
- `amt_per_pop`: Amount relative to city population

#### **Velocity Features**
- `transactions_per_hour`: Number of transactions per card per hour
- `is_high_velocity`: Multiple transactions in short timeframe

#### **Risk Scoring**
- `category_risk_score`: Fraud rate by merchant category
- `is_high_risk_category`: Categories with high fraud rates

#### **Encoded Categorical Features**
- `merchant_encoded`: Label-encoded merchant names
- `category_encoded`: Label-encoded transaction categories
- `state_encoded`: Label-encoded states
- `job_encoded`: Label-encoded customer jobs
- `gender_encoded`: Binary encoding (0=Female, 1=Male)

### 3. Feature Selection

**Correlation-Based Selection:**
- Removed features with correlation > 0.9
- Avoided multicollinearity issues
- Retained ~30 most predictive features

### 4. Data Splitting & Scaling

**Train/Test Split:**
- 80% training, 20% testing
- Stratified split to maintain fraud ratio
- Random state = 42 for reproducibility

**Scaling:**
- **RobustScaler** (better for outliers in fraud data)
- Fitted on training data only
- Applied to both train and test sets

## 🎯 Model Performance Metrics

For imbalanced fraud detection, we focus on:

- **Precision:** How many predicted frauds are actually fraud (minimize false alarms)
- **Recall:** How many actual frauds we catch (minimize missed fraud)
- **F1-Score:** Balanced measure of precision and recall
- **ROC-AUC:** Overall model discrimination ability
- **Confusion Matrix:** Detailed breakdown of predictions

## 📁 Project Structure

```
fraud-detection/
│
├── data/
│   ├── fraudTrain.csv              # Training dataset
│   ├── fraudTest.csv               # Test dataset
│   └── preprocessed_fraud_data.csv # Cleaned dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Data preprocessing
│   ├── 03_feature_engineering.ipynb # Feature creation
│   ├── 04_modeling.ipynb           # Model training & evaluation
│   └── 05_results_analysis.ipynb   # Results interpretation
│
├── src/
│   ├── preprocessing.py            # Data preprocessing functions
│   ├── feature_engineering.py     # Feature creation functions
│   ├── models.py                   # Model definitions
│   └── evaluation.py              # Evaluation metrics
│
├── models/
│   ├── fraud_detection_model.pkl   # Trained model
│   ├── robust_scaler.pkl          # Feature scaler
│   └── label_encoders.pkl         # Categorical encoders
│
├── results/
│   ├── feature_importance.png     # Feature importance plot
│   ├── roc_curves.png             # ROC curves comparison
│   └── confusion_matrix.png       # Confusion matrix
│
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost  # optional, for XGBoost models
```

### Running the Project

1. **Data Preprocessing:**
```python
# Load and clean the data
df = load_data()
df_clean = preprocess_data(df)
```

2. **Feature Engineering:**
```python
# Create predictive features
df_features = create_features(df_clean)
```

3. **Model Training:**
```python
# Train and evaluate models
X_train, X_test, y_train, y_test = prepare_data(df_features)
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
```

## 📈 Key Results & Insights

### Most Important Features for Fraud Detection:
1. **Transaction Amount Features:** `log_amount`, `is_high_amount`
2. **Geographic Features:** `distance_km`, `is_far_transaction`
3. **Time Features:** `is_night_transaction`, `hour`
4. **Velocity Features:** `transactions_per_hour`, `is_high_velocity`
5. **Risk Scores:** `category_risk_score`, `merchant_risk_score`

### Business Insights:
- **Night transactions** (10PM-6AM) have 3x higher fraud rate
- **High-velocity** transactions (multiple per hour) are highly suspicious
- **Geographic distance** is a strong fraud indicator
- **Certain merchant categories** have significantly higher fraud rates
- **Round amounts** and **high amounts** are more likely to be fraudulent

## 🔮 Future Improvements

1. **Advanced Models:**
   - Deep learning approaches (Neural Networks)
   - Ensemble methods (Gradient Boosting)
   - Real-time streaming models

2. **Feature Enhancement:**
   - Time series features (spending patterns)
   - Network analysis (merchant connections)
   - Anomaly detection features

3. **Production Deployment:**
   - Real-time scoring API
   - Model monitoring and drift detection
   - Automated retraining pipeline

4. **Business Integration:**
   - Risk-based authentication
   - Dynamic transaction limits
   - Fraud analyst dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset providers for the synthetic credit card data
- Scikit-learn community for machine learning tools
- Open source contributors to the fraud detection community



---

**⚠️ Disclaimer:** This project uses synthetic data for educational purposes. In production fraud detection systems, additional privacy and security measures are required.
