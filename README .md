# Credit Card Fraud Detection

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning. Since financial fraud poses a major challenge worldwide, this solution applies classification algorithms to distinguish between legitimate and fraudulent transactions.

The project uses anonymized credit card transaction data with severe class imbalance (fraudulent cases are rare compared to normal ones). The workflow includes **data preprocessing, feature scaling, model training, evaluation, and comparison of multiple algorithms** to identify the most effective fraud detection approach.

---

## 📂 Dataset
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Size**: 284,807 transactions  
- **Features**:  
  - `V1–V28`: Principal component analysis (PCA)-transformed features  
  - `Amount`: Transaction amount  
  - `Time`: Seconds elapsed between transactions  
  - `Class`: Target variable (0 = legitimate, 1 = fraud)  

---

## ⚙️ Workflow
1. **Data Preprocessing**  
   - Handle missing values (if any)  
   - Feature scaling with StandardScaler  
   - Address class imbalance using oversampling/undersampling or SMOTE  

2. **Model Training**  
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
   - Support Vector Machine (SVM)  

3. **Model Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix  
   - ROC-AUC curve  

4. **Results & Insights**  
   - Models compared on fraud detection performance  
   - Trade-offs between recall (catching frauds) and precision (avoiding false alarms)  

---

## 🚀 Results
- Logistic Regression → Baseline performance  
- Random Forest → Strong recall, interpretable  
- XGBoost → High precision and recall, best overall trade-off  
- SVM → Balanced but computationally expensive  

> 🏆 **XGBoost achieved the best fraud detection performance** with high recall and precision, making it suitable for financial fraud prevention systems.  

---

## 📦 Requirements
Create a virtual environment and install dependencies:  

```bash
pip install -r requirements.txt
```

**requirements.txt**  
```
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
```

---

## 📊 Visualization
- Fraud vs. Legitimate transaction distribution  
- Correlation heatmaps  
- ROC and Precision-Recall curves  

---

## 🔮 Future Improvements
- Deploy the model as an API for real-time fraud detection  
- Experiment with deep learning (e.g., Autoencoders, LSTMs)  
- Use cost-sensitive learning to minimize financial loss  

---

## 🤝 Acknowledgments
- Dataset by **Université Libre de Bruxelles (ULB)** via Kaggle  
- Open-source libraries: Scikit-learn, XGBoost, Matplotlib, Pandas  
