# Machine-Learning-Assignment-2

## Problem Statement
Design and assess a **multiclass classification** system to predict **Cancer_Type** (Lung, Breast, Colon, Prostate, Skin) from simulated demographic, lifestyle, environmental, and genetic risk factors, with careful handling of class imbalance and evaluation using **F1-score, precision, recall, accuracy, AUC, and MCC**. Additionally, develop and deploy an **interactive Streamlit application** that allows users to upload a dataset, select a classification model, and visualize prediction results for population-level analysis.

## Dataset Description
This dataset contains medically-informed simulated cancer risk profiles created for machine learning, research, and educational analysis, with all records representing synthetic patients and no real individuals. It includes 2,000 records with 21 encoded features spanning demographics, lifestyle, environmental, genetic, and medical factors, along with engineered fields such as an Overall_Risk_Score and derived Risk_Level. The primary target, Cancer_Type (Lung, Breast, Colon, Prostate, Skin), is well-suited for multiclass classification using balanced evaluation metrics like macro-F1, accuracy, and confusion matrices, while Risk_Level (Low, Medium, High) supports optional risk stratification. The dataset enables users to explore risk distributions and correlations across lifestyle factors, build visual dashboards for population-level cancer risk, and practice class-imbalance handling and model interpretability, while preserving realistic constraints (e.g., Prostate cancer only in males, rare male Breast cases) and a risk score that aligns directionally with known exposure intensities. It is intended strictly for research and educational purposes, not for medical or diagnostic use.

## Models used
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :-------------: | :--------: | :---: | :---------: | :------: | :--: | :---: |
| Logistic Regression | | | | | |
| Decision Tree | | | | | |
| KNN | | | | | |
| Naive Bayes | | | | | |
| Random Forest (Ensemble) | | | | | |
| XGBoost (Ensemble) | | | | | |

| ML Model Name | Observation about model performance |
| :-------------: | :-----------------------------------: |
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Random Forest (Ensemble) | |
| XGBoost (Ensemble) | |
