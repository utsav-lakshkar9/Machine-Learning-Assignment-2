# Multiclass Cancer Risk Classification from Simulated Patient Data

## Problem Statement
Design and assess a **multiclass classification** system to predict **Cancer_Type** (Lung, Breast, Colon, Prostate, Skin) from simulated demographic, lifestyle, environmental, and genetic risk factors, with careful handling of class imbalance and evaluation using **F1-score, precision, recall, accuracy, AUC, and MCC**. Additionally, develop and deploy an **interactive Streamlit application** that allows users to upload a dataset, select a classification model, and visualize prediction results for population-level analysis.

## Dataset Description
This dataset contains medically-informed simulated cancer risk profiles created for machine learning, research, and educational analysis, with all records representing synthetic patients and no real individuals. It includes 2,000 records with 21 encoded features spanning demographics, lifestyle, environmental, genetic, and medical factors, along with engineered fields such as an Overall_Risk_Score and derived Risk_Level. The primary target, Cancer_Type (Lung, Breast, Colon, Prostate, Skin), is well-suited for multiclass classification using balanced evaluation metrics like macro-F1, accuracy, and confusion matrices, while Risk_Level (Low, Medium, High) supports optional risk stratification. The dataset enables users to explore risk distributions and correlations across lifestyle factors, build visual dashboards for population-level cancer risk, and practice class-imbalance handling and model interpretability, while preserving realistic constraints (e.g., Prostate cancer only in males, rare male Breast cases) and a risk score that aligns directionally with known exposure intensities. It is intended strictly for research and educational purposes, not for medical or diagnostic use.

## Models used
| ML Model Name | Accuracy | Precision | Recall | F1 | MCC | AUC |
| :-------------: | :--------: | :---: | :---------: | :------: | :--: | :---: |
| Logistic Regression | 0.894 |  0.729 |  0.713 |  0.717 |  0.664 |  0.938 |
| Decision Tree | 0.853 |  0.617 |  0.609 |  0.611 |  0.535 |  0.758 |
| KNN | 0.871 |  0.666 |  0.645 |  0.645 |  0.590 |  0.866 |
| Naive Bayes | 0.795 |  0.612 |  0.494 |  0.426 |  0.422 |  0.823 |
| Random Forest (Ensemble) | 0.893 |  0.721 |  0.705 |  0.707 |  0.660 |  0.930 |
| XGBoost (Ensemble) | 0.897 |  0.742 |  0.720 |  0.726 |  0.673 |  0.939 |

| ML Model Name | Observation about model performance |
| :-------------: | :-----------------------------------: |
| Logistic Regression | Strong and well-balanced baseline model with high accuracy and excellent AUC |
| Decision Tree | Simpler model with comparatively lower accuracy and weakest discrimination ability |
| KNN | Moderate performance with balanced precision and recall but inferior to ensemble methods |
| Naive Bayes | Lowest overall performance, likely due to unrealistic feature independence assumptions |
| Random Forest (Ensemble) | Robust ensemble model providing stable and reliable performance across metrics |
| XGBoost (Ensemble) | Best-performing model overall with superior accuracy, F1-score, and class separation |
