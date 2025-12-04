Diabetes Prediction Model for Kaggle Playground Series - S5E12
Project Overview
This project aims to build a classification model to predict the probability of diagnosed_diabetes using the 'playground-series-s5e12' dataset from Kaggle. The goal is to generate a submission file in the specified format, targeting a public score of at least 0.78940 (AUC-ROC).

Dataset
The dataset was generated from a deep learning model trained on the Diabetes Health Indicators Dataset. It includes various health indicators, demographic information, and lifestyle factors. The target variable is diagnosed_diabetes, a binary flag indicating whether an individual has been diagnosed with diabetes.

Methodology
1. Data Download and Initial Setup
Tool: Kaggle API (kaggle-api).
Challenges: Initial attempts faced 401 Client Error: Unauthorized due to missing or invalid kaggle.json credentials and not having accepted the competition rules. This was resolved by manually uploading the data to the Colab environment.
Files: train.csv, test.csv, sample_submission.csv were loaded into pandas DataFrames.
2. Data Exploration and Analysis
Initial Checks: Verified data types and identified minor missing values.
Visualizations: Histograms for numerical features, count plots for categorical features, box plots to show numerical feature distributions against the target, and count plots with hue for categorical features against the target.
Correlation Analysis: A heatmap of the correlation matrix for numerical features and the target variable (diagnosed_diabetes) was generated to understand feature relationships.
Key Findings:
age, bmi, waist_to_hip_ratio, systolic_bp, diastolic_bp, cholesterol_total, ldl_cholesterol, triglycerides, family_history_diabetes, hypertension_history, and cardiovascular_history showed strong associations with diagnosed_diabetes.
hdl_cholesterol exhibited a negative correlation.
Identified potential multicollinearity among some numerical features.
3. Data Preprocessing
Missing Values: Rows with minimal missing values (single instances per column) were dropped from both train_df and test_df.
Feature Type Conversion: Binary features (family_history_diabetes, hypertension_history, cardiovascular_history, diagnosed_diabetes) were converted to boolean types, and later to integers for numerical processing.
Feature Engineering: Five new interaction features were created:
BMI_Age_Interaction (bmi * age)
WH_Ratio_Age_Interaction (waist_to_hip_ratio * age)
BP_Interaction (systolic_bp * diastolic_bp)
Cholesterol_Ratio (ldl_cholesterol / hdl_cholesterol, with robust handling for division by zero and NaNs)
History_Sum (family_history_diabetes + hypertension_history + cardiovascular_history)
Scaling and Encoding: A ColumnTransformer was used to apply StandardScaler to numerical features and OneHotEncoder to categorical features, ensuring all features were properly scaled and encoded for model training. The 'id' column was separated before transformation.
4. Model Training and Selection
Initial Model: Logistic Regression was used as a baseline model, achieving an Accuracy of 0.6623 and an AUC-ROC of 0.6904 on a validation set.
Model Experimentation: RandomForestClassifier and XGBClassifier were introduced:
RandomForestClassifier: Accuracy 0.6562, AUC-ROC 0.6829.
XGBClassifier: Accuracy 0.6722, AUC-ROC 0.7053.
Selection: XGBoost showed superior performance and was selected for further optimization.
5. Hyperparameter Tuning
Technique: GridSearchCV with 3-fold cross-validation was applied to the XGBClassifier.
Parameters Tuned: n_estimators, max_depth, learning_rate, subsample, colsample_bytree.
Best Parameters Found: {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.9}.
Optimized Performance: The optimized XGBoost model achieved a cross-validated AUC-ROC score of 0.7164, demonstrating an improvement.
6. Prediction and Submission File Generation
Final Model Training: The optimized XGBoost model was retrained on the entire preprocessed training dataset (X_processed_df and y).
Predictions: Probability predictions for diagnosed_diabetes were generated on the X_test_processed_df.
Submission File: A submission2.csv file was created, combining the test_ids (from sample_submission_df) with the final_test_predictions_proba. Predictions for IDs not present in our processed test set were filled with a default probability of 0.5 to meet the required 300,000 rows.
Performance
Logistic Regression (Validation AUC-ROC): 0.6904
XGBoost (Validation AUC-ROC, initial): 0.7053
XGBoost (Cross-validated AUC-ROC, optimized): 0.7164
Kaggle Public Score: Core: 0.57840 (This score is lower than expected based on validation metrics, suggesting differences between the local validation set and the public test set, or issues with submission process previously.)
Next Steps / Future Work
Re-evaluate Submission Process: Address persistent 401 Client Error: Unauthorized during Kaggle submission. Confirm acceptance of competition rules and valid API credentials.
Further Feature Engineering: Explore more complex interaction terms, polynomial features, or domain-specific features.
Advanced Models: Experiment with other ensemble methods like LightGBM, CatBoost, or deep learning models.
Stacking/Ensembling: Combine predictions from multiple models to potentially improve overall performance.
Cross-Validation Strategy: Use more robust cross-validation techniques (e.g., stratified k-fold) for local validation to better mimic competition evaluation.
Error Analysis: Investigate misclassified instances to understand model weaknesses and guide further improvements.chniques, models, performance gains, and submission file status must be provided.
