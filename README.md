🩺 Diabetes Prediction Model
Kaggle Playground Series – Season 5, Episode 12 (S5E12)

This project builds a machine-learning classification model to predict the probability of diagnosed_diabetes using the Kaggle Playground Series – S5E12 dataset.
The target leaderboard performance goal was an AUC-ROC ≥ 0.78940, with a full end-to-end pipeline including EDA, preprocessing, model selection, hyperparameter tuning, and submission generation.

📁 Dataset

The dataset originates from a synthetic deep learning model trained on the Diabetes Health Indicators Dataset. It includes:

Demographic variables

Lifestyle factors

Medical history

Clinical measurements

Target variable:
diagnosed_diabetes — a binary indicator (0/1) for diabetes diagnosis.

Files used:

train.csv

test.csv

sample_submission.csv

🔧 Methodology
1. Data Download & Setup

Initially attempted using Kaggle API, but encountered:

401 Client Error: Unauthorized

Invalid/missing kaggle.json

Competition rules not accepted

Solution: Manual upload of dataset files into Google Colab.

2. Exploratory Data Analysis (EDA)

Performed the following:

✔ Basic Integrity Checks

Validated data types

Identified minimal missing values

✔ Visualizations

Histograms (numerical features)

Count plots (categorical features)

Boxplots comparing numerical features & target

Count plots with hue = diagnosed_diabetes

Correlation heatmap

✔ Key Findings

Strong associations with diabetes were found in:

age, bmi, waist_to_hip_ratio

systolic_bp, diastolic_bp

cholesterol_total, ldl_cholesterol, triglycerides

family_history_diabetes, hypertension_history, cardiovascular_history

Negative correlation:

hdl_cholesterol

Potential multicollinearity was also detected among numerical variables.

🧹 Data Preprocessing
✔ Missing Values

Rows with isolated missing entries (1 per column) were dropped in both train & test sets.

✔ Type Conversion

Binary fields converted to:

Boolean → Integer for modeling
(family_history_diabetes, hypertension_history, cardiovascular_history, diagnosed_diabetes)

✔ Feature Engineering

Five new features were added:

BMI_Age_Interaction = bmi * age

WH_Ratio_Age_Interaction = waist_to_hip_ratio * age

BP_Interaction = systolic_bp * diastolic_bp

Cholesterol_Ratio = ldl_cholesterol / hdl_cholesterol
(Safe division handling zero/NaN)

History_Sum = sum of all family/medical history binary indicators

✔ Scaling & Encoding

Using ColumnTransformer:

StandardScaler → numerical variables

OneHotEncoder → categorical variables

id column removed before transformation

🤖 Model Training & Selection
Baseline Model

Logistic Regression

Accuracy: 0.6623

AUC-ROC: 0.6904

Advanced Models Tested
Model	Accuracy	AUC-ROC
RandomForestClassifier	0.6562	0.6829
XGBClassifier	0.6722	0.7053

XGBoost performed best, becoming the primary candidate for optimization.

⚙️ Hyperparameter Tuning

Approach: GridSearchCV (3-fold cross-validation)
Parameters tuned:

n_estimators

max_depth

learning_rate

subsample

colsample_bytree

Best parameters identified:

{
    'colsample_bytree': 0.7,
    'learning_rate': 0.05,
    'max_depth': 5,
    'n_estimators': 300,
    'subsample': 0.9
}

📈 Optimized Performance

Cross-validated AUC-ROC: 0.7164

📤 Prediction & Submission
✔ Final Model

Retrained optimized XGBoost on all processed training data.

✔ Predictions

Generated prediction probabilities for all test rows.

✔ Submission File (submission2.csv)

Contains 300,000 rows as required

Rows missing from processed test set filled with default probability 0.5

Columns:

id

diagnosed_diabetes (probability)

⚠ Kaggle Public Score

0.57840 — significantly lower than local validation results.
Likely causes:

Validation–test distribution mismatch

Submission misalignment

Differences in feature preprocessing

Issues with missing-row probability padding

🚧 Next Steps / Future Improvements
✔ Kaggle Submission Fixes

Resolve 401 authorization errors

Ensure:

API credentials correctly configured

Competition rules accepted

Proper dataset file structure

✔ Modeling Improvements

Try LightGBM, CatBoost, or small neural nets

More advanced feature engineering:

Polynomial features

Clinical domain-based ratios & interactions

Apply robust cross-validation:

Stratified K-Fold (recommended)

Ensemble & stacking approaches

Perform post-hoc error analysis to find major failure patterns

📦 Repository Structure
├── train.csv
├── test.csv
├── sample_submission.csv
├── diabetes_prediction.ipynb
├── submission2.csv
└── README.md

🏁 Conclusion

This project implements a complete Kaggle workflow—from EDA through modeling and submission. While local performance exceeded baseline expectations, public leaderboard results revealed opportunities for improved validation and preprocessing consistency.
