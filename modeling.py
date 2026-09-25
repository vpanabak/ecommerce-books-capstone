import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
import joblib

print("=== PART B: PREDICTIVE MODELING & PIPELINE ===")

# Read the single committed dataset (or cleaned version)
df = pd.read_csv('titanic_cleaned.csv')

# Feature selection
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

X = df[features]
y = df[target]

# Task 7: Stratified Train/Test Split
# Stratification maintains the ~38% survival ratio across both train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Task 8: Leak-free Preprocessing using ColumnTransformer
numeric_features = ['age', 'fare', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked', 'pclass']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Task 9 & 10: Train & Evaluate 3 Classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = []

for name, clf in classifiers.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
        'AUC-ROC': round(auc, 4)
    })

df_clf_results = pd.DataFrame(results)

# Render Decision Tree plot
dt_pipe = Pipeline([('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(max_depth=3, random_state=42))])
dt_pipe.fit(X_train, y_train)
plt.figure(figsize=(12, 8))
plot_tree(
    dt_pipe.named_steps['classifier'],
    filled=True,
    feature_names=dt_pipe.named_steps['preprocessor'].get_feature_names_out(),
    class_names=['Died', 'Survived']
)
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree.png")
print("Decision tree saved to 'decision_tree.png'.")

# Task 11: Class Imbalance Comparison
print("\n--- Imbalance Handling Comparison ---")

# Apply preprocessor to training data for manual SMOTE demonstration
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# (a) Baseline
rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train_proc, y_train)
y_pred_base = rf_base.predict(X_test_proc)

# (b) Class Weight Balanced
rf_bal = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_bal.fit(X_train_proc, y_train)
y_pred_bal = rf_bal.predict(X_test_proc)

# (c) SMOTE (Train fold ONLY)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)
rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test_proc)

imbalance_summary = pd.DataFrame([
    {'Variant': 'Baseline (No handling)', 'Precision': precision_score(y_test, y_pred_base), 'Recall': recall_score(y_test, y_pred_base), 'F1': f1_score(y_test, y_pred_base)},
    {'Variant': "Class Weight 'balanced'", 'Precision': precision_score(y_test, y_pred_bal), 'Recall': recall_score(y_test, y_pred_bal), 'F1': f1_score(y_test, y_pred_bal)},
    {'Variant': 'SMOTE (Train fold only)', 'Precision': precision_score(y_test, y_pred_smote), 'Recall': recall_score(y_test, y_pred_smote), 'F1': f1_score(y_test, y_pred_smote)}
])
print(imbalance_summary)

# Task 12: Hyperparameter Tuning with GridSearchCV & OOB Score
print("\n--- Hyperparameter Tuning (Random Forest) ---")
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, None],
    'classifier__max_features': ['sqrt', 'log2']
}

rf_oob = RandomForestClassifier(oob_score=True, random_state=42)
tuning_pipe = Pipeline([('preprocessor', preprocessor), ('classifier', rf_oob)])

grid_search = GridSearchCV(tuning_pipe, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_rf = best_model.named_steps['classifier']

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Out-Of-Bag (OOB) Score: {best_rf.oob_score_:.4f}")

# Task 13: Regression Side-Task (Predicting Fare)
print("\n--- Regression Side-Task (Predicting Fare) ---")
reg_features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked', 'survived']
X_reg = df[reg_features]
y_reg = df['fare']

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_cat_features = ['sex', 'embarked', 'pclass', 'survived']
reg_num_features = ['age', 'sibsp', 'parch']

reg_preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), reg_num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), reg_cat_features)
])

reg_pipe = Pipeline([('preprocessor', reg_preprocessor), ('regressor', LinearRegression())])
reg_pipe.fit(X_tr_r, y_tr_r)

y_pred_reg = reg_pipe.predict(X_te_r)

mae = mean_absolute_error(y_te_r, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_te_r, y_pred_reg))
r2 = r2_score(y_te_r, y_pred_reg)
n = len(y_te_r)
p = X_te_r.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}, Adj R2: {adj_r2:.4f}")

# Residual Plot
residuals = y_te_r - y_pred_reg
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_reg, residuals, alpha=0.5, color='coral')
plt.axhline(0, color='black', linestyle='--')
plt.title('Fare Regression Residual Plot')
plt.xlabel('Predicted Fare')
plt.ylabel('Residuals')
plt.tight_layout()
plt.savefig('fare_residuals.png')
print("Residual plot saved to 'fare_residuals.png'.")

# Task 14: Final Model Comparison Table & Selection
print("\n=== FINAL COMPARISON TABLES ===")
print("Classifier Metrics:")
print(df_clf_results.to_string(index=False))

df_reg_results = pd.DataFrame([{
    'Model': 'Linear Regression (Fare)',
    'MAE': round(mae, 2),
    'RMSE': round(rmse, 2),
    'R2': round(r2, 4),
    'Adj R2': round(adj_r2, 4)
}])
print("\nRegression Metrics:")
print(df_reg_results.to_string(index=False))

# Task 15: Save Complete End-to-End Pipeline
pipeline_filename = "full_titanic_pipeline.pkl"
joblib.dump(best_model, pipeline_filename)
print(f"\nSaved complete pipeline to '{pipeline_filename}'.")

# Reload verification on raw, unpreprocessed sample
reloaded_pipeline = joblib.load(pipeline_filename)

raw_sample = pd.DataFrame([{
    'pclass': 1,
    'sex': 'female',
    'age': 29.0,
    'sibsp': 0,
    'parch': 0,
    'fare': 211.3375,
    'embarked': 'S'
}])

sample_pred = reloaded_pipeline.predict(raw_sample)
sample_prob = reloaded_pipeline.predict_proba(raw_sample)[:, 1]

print("\n--- Pipeline Reload Test ---")
print("Raw Unpreprocessed Input:")
print(raw_sample)
print(f"Predicted Class: {sample_pred[0]} (1 = Survived, 0 = Died)")
print(f"Survival Probability: {sample_prob[0]:.4f}")
print("✅ PIPELINE RELOAD & INFERENCE VERIFIED SUCCESSFULLY!")