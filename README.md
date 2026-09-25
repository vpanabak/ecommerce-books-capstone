# Module 2: Analytics & Predictive Modeling Pipeline (`/analytics`)

This module documents the complete end-to-end data science workflow on the Titanic dataset, spanning data profiling, missing-value strategy enforcement, bivariate/multivariate EDA, leak-free Scikit-Learn preprocessing, model training, class imbalance evaluation, hyperparameter tuning, regression analysis, and end-to-end pipeline deployment.

---

## 1. Data Profiling & Threshold-Based Missing Value Strategy

### Measured Missing Value Rates (Total Rows: 891)
* **`deck`**: 688 missing (**77.10%**)
* **`age`**: 177 missing (**19.87%**)
* **`embarked` / `embark_town`**: 2 missing (**0.22%**)

### Strategy Decisions & Justification
1. **`embarked` & `embark_town` (<5% missing):** Dropped the 2 affected rows. Since the missing proportion is negligible (<0.5%), dropping rows introduces no statistical bias.
2. **`age` (5%–30% missing):** Applied **median imputation**. Replacing missing age values with the median age (28.0) preserves sample size without distorting distribution centrality.
3. **`deck` (>30% missing):** Retained the column but explicitly encoded missing entries as a distinct category (**"Missing"**). Dropping the column entirely would lose crucial spatial location signal (e.g., upper deck proximity to lifeboats), whereas imputing would introduce high statistical noise.

---

## 2. Univariate & Bivariate Findings

### Outliers & Distribution Skewness
* **Age Outliers (IQR Rule):** 9 outliers detected beyond $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$.
* **Fare Outliers (IQR Rule):** 116 outliers detected.
* **Fare Distribution Skewness:**
  * **Mean:** £32.20 | **Median:** £14.45 | **Mode:** £8.05
  * **Conclusion:** Since $\text{Mean} > \text{Median} > \text{Mode}$, the `fare` distribution is **strongly right-skewed** with a long right tail driven by luxury first-class suites.

### Survival Rate Breakdown
* **By Sex:** Female = **74.20%** | Male = **18.89%**
* **By Pclass:** 1st Class = **62.96%** | 2nd Class = **47.28%** | 3rd Class = **24.24%**
* **By Sex & Pclass Combined:**
  * 1st Class Female: **96.81%** | 1st Class Male: **36.89%**
  * 2nd Class Female: **92.11%** | 2nd Class Male: **15.74%**
  * 3rd Class Female: **50.00%** | 3rd Class Male: **13.54%**

### Correlation Heatmap (Top 2 Off-Diagonal Pairs)
Restricted strictly to `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare` (`adult_male` and `alone` excluded):
1. **`pclass` and `fare` ($\vert{}r\vert{} = 0.55$):** Strong inverse correlation; higher numerical classes (3rd class) pay significantly lower fares.
2. **`sibsp` and `parch` ($\vert{}r\vert{} = 0.41$):** Moderate positive correlation; passengers traveling with siblings/spouses were also more likely to travel with parents/children (family groups).

---

## 3. Class Imbalance Comparison
Evaluated on Random Forest across training variants:
* **Baseline (No Handling):** Precision: 0.7759 | Recall: 0.7143 | F1: 0.7438
* **Class Weight `'balanced'`:** Precision: 0.7627 | Recall: 0.7143 | F1: 0.7377
* **SMOTE (Train fold ONLY):** Precision: 0.7419 | Recall: 0.7302 | F1: 0.7360
* **Conclusion:** The dataset exhibits a moderate class balance (~38% positive class). Baseline and class-weighted models achieved superior precision and F1 scores. SMOTE marginally improved recall but slightly degraded precision due to synthetic noise generation.

---

## 4. Hyperparameter Tuning & Regression Side-Task

### GridSearchCV Best Parameters & OOB Score
* **Best Parameters:** `{'classifier__max_depth': 5, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 100}`
* **Out-Of-Bag (OOB) Score:** **0.8174**

### Fare Regression Results & Residual Analysis
* **MAE:** £18.84 | **RMSE:** £34.22 | **$R^2$:** 0.3842 | **Adjusted $R^2$:** 0.3541
* **Heteroscedasticity Conclusion:** The residual plot displays clear **heteroscedasticity**. As predicted fares increase, the spread/variance of residuals expands fan-like, reflecting high variance in upper-class ticket pricing.

---

## 5. Model Comparison Tables & Deployment Recommendation

### Classification Metrics Comparison
| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8034 | 0.7586 | 0.6984 | 0.7273 | 0.8530 |
| **Decision Tree** | 0.7921 | 0.7843 | 0.6349 | 0.7018 | 0.8351 |
| **Random Forest (Tuned)** | **0.8202** | **0.7857** | **0.6984** | **0.7395** | **0.8655** |

### Regression Metrics Comparison
| Model | MAE | RMSE | $R^2$ | Adjusted $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (Fare)** | 18.84 | 34.22 | 0.3842 | 0.3541 |

### Final Deployment Recommendation
I recommend deploying the **Tuned Random Forest Classifier**. It achieves the highest overall **Accuracy (82.02%)**, **F1 Score (0.7395)**, and **AUC-ROC (0.8655)** among all candidate models. Its ensemble architecture handles non-linear interactions between class, sex, and fare effectively without overfitting.