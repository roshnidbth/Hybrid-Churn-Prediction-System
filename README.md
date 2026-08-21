# Hybrid Churn Prediction System

A two-part churn prediction project that combines **supervised classification**, **unsupervised clustering**, and **cross-domain data fusion** to predict and understand customer churn.

The project explores churn from two angles:
1. **Single-domain deep-dive** — supervised + unsupervised modeling on the Telco dataset, including feature engineering, feature selection, and a full clustering comparison (K-Means, GMM, DBSCAN).
2. **Multi-domain fusion** — merging Telco, E-Commerce, and Banking churn datasets into one unified schema to train a single cross-domain XGBoost model.

---

## 📁 Project Structure

| File | Description |
|---|---|
| `Data_preprocessing_and_clustering.ipynb` | EDA, cleaning, feature engineering, supervised baselines, and unsupervised clustering (K-Means, GMM, DBSCAN) on the Telco dataset. |
| `Model_training_and_evaluation.ipynb` | Merges Telco + E-Commerce + Bank datasets into a unified feature schema and trains/evaluates a combined multi-domain churn model. |

---

## 📊 Datasets

| Dataset | Domain | Rows | Target |
|---|---|---|---|
| Telco Customer Churn | Telecom | 7,043 | `Churn` (Yes/No) |
| E-Commerce Dataset | Online Retail | 5,630 | `Churn` (0/1) |
| Churn Modelling (Bank) | Banking | 10,000 | `Exited` (0/1) |

For the multi-domain model, these were mapped to five shared features (`tenure`, `monthly_charges`, `gender`, `num_products`, `is_active`) plus a `source` tag, producing a combined dataset of **22,673 rows** with an overall churn rate of **21.4%**.

---

## 🔍 Part 1 — Telco: Preprocessing, Supervised Models & Clustering

**Pipeline:** cleaning → EDA → feature engineering (`tenure_group`, `num_add_services`, `monthly_charge_ratio`) → encoding/scaling → supervised baselines → clustering.

### Supervised Model Comparison (Telco only)
| Model | Accuracy | F1 (Churn=1) | ROC-AUC |
|---|---|---|---|
| **Logistic Regression** | **0.8048** | **0.5840** | **0.8433** ◄ Best |
| Random Forest | 0.7828 | 0.5433 | 0.8201 |
| XGBoost | 0.7928 | 0.5731 | 0.8367 |

### Clustering Comparison (Telco only)
| Method | Silhouette Score | ARI vs. True Churn | # Clusters |
|---|---|---|---|
| **K-Means (K=2)** | **0.1989** | -0.0064 | 2 ◄ Best separation |
| GMM (10 components, BIC-selected) | 0.0664 | **0.0213** ◄ Best alignment with churn | 10 |
| DBSCAN (auto-eps via KNN knee method) | -1.0000 | 0.0000 | 1 (collapsed — 4 noise points) |

**Key insight:** unsupervised clusters do not naturally align with churn labels — the K-Means clusters instead separate customers by *spend profile* (high-value, low-churn vs. low-value, high-churn), visible in the per-cluster profiling:

| Cluster | Avg Monthly Charges | Avg Total Charges | Avg Add-on Services | Churn Rate |
|---|---|---|---|---|
| 0 | $49.09 | $828.64 | 0.77 | 31.6% |
| 1 | $87.47 | $4,387.56 | 3.88 | 19.2% |

DBSCAN failed to find meaningful structure at the auto-estimated `eps`, collapsing nearly all points into a single cluster — a useful negative result showing DBSCAN's sensitivity to density-based hyperparameters on this feature space.

---

## 🔀 Part 2 — Combined Multi-Domain Model

Telco, E-Commerce, and Bank datasets were mapped to a shared 5-feature schema and merged into one dataset (22,673 rows) with a `source` column indicating dataset origin, then split 80/20 (stratified) for training.

### Model Comparison (Combined Dataset)
| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7841 | 0.0161 | 0.7264 |
| Random Forest | 0.8291 | 0.4844 | 0.8391 |
| **XGBoost** | 0.8282 | **0.4990** | 0.8369 ◄ Best |

**5-Fold Stratified Cross-Validation (XGBoost):**
- Accuracy: 0.8262 ± 0.0036
- F1 Score: 0.4773 ± 0.0184
- ROC-AUC: 0.8316 ± 0.0033

Feature importance and PCA-based visualizations were used to interpret model behavior and confirm the combined feature space retained meaningful churn signal across all three domains. The final model and scaler were serialized (`combined_churn_model.pkl`) for reuse.

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML/Modeling:** Scikit-learn (Logistic Regression, Random Forest, K-Means, GMM, DBSCAN, PCA), XGBoost
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab / Jupyter Notebook

---

## ▶️ How to Run

1. Open either notebook in [Google Colab](https://colab.research.google.com) (badges are included at the top of each file) or run locally with Jupyter.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl
   ```
3. **For `Data_preprocessing_and_clustering.ipynb`:** requires the Telco Customer Churn CSV.
4. **For `Model_training_and_evaluation.ipynb`:** requires all three source files — Telco CSV, E-Commerce XLSX, and Bank Churn Modelling CSV — uploaded when prompted.

---

## 💡 Key Takeaways

- A single-domain model (Telco-only Logistic Regression) achieved the highest ROC-AUC (0.843) of any model in the project, showing that simpler models can outperform ensembles on smaller, well-understood feature sets.
- The cross-domain XGBoost model generalized reasonably well (0.837 ROC-AUC) across three structurally different datasets after feature harmonization, at some cost to F1 score.
- Unsupervised clustering surfaced a meaningful *value-based* customer segmentation but was not a strong proxy for churn itself — a useful distinction between "natural" data structure and the specific target label.

## 🔭 Future Improvements

- Replace manual cross-dataset feature mapping with a more principled domain-adaptation approach.
- Address class imbalance (SMOTE / class weighting) to improve minority-class (churn) recall.
- Add SHAP-based explainability on top of feature importance.
- Package the best model behind a lightweight inference API (FastAPI) for real-time scoring.
