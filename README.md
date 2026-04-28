# 🏠 California Housing Price Prediction
### ML Lab Final Project · SP24-BDS-011

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Complete-2ea44f?style=for-the-badge" />
</p>

> End-to-end machine learning pipeline for predicting median house values across California districts using the classic 1990 census dataset. Covers data loading, EDA, preprocessing, multi-model training, hyperparameter tuning, and evaluation.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Model Results](#-model-results)
- [Quick Start](#-quick-start)
- [Notebooks Guide](#-notebooks-guide)
- [Documentation](#-documentation)
- [Improvement Roadmap](#-improvement-roadmap)
- [Course Info](#-course-info)

---

## 🔍 Overview

This project explores the full ML workflow applied to a real-world regression problem:

| Stage | Description |
|---|---|
| **Data Loading** | California Housing Dataset (sklearn, 20,640 samples, 8 features) |
| **EDA** | Statistical analysis, correlation heatmaps, geographic visualizations |
| **Preprocessing** | Train/test split, StandardScaler feature normalization |
| **Modeling** | 6 regression models trained and compared |
| **Tuning** | GridSearchCV hyperparameter optimization |
| **Evaluation** | R², RMSE, MAE with cross-validation |

**Best achieved performance:** R² = **0.805**, RMSE = **0.505** (Random Forest)

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | `sklearn.datasets.fetch_california_housing` |
| **Origin** | 1990 California Census |
| **Samples** | 20,640 districts |
| **Features** | 8 numeric |
| **Target** | `MedHouseVal` — Median house value (in $100,000 units) |

### Features

| Feature | Description |
|---|---|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

---

## 📁 Project Structure

```
Lab Final/
│
├── notebooks/
│   ├── california_housing_prediction.ipynb          # Main notebook — full ML pipeline
│   ├── california_housing_prediction_COMMENTED.ipynb # Annotated version (500+ comments)
│   └── lab_terminal.ipynb                           # Supplementary experiments
│
├── scripts/
│   ├── add_notebook_comments.py                     # Utility: adds comments to notebooks
│   └── generate_comprehensive_pdf.py                # Utility: exports analysis to PDF
│
├── docs/
│   ├── Comprehensive_Code_Analysis.md               # 150+ page line-by-line code analysis
│   ├── Comprehensive_Code_Analysis_Report.pdf       # PDF version of analysis
│   ├── DELIVERABLES_SUMMARY.md                      # Summary of all deliverables
│   ├── README_COMMENTED_NOTEBOOKS.md                # Guide to commented notebooks
│   ├── SP24-BDS-011 Project Report.docx             # Final submission report
│   └── Project, Submission date 15-12-25.pdf        # Submission PDF
│
├── .venv/                                           # Python virtual environment
└── README.md                                        # You are here
```

---

## 🏆 Model Results

All models trained on an 80/20 train-test split with `random_state=42`.

| Model | R² Score | RMSE | MAE | Notes |
|---|---|---|---|---|
| Linear Regression | 0.652 | 0.675 | 0.486 | Baseline |
| Ridge Regression | 0.612 | — | — | L2 regularization |
| Lasso Regression | 0.610 | — | — | L1 regularization |
| Decision Tree | 0.623 | 0.717 | 0.462 | Prone to overfitting |
| **Random Forest** ✅ | **0.805** | **0.505** | **0.327** | **Best model** |
| Gradient Boosting | 0.776 | 0.511 | — | Close second |

> 📈 **Projected performance with improvements** (XGBoost + feature engineering + tuning): R² **0.86–0.89** (+7–11%)

---

## 🚀 Quick Start

### 1. Activate the Virtual Environment

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, the core packages are:
> ```bash
> pip install numpy pandas matplotlib seaborn scikit-learn jupyter
> ```

### 3. Launch Jupyter

```bash
jupyter notebook
```

Then open `notebooks/california_housing_prediction.ipynb`.

---

## 📓 Notebooks Guide

### `california_housing_prediction.ipynb` — **Start Here**
The main project notebook. Runs the full pipeline from data loading to model evaluation.

**Sections:**
1. Library Imports & Configuration
2. Dataset Loading & Initial Inspection
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing & Scaling
5. Model Training (6 models)
6. Model Evaluation & Comparison
7. Hyperparameter Tuning (GridSearchCV)
8. Final Results & Visualization

---

### `california_housing_prediction_COMMENTED.ipynb` — **For Learning**
Identical pipeline with **500+ inline comments** explaining:
- Every import and parameter
- Mathematical formulas behind each algorithm
- Why certain choices were made
- Common pitfalls and how to avoid them
- Alternative approaches

---

### `lab_terminal.ipynb` — **Supplementary**
Additional experiments and exploratory work done during lab sessions.

---

## 📚 Documentation

All extended documentation lives in the `docs/` folder:

| Document | Size | Purpose |
|---|---|---|
| `Comprehensive_Code_Analysis.md` | 34 KB (~150 pages) | Deep line-by-line analysis + improvement guide |
| `DELIVERABLES_SUMMARY.md` | 15 KB | Summary of all project deliverables |
| `README_COMMENTED_NOTEBOOKS.md` | 6.6 KB | Quick reference for commented notebooks |
| `Comprehensive_Code_Analysis_Report.pdf` | PDF | Printable version of the analysis |

### Reading Path

**Just want to understand the code?**
→ `Comprehensive_Code_Analysis.md` → Phases 1–4 → "Best Practices" → "Conclusion"
*(~30–45 minutes)*

**Want to learn ML deeply?**
→ `README_COMMENTED_NOTEBOOKS.md` → `Comprehensive_Code_Analysis.md` → run `_COMMENTED.ipynb`
*(~4–6 hours)*

**Need production-ready code snippets?**
→ Search "Production" or "Pipeline" in `Comprehensive_Code_Analysis.md`
*(~2–3 hours)*

---

## 🗺️ Improvement Roadmap

### Priority 1 — Quick Wins (Low effort, high impact)
- [ ] Replace blanket `warnings.filterwarnings('ignore')` with selective filtering
- [ ] Add `sklearn.pipeline.Pipeline` to prevent data leakage
- [ ] Centralize config in a `CONFIG` dictionary (random seed, test size, CV folds)
- [ ] Implement stratified train-test split

### Priority 2 — Performance Gains (~2–5 hours)
- [ ] Feature engineering: geographic clusters, income/room ratios, polynomial features
- [ ] Add XGBoost and LightGBM (+2–5% R² expected)
- [ ] Implement model versioning with metadata

### Priority 3 — Production Readiness (~20+ hours)
- [ ] Set up MLflow for experiment tracking
- [ ] Create FastAPI prediction endpoint
- [ ] Docker containerization
- [ ] Unit tests with pytest

---

## 🎓 Course Info

| Field | Detail |
|---|---|
| **Course** | Machine Learning |
| **Student ID** | SP24-BDS-011 |
| **Submission Date** | December 15, 2025 |
| **Status** | ✅ Completed |

---

<p align="center">
  Made with 🤖 scikit-learn · 📊 pandas · 📈 matplotlib
</p>
