# 📂 Project Structure Guide

This project follows a modular, professional data science structure designed for scalability and reproducibility.

## 🏗️ Directory Layout

```
real_estate_capstone/
│
├── 📄 main.py                    # 🚀 Master Script: Runs the entire pipeline (EDA -> ML -> Insights)
├── 📄 requirements.txt           # Python dependencies
│
├── 📁 data/                      # Centralized Data Storage
│   ├── 📁 processed/             # Cleaned & Preprocessed Data (ahmedabad_real_estate_cleaned.csv)
│   └── 📁 insights/              # Raw Insight Reports (top_10_expensive.csv, etc.)
│
├── 📁 src/                       # Source Code (Modularized)
│   ├── 📄 eda.py                 # Exploratory Data Analysis & Visualization
│   ├── 📄 ml_modeling.py         # Machine Learning Training, Tuning & Evaluation
│   └── 📄 business_analysis.py   # Business Logic & Insight Generation
│
├── 📁 outputs/                   # Generated Artifacts (Do not edit manually)
│   ├── 📁 models/                # Trained ML Models (best_model.pkl)
│   ├── 📁 figures/               # Visualizations
│   │   ├── 📁 eda/               # Market Analysis Charts
│   │   ├── 📁 ml/                # Model Performance Charts
│   │   └── 📁 business/          # Business Insight Charts
│   └── 📁 reports/               # CSV Results & Metrics
│
└── 📁 docs/                      # Documentation
    └── 📄 PROJECT_STRUCTURE.md   # This file
```

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Full Pipeline**:
    ```bash
    python main.py
    ```
    This will:
    - Load data from `data/processed/`
    - Generate EDA plots in `outputs/figures/eda/`
    - Train ML models and save the best one to `outputs/models/`
    - Generate business reports in `outputs/reports/`

## 🔑 Key Files

-   **`src/ml_modeling.py`**: Contains the logic for the Ensemble Voting Regressor (RF + GB + LR) that achieved **80.4% Accuracy**.
-   **`outputs/models/best_model.pkl`**: The final trained model ready for deployment.
-   **`outputs/reports/comprehensive_model_results.csv`**: Detailed metrics for all tested models.
