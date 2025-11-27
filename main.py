import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import eda
from src import ml_modeling
from src import business_analysis

def main():
    print("\n" + "="*60)
    print("STARTING REAL ESTATE CAPSTONE PIPELINE")
    print("="*60)
    
    # 1. Run EDA
    print("\n[1/3] Running Exploratory Data Analysis (EDA)...")
    try:
        eda.main()
        print("EDA Complete. Visualizations saved to outputs/figures/eda/")
    except Exception as e:
        print(f"EDA Failed: {e}")

    # 2. Run ML Modeling
    print("\n[2/3] Running ML Modeling & Evaluation...")
    try:
        ml_modeling.main()
        print("ML Modeling Complete. Model saved to outputs/models/")
    except Exception as e:
        print(f"ML Modeling Failed: {e}")

    # 3. Run Business Insights
    print("\n[3/3] Generating Business Insights...")
    try:
        business_analysis.main()
        print("Business Insights Complete. Reports saved to outputs/reports/")
    except Exception as e:
        print(f"Business Insights Failed: {e}")

    print("\n" + "="*60)
    print("PIPELINE EXECUTION FINISHED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()
