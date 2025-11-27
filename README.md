# 🏙️ Ahmedabad Real Estate Smart-Analyst

**A Professional End-to-End Data Science Solution for Real Estate Valuation and Strategic Investment Analysis.**

---

## 📖 Project Overview
This project is a comprehensive machine learning and analytics pipeline designed to solve the opacity in the Ahmedabad real estate market. By leveraging data from major property portals, we have built a system that not only predicts property prices with high accuracy but also uncovers deep strategic insights for investors, developers, and homebuyers.

**Goal:** To transform raw real estate listings into actionable intelligence—identifying undervalued assets, high-growth corridors, and precise fair market values.

---

## 🧠 Technical Deep Dive: How It Works

We use advanced statistical and machine learning techniques to ensure our analysis is robust and reality-based. Here is what happens under the hood:

### 1. 🧹 Outlier Removal using IQR (Interquartile Range)
Real estate data is messy. You often find data entry errors (e.g., a 10,000 sqft apartment for ₹5 Lakhs) or extreme outliers (a ₹50 Crore palace).
*   **What is IQR?**: It stands for **Interquartile Range**. It is a statistical measure of where the "middle 50%" of your data lies.
*   **How we use it**: We calculate the price-per-sqft for every property. Any property that is significantly cheaper (below Q1 - 1.5*IQR) or significantly more expensive (above Q3 + 1.5*IQR) than the normal range is flagged as an outlier and removed.
*   **Benefit**: This ensures our model learns from *market trends*, not data errors.

### 2. 🎯 Locality Target Encoding
"Location, Location, Location" is the golden rule. But how do you feed a location name like "Bopal" to a math model?
*   **The Old Way**: Assigning a random number (Label Encoding) or creating 100s of columns (One-Hot Encoding).
*   **Our Way (Target Encoding)**: We replace the name "Bopal" with the **average property price** in Bopal.
*   **Benefit**: The model immediately understands that "Ambli" (High Price) is more valuable than "Vastral" (Lower Price) without needing thousands of data points.

### 3. 🤖 Ensemble Machine Learning
We don't rely on just one brain. We use a **Voting Regressor**, which is like a committee of experts:
*   **Gradient Boosting**: Great at finding complex, non-linear patterns.
*   **Random Forest**: Excellent at handling noisy data and preventing overfitting.
*   **Linear Regression**: Provides a stable baseline trend.
*   **The Result**: The final prediction is the average of these experts, leading to a highly stable **80.4% Accuracy (R2 Score)**.

---

## 🚀 From Scrape to Insight: Example Outputs

Here is exactly what the system produces at each stage:

### Step 1: 🕵️‍♀️ Data Collection (Scraping)
We scrape raw data from property portals.
*   **Input**: "Ahmedabad"
*   **Raw Output**:
    ```json
    {
      "Title": "3 BHK Flat in Shela",
      "Price": "₹ 75 Lac",
      "Area": "1500 sqft",
      "Locality": "Shela, Ahmedabad West"
    }
    ```

### Step 2: 🧼 Cleaning & Engineering
We clean the text and create new features.
*   **Processed Data**:
    *   `Price_Lakhs`: 75.0
    *   `Area_SqFt`: 1500
    *   `Price_Per_SqFt`: 5000
    *   `Luxury_Score`: **High** (Derived from amenities like Pool/Gym)

### Step 3: 🔮 Prediction (Machine Learning)
You give the model a property details, and it tells you the **Fair Market Value**.
*   **Input**: 3 BHK, 1600 SqFt, Gated Community in **Gota**.
*   **Model Prediction**: **₹ 82.4 Lakhs**
*   **Use Case**: If the seller is asking ₹ 95 Lakhs, you know it's **Overpriced**.

### Step 4: 💡 Strategic Insights
The system scans the entire market to find opportunities.

**Example 1: Undervalued Gems** (from `data/insights/top_10_undervalued_localities.csv`)
> *"Look at **Chandkheda**. It has a high Value Score (Good Infrastructure) but prices are 15% lower than the city average. High appreciation potential!"*

**Example 2: Bachelor Hubs** (from `data/insights/top_10_bachelor_hubs.csv`)
> *"Investors should target **Navrangpura**. It has the highest density of 1 BHK units and high rental demand from students."*

---

## 📂 Project Structure
The repository is organized for modularity and scalability:

```bash
real_estate_capstone/
├── data/
│   ├── processed/         # Cleaned & Engineered Data (CSV)
│   └── insights/          # Generated Strategic Reports (Top 10 Lists, Deal Picks)
│
├── src/
│   ├── eda.py             # Visualization & Statistical Analysis Engine
│   ├── ml_modeling.py     # Model Training, Tuning & Evaluation Pipeline
│   ├── business_analysis.py # Business Logic & Insight Generation
│   └── main.py            # Orchestrator Script
│
├── outputs/
│   ├── models/            # Serialized Models (best_model.pkl)
│   ├── figures/           # All generated charts (EDA, ML, Business)
│   └── reports/           # Detailed Metrics & Analysis CSVs
│
└── docs/                  # Documentation & Setup Guides
```

---

## 🛠️ How to Run the Project

### Prerequisites
*   Python 3.8+
*   Git

### Installation
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/real-estate-capstone.git
    cd real-estate-capstone
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Execution
Run the entire pipeline (Data Cleaning -> EDA -> Modeling -> Insights) with one command:
```bash
python main.py
```

---

## 📝 License
This project is open-source and available for educational and commercial analysis purposes.
