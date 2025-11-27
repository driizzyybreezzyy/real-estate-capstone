import pandas as pd
import numpy as np
import re
import os

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    print("  > Cleaning data...")
    initial_count = len(df)
    
    # 1. Remove Duplicates (Strict)
    df = df.drop_duplicates(subset=['Property_Title', 'Price_Lakhs', 'Area_SqFt_Clean', 'Locality'], keep='first')
    print(f"    - Removed {initial_count - len(df)} duplicates")
    
    # 2. Handle Missing Critical Values
    # Drop rows where Price or Area is missing as they are useless for analysis
    df = df.dropna(subset=['Price_Lakhs', 'Area_SqFt_Clean'])
    
    # 3. Impute Missing BHK and Bathrooms
    # Logic: If BHK is missing, estimate from Area (approx 600 sqft per BHK for apts)
    def impute_bhk(row):
        if pd.notna(row['BHK_Clean']): return row['BHK_Clean']
        if pd.notna(row['Area_SqFt_Clean']):
            return max(1, round(row['Area_SqFt_Clean'] / 600))
        return np.nan
    
    df['BHK_Clean'] = df.apply(impute_bhk, axis=1)
    
    # Logic: Bathrooms usually = BHK or BHK + 1
    df['Bathrooms_Clean'] = df['Bathrooms_Clean'].fillna(df['BHK_Clean'])
    
    # 4. Remove Outliers (Business Logic)
    # Price: < 5 Lakhs (likely rent/error) or > 50 Cr (commercial/outlier)
    df = df[(df['Price_Lakhs'] >= 5) & (df['Price_Lakhs'] <= 5000)]
    
    # Area: < 300 sqft (too small) or > 20000 (likely commercial land)
    df = df[(df['Area_SqFt_Clean'] >= 300) & (df['Area_SqFt_Clean'] <= 20000)]
    
    print(f"    - Retained {len(df)} rows after cleaning and outlier removal")
    return df

def engineer_indian_features(df):
    print("  > Engineering 'Indian Businessman' features...")
    
    # 1. Carpet Area Estimation (The "Real" Area)
    # Standard practice: Super Built-up * 0.70 = Carpet Area
    df['Carpet_Area_Est'] = (df['Area_SqFt_Clean'] * 0.70).round(0)
    
    # 2. Vastu Compliance
    def check_vastu(text):
        if pd.isna(text): return 0
        text = str(text).lower()
        if any(x in text for x in ['vastu', 'east facing', 'north facing', 'pooja room']):
            return 1
        return 0
    df['Vastu_Compliant'] = df['Raw_JSON'].apply(check_vastu)
    
    # 3. Transaction Type (Resale vs New)
    def check_transaction(text):
        if pd.isna(text): return "Resale" # Default
        text = str(text).lower()
        if any(x in text for x in ['new booking', 'under construction', 'possession soon', 'fresh']):
            return "New Booking"
        return "Resale"
    df['Transaction_Type'] = df['Raw_JSON'].apply(check_transaction)
    
    # 4. Gated Community / Lifestyle
    def check_gated(text):
        if pd.isna(text): return 0
        text = str(text).lower()
        if any(x in text for x in ['gated', 'security', 'society', 'cctv', 'guard']):
            return 1
        return 0
    df['Gated_Community'] = df['Raw_JSON'].apply(check_gated)
    
    # 5. Price Fairness (The "Deal" Metric)
    # Calculate median Price/SqFt per locality
    locality_medians = df.groupby('Locality')['Price_Per_SqFt'].median()
    
    def assess_fairness(row):
        loc = row['Locality']
        price_psf = row['Price_Per_SqFt']
        
        if loc not in locality_medians or pd.isna(price_psf):
            return "Unknown"
            
        median_price = locality_medians[loc]
        if median_price == 0: return "Unknown"
        
        ratio = price_psf / median_price
        
        if ratio < 0.85: return "Undervalued (Great Deal)"
        elif ratio > 1.15: return "Overvalued (Premium)"
        else: return "Fair Value"
        
    df['Price_Fairness'] = df.apply(assess_fairness, axis=1)
    
    # 6. Locality Tier (Refined)
    # Based on Price/SqFt
    def refine_tier(price_psf):
        if pd.isna(price_psf): return "Mid-Segment"
        if price_psf < 3500: return "Affordable"
        elif price_psf < 6000: return "Mid-Segment"
        elif price_psf < 9000: return "Premium"
        else: return "Luxury"
        
    df['Locality_Tier'] = df['Price_Per_SqFt'].apply(refine_tier)
    
    return df

def main():
    input_file = "ahmedabad_real_estate_final_dataset.csv"
    output_file = "ahmedabad_real_estate_cleaned.csv"
    
    print(f"Starting Data Preparation on {input_file}...")
    
    try:
        df = load_data(input_file)
        
        # Clean
        df = clean_data(df)
        
        # Engineer Features
        df = engineer_indian_features(df)
        
        # Save
        df.to_csv(output_file, index=False)
        print(f"Success! Cleaned data saved to {output_file}")
        print(f"Final Shape: {df.shape}")
        
        # Preview Insights
        print("\n--- Business Insights Preview ---")
        print(df['Price_Fairness'].value_counts())
        print("\n")
        print(df['Locality_Tier'].value_counts())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
