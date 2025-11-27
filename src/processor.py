import pandas as pd
import numpy as np
import re
import os

def clean_price(price_str):
    if pd.isna(price_str) or price_str == 'N/A':
        return np.nan
    
    price_str = str(price_str).replace('₹', '').replace(',', '').strip().lower()
    
    try:
        if 'cr' in price_str:
            return float(re.findall(r"[\d\.]+", price_str)[0]) * 100
        elif 'l' in price_str or 'lac' in price_str:
            return float(re.findall(r"[\d\.]+", price_str)[0])
        elif 'k' in price_str: 
             return float(re.findall(r"[\d\.]+", price_str)[0]) / 100
        else:
            val = float(re.findall(r"[\d\.]+", price_str)[0])
            if val > 10000: 
                return val / 100000
            return val 
    except:
        return np.nan

def clean_area(area_str):
    if pd.isna(area_str) or area_str == 'N/A':
        return np.nan
    area_str = str(area_str).replace(',', '').lower()
    try:
        val = float(re.findall(r"[\d\.]+", area_str)[0])
        if 'sqyrd' in area_str or 'sq.yrd' in area_str:
            return val * 9 
        return val
    except:
        return np.nan

def clean_bhk(bhk_str):
    if pd.isna(bhk_str) or bhk_str == 'N/A':
        return np.nan
    try:
        return int(re.findall(r"\d+", str(bhk_str))[0])
    except:
        return np.nan

def clean_bathrooms(bath_str):
    if pd.isna(bath_str) or bath_str == 'N/A':
        return np.nan
    try:
        return int(re.findall(r"\d+", str(bath_str))[0])
    except:
        return np.nan

def extract_floor(raw_text):
    if pd.isna(raw_text): return "N/A"
    match = re.search(r'(ground|lower basement|upper basement|\d+)(?:st|nd|rd|th)?\s+floor', str(raw_text), re.IGNORECASE)
    if match:
        return match.group(0).title()
    return "N/A"

def extract_project(raw_text, title):
    if isinstance(title, str) and " in " in title:
        parts = title.split(" in ")
        if len(parts) > 1:
            potential_proj = parts[0].strip()
            if not re.search(r'\d+\s*BHK', potential_proj, re.IGNORECASE):
                return potential_proj
    return "N/A"

def determine_property_type(title, raw_text):
    text = (str(title) + " " + str(raw_text)).lower()
    if 'plot' in text or 'land' in text:
        return 'Plot'
    elif 'villa' in text or 'bungalow' in text or 'house' in text:
        return 'Villa/Bungalow'
    elif 'apartment' in text or 'flat' in text or 'penthouse' in text:
        return 'Apartment'
    return 'Apartment' 

def calculate_luxury_score(raw_text):
    if pd.isna(raw_text): return 0
    keywords = ['pool', 'gym', 'club', 'garden', 'premium', 'luxury', 'gated', 'security', 'modular kitchen', 'furnished']
    score = 0
    text = str(raw_text).lower()
    for kw in keywords:
        if kw in text:
            score += 1
    return score

def extract_date(text):
    if pd.isna(text): return "N/A"
    match = re.search(r'posted\s*(\d+\s*\w+\s*ago)', str(text), re.IGNORECASE)
    if match: return match.group(1)
    return "N/A"

def main():
    print("Loading raw data...")
    input_file = "ahmedabad_real_estate_raw_data.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    df = pd.read_csv(input_file)
    
    print("Processing data...")
    
    # 1. Cleaning
    df['Price_Lakhs'] = df['Price'].apply(clean_price)
    df['Area_SqFt_Clean'] = df['Area'].apply(clean_area)
    df['BHK_Clean'] = df['BHK'].apply(clean_bhk)
    df['Bathrooms_Clean'] = df['Bathrooms'].apply(clean_bathrooms)
    
    # 2. Standardization
    df['Furnishing_Status_Clean'] = df['Furnishing'].replace({'N/A': 'Unfurnished'})
    
    # 3. Extraction
    df['Floor_Number'] = df['Raw_Details'].apply(extract_floor)
    df['Project_Name'] = df.apply(lambda x: extract_project(x['Raw_Details'], x['Property Title']), axis=1)
    df['Property_Type'] = df.apply(lambda x: determine_property_type(x['Property Title'], x['Raw_Details']), axis=1)
    df['Extracted_Locality'] = df['Locality'] 
    df['Posted_Date'] = df['Raw_Details'].apply(extract_date)
    
    # 4. Feature Engineering
    df['Price_Per_SqFt'] = (df['Price_Lakhs'] * 100000) / df['Area_SqFt_Clean']
    df['Price_Per_SqFt'] = df['Price_Per_SqFt'].round(2)
    
    # Locality Tier
    locality_stats = df.groupby('Locality')['Price_Per_SqFt'].mean()
    overall_mean = df['Price_Per_SqFt'].mean()
    
    def get_tier(loc):
        if pd.isna(loc) or loc not in locality_stats:
            val = overall_mean
        else:
            val = locality_stats[loc]
            
        if pd.isna(val): return "Tier 2" 
        
        if val > df['Price_Per_SqFt'].quantile(0.8):
            return "Tier 1 (Premium)"
        elif val < df['Price_Per_SqFt'].quantile(0.3):
            return "Tier 3 (Budget)"
        else:
            return "Tier 2 (Mid-Segment)"
            
    df['Locality_Tier'] = df['Locality'].apply(get_tier)
    
    # Property Category
    def get_category(price):
        if pd.isna(price): return "Unknown"
        if price > 150: return "Luxury"
        elif price > 75: return "Premium"
        else: return "Budget"
    df['Property_Category'] = df['Price_Lakhs'].apply(get_category)
    
    # BHK Type
    def get_bhk_type(bhk):
        if pd.isna(bhk): return "Unknown"
        if bhk >= 4: return "4+ BHK"
        return f"{int(bhk)} BHK"
    df['BHK_Type'] = df['BHK_Clean'].apply(get_bhk_type)
    
    # 5. Creative Fields
    df['Luxury_Score'] = df['Raw_Details'].apply(calculate_luxury_score)
    df['Space_Efficiency'] = df['Area_SqFt_Clean'] / df['BHK_Clean']
    df['Space_Efficiency'] = df['Space_Efficiency'].round(2)
    
    # Mapping
    df['Area_SqFt'] = df['Area']
    df['Furnishing_Status'] = df['Furnishing']
    df['Source_Website'] = df['Source']
    df['Raw_JSON'] = df['Raw_Details']
    
    rename_map = {
        'Property Title': 'Property_Title',
        'Seller Type': 'Seller_Type'
    }
    df = df.rename(columns=rename_map)
    
    requested_cols = [
        'Property_Title', 'Price', 'Area_SqFt', 'BHK', 'Bathrooms', 'Furnishing_Status', 
        'Property_Type', 'Seller_Type', 'Project_Name', 'Locality', 'Posted_Date', 
        'Floor_Number', 'Source_Website', 'Raw_JSON', 'Price_Lakhs', 'Area_SqFt_Clean', 
        'BHK_Clean', 'Extracted_Locality', 'Furnishing_Status_Clean', 'Bathrooms_Clean', 
        'Price_Per_SqFt', 'Locality_Tier', 'Property_Category', 'BHK_Type',
        'Luxury_Score', 'Space_Efficiency'
    ]
    
    for col in requested_cols:
        if col not in df.columns:
            df[col] = "N/A"
            
    final_df = df[requested_cols]
    
    output_csv = "ahmedabad_real_estate_final_dataset.csv"
    output_xlsx = "ahmedabad_real_estate_final_dataset.xlsx"
    
    final_df.to_csv(output_csv, index=False)
    final_df.to_excel(output_xlsx, index=False)
    
    print(f"Success! Processed {len(final_df)} rows.")
    print(f"Saved to {output_csv} and {output_xlsx}")

if __name__ == "__main__":
    main()
