import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    # Construct absolute path to data file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'ahmedabad_real_estate_cleaned.csv')
    return pd.read_csv(data_path)

def remove_outliers(df):
    """
    Remove extreme outliers based on Price_Per_SqFt using IQR method
    """
    Q1 = df['Price_Per_SqFt'].quantile(0.25)
    Q3 = df['Price_Per_SqFt'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    initial_len = len(df)
    df_clean = df[(df['Price_Per_SqFt'] >= lower_bound) & (df['Price_Per_SqFt'] <= upper_bound)]
    final_len = len(df_clean)
    
    print(f"[-] Removed {initial_len - final_len} outliers (Price/SqFt > 1.5*IQR)")
    return df_clean

def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 7)

def save_plot(filename):
    # Construct absolute path to output directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs', 'figures', 'business')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# ===== USE CASE 1: Best Locality for Development =====
def analyze_development_opportunities(df):
    """
    Identify best localities for residential vs commercial development
    """
    print("\n=== USE CASE 1: Best Locality for Development ===")
    
    # Calculate locality metrics
    locality_stats = df.groupby('Locality').agg({
        'Price_Lakhs': ['count', 'median', 'mean'],
        'Price_Per_SqFt': 'median',
        'Area_SqFt_Clean': 'median',
        'BHK_Clean': 'mean',
        'Property_Type': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
    }).reset_index()
    
    locality_stats.columns = ['Locality', 'Supply', 'Median_Price', 'Mean_Price', 
                               'Price_Per_SqFt', 'Median_Area', 'Avg_BHK', 'Dominant_Type']
    
    # Filter top 20 localities by supply
    top_localities = locality_stats.nlargest(20, 'Supply')
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Supply vs Price/SqFt
    scatter = axes[0].scatter(top_localities['Supply'], top_localities['Price_Per_SqFt'], 
                              s=top_localities['Median_Price']*2, alpha=0.6, c=range(len(top_localities)), cmap='viridis')
    for idx, row in top_localities.iterrows():
        axes[0].annotate(row['Locality'].split(',')[0], 
                        (row['Supply'], row['Price_Per_SqFt']), 
                        fontsize=8, alpha=0.7)
    axes[0].set_xlabel('Supply (# of Properties)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price per SqFt (₹)', fontsize=12, fontweight='bold')
    axes[0].set_title('Development Opportunity Matrix', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Property Type Distribution
    property_dist = df.groupby(['Locality', 'Property_Type']).size().unstack(fill_value=0)
    top_10_locs = locality_stats.nlargest(10, 'Supply')['Locality']
    property_dist_top = property_dist.loc[property_dist.index.isin(top_10_locs)]
    property_dist_top.plot(kind='barh', stacked=True, ax=axes[1], colormap='Set3')
    axes[1].set_xlabel('Number of Properties', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Locality', fontsize=12, fontweight='bold')
    axes[1].set_title('Property Type Distribution (Top 10 Localities)', fontsize=14, fontweight='bold')
    axes[1].legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_plot('use_case_1_development_opportunities.png')
    
    # Recommendations
    recommendations = {
        'High_Demand_Low_Supply': top_localities[top_localities['Supply'] < top_localities['Supply'].median()].nlargest(3, 'Price_Per_SqFt'),
        'Residential_Focus': top_localities[top_localities['Dominant_Type'] == 'Apartment'].nlargest(5, 'Supply'),
        'Premium_Development': top_localities.nlargest(5, 'Price_Per_SqFt')
    }
    
    return locality_stats, recommendations

# ===== USE CASE 2: Investment Zones =====
def analyze_investment_zones(df):
    """
    Identify affordable and high-value investment zones
    """
    print("\n=== USE CASE 2: Affordable & High-Value Investment Zones ===")
    
    # Calculate value score
    df['Value_Score'] = (df['Area_SqFt_Clean'] / df['Price_Lakhs']) * 100
    
    locality_value = df.groupby('Locality').agg({
        'Price_Per_SqFt': 'median',
        'Value_Score': 'mean',
        'Price_Lakhs': ['median', 'count'],
        'BHK_Clean': 'mean'
    }).reset_index()
    
    locality_value.columns = ['Locality', 'Price_Per_SqFt', 'Value_Score', 'Median_Price', 'Supply', 'Avg_BHK']
    
    # Filter localities with sufficient supply
    locality_value = locality_value[locality_value['Supply'] >= 10]
    
    # Categorize
    price_quartiles = locality_value['Price_Per_SqFt'].quantile([0.25, 0.75])
    value_quartiles = locality_value['Value_Score'].quantile([0.25, 0.75])
    
    def categorize_investment(row):
        if row['Price_Per_SqFt'] < price_quartiles[0.25]:
            if row['Value_Score'] > value_quartiles[0.75]:
                return 'Best Value (Affordable + High Value)'
            else:
                return 'Affordable'
        elif row['Price_Per_SqFt'] > price_quartiles[0.75]:
            return 'Premium'
        else:
            return 'Mid-Segment'
    
    locality_value['Investment_Category'] = locality_value.apply(categorize_investment, axis=1)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Investment Matrix
    categories = locality_value['Investment_Category'].unique()
    colors = {'Best Value (Affordable + High Value)': 'green', 'Affordable': 'blue', 'Mid-Segment': 'orange', 'Premium': 'red'}
    
    for category in categories:
        subset = locality_value[locality_value['Investment_Category'] == category]
        axes[0].scatter(subset['Price_Per_SqFt'], subset['Value_Score'], 
                       label=category, alpha=0.6, s=100, c=colors.get(category, 'gray'))
    
    axes[0].axhline(y=value_quartiles[0.75], color='gray', linestyle='--', alpha=0.5, label='High Value Threshold')
    axes[0].axvline(x=price_quartiles[0.25], color='gray', linestyle='--', alpha=0.5, label='Affordable Threshold')
    axes[0].set_xlabel('Price per SqFt (₹)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Value Score (Area/Price)', fontsize=12, fontweight='bold')
    axes[0].set_title('Investment Zone Matrix', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Top 10 Best Value Localities
    best_value = locality_value.nlargest(10, 'Value_Score')
    axes[1].barh(range(len(best_value)), best_value['Value_Score'], color='green', alpha=0.7)
    axes[1].set_yticks(range(len(best_value)))
    axes[1].set_yticklabels([loc.split(',')[0] for loc in best_value['Locality']], fontsize=10)
    axes[1].set_xlabel('Value Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Top 10 Best Value Localities', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_plot('use_case_2_investment_zones.png')
    
    return locality_value

# ===== USE CASE 3: New vs Resale Market =====
def analyze_new_vs_resale(df):
    """
    Compare New Booking vs Resale prices (proxy for buy advantage)
    """
    print("\n=== USE CASE 3: New vs Resale Market Advantage ===")
    
    # Filter data
    transaction_data = df[df['Transaction_Type'].isin(['New Booking', 'Resale'])]
    
    # Calculate average prices by locality and transaction type
    locality_transaction = transaction_data.groupby(['Locality', 'Transaction_Type']).agg({
        'Price_Per_SqFt': 'median',
        'Price_Lakhs': ['median', 'count']
    }).reset_index()
    
    locality_transaction.columns = ['Locality', 'Transaction_Type', 'Price_Per_SqFt', 'Median_Price', 'Count']
    
    # Pivot to compare
    comparison = locality_transaction.pivot_table(
        index='Locality', 
        columns='Transaction_Type', 
        values='Price_Per_SqFt'
    ).reset_index()
    
    # Filter localities with both types
    comparison = comparison.dropna()
    comparison['Price_Difference'] = comparison['New Booking'] - comparison['Resale']
    comparison['Percent_Difference'] = (comparison['Price_Difference'] / comparison['Resale']) * 100
    
    # Get top 15 localities by total supply
    top_localities = transaction_data.groupby('Locality').size().nlargest(15).index
    comparison_top = comparison[comparison['Locality'].isin(top_localities)]
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Price Comparison
    x = range(len(comparison_top))
    width = 0.35
    axes[0].bar([i - width/2 for i in x], comparison_top['New Booking'], width, label='New Booking', color='skyblue', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], comparison_top['Resale'], width, label='Resale', color='coral', alpha=0.8)
    axes[0].set_xlabel('Locality', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price per SqFt (₹)', fontsize=12, fontweight='bold')
    axes[0].set_title('New vs Resale Price Comparison (Top 15 Localities)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([loc.split(',')[0] for loc in comparison_top['Locality']], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Percent Difference
    colors = ['green' if x < 0 else 'red' for x in comparison_top['Percent_Difference']]
    axes[1].barh(range(len(comparison_top)), comparison_top['Percent_Difference'], color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(comparison_top)))
    axes[1].set_yticklabels([loc.split(',')[0] for loc in comparison_top['Locality']], fontsize=10)
    axes[1].set_xlabel('Price Difference (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('New vs Resale Premium (Negative = Resale Cheaper)', fontsize=14, fontweight='bold')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_plot('use_case_3_new_vs_resale.png')
    
    return comparison_top

# ===== USE CASE 4: Family vs Bachelor Areas =====
def analyze_family_vs_bachelor(df):
    """
    Identify family-friendly vs bachelor-friendly areas
    """
    print("\n=== USE CASE 4: Family-Friendly vs Bachelor-Friendly Areas ===")
    
    # Define criteria
    df['Family_Score'] = 0
    df['Bachelor_Score'] = 0
    
    # Family indicators
    df.loc[df['BHK_Clean'] >= 3, 'Family_Score'] += 2
    df.loc[df['Gated_Community'] == 1, 'Family_Score'] += 1
    df.loc[df['Price_Per_SqFt'] > df['Price_Per_SqFt'].median(), 'Family_Score'] += 1
    
    # Bachelor indicators
    df.loc[df['BHK_Clean'] <= 2, 'Bachelor_Score'] += 2
    df.loc[df['Price_Per_SqFt'] < df['Price_Per_SqFt'].median(), 'Bachelor_Score'] += 1
    df.loc[df['Furnishing_Status_Clean'] != 'Unfurnished', 'Bachelor_Score'] += 1
    
    # Aggregate by locality
    locality_profile = df.groupby('Locality').agg({
        'Family_Score': 'mean',
        'Bachelor_Score': 'mean',
        'BHK_Clean': 'mean',
        'Price_Per_SqFt': 'median',
        'Price_Lakhs': 'count'
    }).reset_index()
    
    locality_profile.columns = ['Locality', 'Family_Score', 'Bachelor_Score', 'Avg_BHK', 'Price_Per_SqFt', 'Supply']
    
    # Filter sufficient supply
    locality_profile = locality_profile[locality_profile['Supply'] >= 10]
    
    # Categorize
    def categorize_area(row):
        if row['Family_Score'] > row['Bachelor_Score'] * 1.2:
            return 'Family-Friendly'
        elif row['Bachelor_Score'] > row['Family_Score'] * 1.2:
            return 'Bachelor-Friendly'
        else:
            return 'Mixed'
    
    locality_profile['Category'] = locality_profile.apply(categorize_area, axis=1)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Score Comparison
    categories = locality_profile['Category'].unique()
    colors_map = {'Family-Friendly': 'blue', 'Bachelor-Friendly': 'orange', 'Mixed': 'gray'}
    
    for category in categories:
        subset = locality_profile[locality_profile['Category'] == category]
        axes[0].scatter(subset['Family_Score'], subset['Bachelor_Score'], 
                       label=category, alpha=0.6, s=100, c=colors_map.get(category, 'gray'))
    
    axes[0].plot([0, 5], [0, 5], 'k--', alpha=0.3, label='Equal Score Line')
    axes[0].set_xlabel('Family Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Bachelor Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Locality Profile Matrix', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: BHK Distribution by Category
    bhk_dist = df.groupby(['Locality', 'BHK_Clean']).size().unstack(fill_value=0)
    
    # Get top family and bachelor localities
    top_family = locality_profile[locality_profile['Category'] == 'Family-Friendly'].nlargest(5, 'Family_Score')
    top_bachelor = locality_profile[locality_profile['Category'] == 'Bachelor-Friendly'].nlargest(5, 'Bachelor_Score')
    
    combined = pd.concat([top_family, top_bachelor])
    bhk_dist_subset = bhk_dist.loc[bhk_dist.index.isin(combined['Locality'])]
    
    bhk_dist_subset.plot(kind='barh', stacked=True, ax=axes[1], colormap='Set2')
    axes[1].set_xlabel('Number of Properties', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Locality', fontsize=12, fontweight='bold')
    axes[1].set_title('BHK Distribution (Top Family & Bachelor Areas)', fontsize=14, fontweight='bold')
    axes[1].legend(title='BHK', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_plot('use_case_4_family_vs_bachelor.png')
    
    return locality_profile

# ===== USE CASE 5: Amenities Impact =====
def analyze_amenities_impact(df):
    """
    Analyze impact of amenities on price
    """
    print("\n=== USE CASE 5: Amenities Impact on Price ===")
    
    # Prepare data
    amenities_analysis = {}
    
    # 1. Furnishing Impact
    furnishing_impact = df.groupby('Furnishing_Status_Clean')['Price_Per_SqFt'].agg(['median', 'mean', 'count']).reset_index()
    furnishing_impact['Premium_%'] = ((furnishing_impact['median'] / furnishing_impact['median'].min()) - 1) * 100
    amenities_analysis['Furnishing'] = furnishing_impact
    
    # 2. Property Type Impact
    property_impact = df.groupby('Property_Type')['Price_Per_SqFt'].agg(['median', 'mean', 'count']).reset_index()
    property_impact['Premium_%'] = ((property_impact['median'] / property_impact['median'].min()) - 1) * 100
    amenities_analysis['Property_Type'] = property_impact
    
    # 3. Vastu Impact
    if 'Vastu_Compliant' in df.columns:
        vastu_impact = df.groupby('Vastu_Compliant')['Price_Per_SqFt'].agg(['median', 'mean', 'count']).reset_index()
        vastu_impact['Vastu_Compliant'] = vastu_impact['Vastu_Compliant'].map({0: 'No', 1: 'Yes'})
        vastu_impact['Premium_%'] = ((vastu_impact['median'] / vastu_impact['median'].min()) - 1) * 100
        amenities_analysis['Vastu'] = vastu_impact
    
    # 4. Gated Community Impact
    if 'Gated_Community' in df.columns:
        gated_impact = df.groupby('Gated_Community')['Price_Per_SqFt'].agg(['median', 'mean', 'count']).reset_index()
        gated_impact['Gated_Community'] = gated_impact['Gated_Community'].map({0: 'No', 1: 'Yes'})
        gated_impact['Premium_%'] = ((gated_impact['median'] / gated_impact['median'].min()) - 1) * 100
        amenities_analysis['Gated_Community'] = gated_impact
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Furnishing Impact
    axes[0, 0].bar(furnishing_impact['Furnishing_Status_Clean'], furnishing_impact['median'], color='skyblue', alpha=0.8)
    axes[0, 0].set_ylabel('Median Price per SqFt (₹)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Furnishing Status Impact', fontsize=14, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Property Type Impact
    axes[0, 1].bar(property_impact['Property_Type'], property_impact['median'], color='coral', alpha=0.8)
    axes[0, 1].set_ylabel('Median Price per SqFt (₹)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Property Type Impact', fontsize=14, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Vastu Impact
    if 'Vastu' in amenities_analysis:
        axes[1, 0].bar(vastu_impact['Vastu_Compliant'], vastu_impact['median'], color='lightgreen', alpha=0.8)
        axes[1, 0].set_ylabel('Median Price per SqFt (₹)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Vastu Compliance Impact', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Gated Community Impact
    if 'Gated_Community' in amenities_analysis:
        axes[1, 1].bar(gated_impact['Gated_Community'], gated_impact['median'], color='gold', alpha=0.8)
        axes[1, 1].set_ylabel('Median Price per SqFt (₹)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Gated Community Impact', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_plot('use_case_5_amenities_impact.png')
    
    return amenities_analysis

# ===== NEW CREATIVE INSIGHTS =====

def generate_undervalued_gems(df):
    """
    Identify localities with high 'Value Score' (Area/Price) but relatively low Price/SqFt.
    """
    print("\n[+] Generating Insight: Undervalued Gems...")
    df['Value_Score'] = (df['Area_SqFt_Clean'] / df['Price_Lakhs']) * 100
    
    stats = df.groupby('Locality').agg({
        'Price_Per_SqFt': 'median',
        'Value_Score': 'mean',
        'Price_Lakhs': 'count'
    }).reset_index()
    
    # Filter for sufficient data and "Mid-Segment" pricing
    city_median_price = df['Price_Per_SqFt'].median()
    candidates = stats[
        (stats['Price_Lakhs'] >= 5) & 
        (stats['Price_Per_SqFt'] < city_median_price * 1.2)
    ]
    
    # Top 10 by Value Score
    top_10 = candidates.nlargest(10, 'Value_Score')
    top_10['Insight_Type'] = 'Undervalued Gem'
    return top_10

def generate_luxury_havens(df):
    """
    Identify top luxury localities based on Price, Size, and Amenities.
    """
    print("\n[+] Generating Insight: Luxury Havens...")
    
    # Calculate a simple "Luxury Score" proxy
    # We don't have a direct amenities count, so we use Price, Area, and Gated Community presence
    stats = df.groupby('Locality').agg({
        'Price_Per_SqFt': 'median',
        'Area_SqFt_Clean': 'median',
        'Gated_Community': 'mean', # % of gated communities
        'Price_Lakhs': 'count'
    }).reset_index()
    
    stats['Luxury_Index'] = (
        (stats['Price_Per_SqFt'] / stats['Price_Per_SqFt'].max()) * 0.4 +
        (stats['Area_SqFt_Clean'] / stats['Area_SqFt_Clean'].max()) * 0.4 +
        stats['Gated_Community'] * 0.2
    )
    
    top_10 = stats[stats['Price_Lakhs'] >= 5].nlargest(10, 'Luxury_Index')
    top_10['Insight_Type'] = 'Luxury Haven'
    return top_10

def generate_bachelor_hotspots(df):
    """
    Identify areas with high density of 1BHKs and affordable rent proxy (low price).
    """
    print("\n[+] Generating Insight: Bachelor Hotspots...")
    
    # Calculate % of 1BHKs
    loc_stats = df.groupby('Locality').apply(lambda x: (x['BHK_Clean'] == 1).mean()).reset_index(name='One_BHK_Ratio')
    price_stats = df.groupby('Locality')['Price_Lakhs'].median().reset_index(name='Median_Price')
    
    merged = pd.merge(loc_stats, price_stats, on='Locality')
    
    # Filter: At least 20% 1BHKs
    hotspots = merged[merged['One_BHK_Ratio'] > 0.15].nlargest(10, 'One_BHK_Ratio')
    hotspots['Insight_Type'] = 'Bachelor Hotspot'
    return hotspots

def generate_family_sanctuaries(df):
    """
    Identify areas with high density of 3BHK+, Gated Communities, and moderate-to-high price.
    """
    print("\n[+] Generating Insight: Family Sanctuaries...")
    
    stats = df.groupby('Locality').agg({
        'BHK_Clean': lambda x: (x >= 3).mean(), # Ratio of 3BHK+
        'Gated_Community': 'mean',
        'Price_Lakhs': 'count'
    }).reset_index()
    
    stats.columns = ['Locality', 'Large_Apt_Ratio', 'Gated_Ratio', 'Supply']
    
    stats['Family_Score'] = (stats['Large_Apt_Ratio'] * 0.6) + (stats['Gated_Ratio'] * 0.4)
    
    top_10 = stats[stats['Supply'] >= 10].nlargest(10, 'Family_Score')
    top_10['Insight_Type'] = 'Family Sanctuary'
    return top_10



def main():
    print("\n" + "="*60)
    print("COMPREHENSIVE BUSINESS INSIGHTS ANALYSIS")
    print("="*60)
    
    df = load_data()
    df = remove_outliers(df) # Apply Noise Reduction
    setup_style()
    
    # Run all analyses
    locality_stats, dev_recommendations = analyze_development_opportunities(df)
    investment_zones = analyze_investment_zones(df)
    new_vs_resale = analyze_new_vs_resale(df)
    family_bachelor = analyze_family_vs_bachelor(df)
    amenities = analyze_amenities_impact(df)
    
    # Generate New Creative Insights
    undervalued = generate_undervalued_gems(df)
    luxury = generate_luxury_havens(df)
    bachelors = generate_bachelor_hotspots(df)
    family = generate_family_sanctuaries(df)
    
    # Save summary data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, 'outputs', 'reports')
    
    locality_stats.to_csv(os.path.join(reports_dir, 'locality_development_stats.csv'), index=False)
    investment_zones.to_csv(os.path.join(reports_dir, 'investment_zones.csv'), index=False)
    new_vs_resale.to_csv(os.path.join(reports_dir, 'new_vs_resale_comparison.csv'), index=False)
    family_bachelor.to_csv(os.path.join(reports_dir, 'family_bachelor_profile.csv'), index=False)
    
    # Save New Insights to data/insights/
    insights_dir = os.path.join(base_dir, 'data', 'insights')
    if not os.path.exists(insights_dir):
        os.makedirs(insights_dir)
        
    undervalued.to_csv(os.path.join(insights_dir, 'top_10_undervalued_localities.csv'), index=False)
    luxury.to_csv(os.path.join(insights_dir, 'top_10_luxury_localities.csv'), index=False)
    bachelors.to_csv(os.path.join(insights_dir, 'top_10_bachelor_hubs.csv'), index=False)
    family.to_csv(os.path.join(insights_dir, 'top_10_family_hubs.csv'), index=False)
    
    print(f"\n[+] Saved 4 NEW Creative Insight Reports to {insights_dir}")
    
    print("\n" + "="*60)
    print("SUCCESS! All business insights generated.")
    print("Outputs saved to: outputs/reports/ and outputs/figures/business/")
    print("="*60)

if __name__ == "__main__":
    main()
