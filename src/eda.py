import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from math import pi

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
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = 'sans-serif'
    # Use a custom color palette
    sns.set_palette("husl")

def save_plot(filename):
    # Construct absolute path to output directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs', 'figures', 'eda')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {filename}")

# --- A. Market Structure ---

def plot_affordability_pyramid(df):
    # Segment data
    bins = [0, 40, 80, 150, 5000]
    labels = ['Budget (<40L)', 'Mid-Segment (40-80L)', 'Premium (80L-1.5Cr)', 'Luxury (>1.5Cr)']
    df['Price_Segment'] = pd.cut(df['Price_Lakhs'], bins=bins, labels=labels)
    
    counts = df['Price_Segment'].value_counts().sort_index(ascending=False)
    
    plt.figure(figsize=(10, 8))
    colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#8B4513'][::-1] # Gold, Silver, Bronze, Brown
    plt.barh(counts.index, counts.values, color=colors, edgecolor='black')
    plt.title('The Affordability Pyramid: Market Segmentation', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Listings')
    
    for i, v in enumerate(counts.values):
        plt.text(v + 10, i, str(v), va='center', fontweight='bold')
        
    save_plot('affordability_pyramid.png')

def plot_market_depth(df):
    # Top 15 localities by volume
    top_locs = df['Locality'].value_counts().head(15).index
    subset = df[df['Locality'].isin(top_locs)]
    
    # Aggregates
    agg = subset.groupby('Locality').agg({'Price_Lakhs': 'mean', 'Property_Title': 'count'}).sort_values('Property_Title', ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Bar plot for Volume
    sns.barplot(x=agg.index, y=agg['Property_Title'], ax=ax1, color='skyblue', alpha=0.6, label='Volume (Listings)')
    ax1.set_ylabel('Listing Volume', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Line plot for Price
    ax2 = ax1.twinx()
    sns.lineplot(x=agg.index, y=agg['Price_Lakhs'], ax=ax2, color='crimson', marker='o', linewidth=2, label='Avg Price (Lakhs)')
    ax2.set_ylabel('Avg Price (Lakhs)', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')
    
    plt.title('Market Depth: Supply Volume vs Average Price', fontsize=16, fontweight='bold')
    save_plot('market_depth.png')

def plot_locality_tier_boxen(df):
    plt.figure(figsize=(12, 8))
    order = ['Affordable', 'Mid-Segment', 'Premium', 'Luxury']
    sns.boxenplot(data=df, x='Locality_Tier', y='Price_Per_SqFt', order=order, palette='Spectral')
    plt.title('Price/SqFt Distribution by Locality Tier', fontsize=16, fontweight='bold')
    plt.ylim(0, 15000)
    save_plot('locality_tier_boxen.png')

# --- B. Buyer Psychology ---

def plot_upgrade_cost(df):
    # Filter for standard BHKs
    subset = df[df['BHK_Clean'].isin([1, 2, 3, 4])]
    avg_prices = subset.groupby('BHK_Clean')['Price_Lakhs'].median()
    
    plt.figure(figsize=(10, 6))
    plt.plot(avg_prices.index, avg_prices.values, marker='o', markersize=10, linewidth=3, color='purple')
    
    # Annotate jumps
    for i in range(len(avg_prices) - 1):
        bhk_curr = avg_prices.index[i]
        bhk_next = avg_prices.index[i+1]
        price_curr = avg_prices.values[i]
        price_next = avg_prices.values[i+1]
        diff = price_next - price_curr
        
        plt.annotate(f"+{diff:.0f}L", 
                     xy=((bhk_curr + bhk_next)/2, (price_curr + price_next)/2),
                     xytext=(0, 10), textcoords='offset points', ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8))

    plt.title('The "Upgrade Cost": Price Jump per Bedroom', fontsize=16, fontweight='bold')
    plt.xlabel('BHK')
    plt.ylabel('Median Price (Lakhs)')
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, linestyle='--')
    save_plot('upgrade_cost_curve.png')

def plot_seller_psychology(df):
    plt.figure(figsize=(10, 6))
    # Filter main seller types
    subset = df[df['Seller_Type'].isin(['Owner', 'Agent', 'Builder'])]
    sns.violinplot(data=subset, x='Seller_Type', y='Price_Per_SqFt', palette='muted', inner='quartile')
    plt.title('Seller Psychology: Who Offers Better Rates?', fontsize=16, fontweight='bold')
    plt.ylim(0, 12000)
    save_plot('seller_psychology.png')

# --- C. Investment Analysis ---

def plot_deal_hunter_matrix(df):
    plt.figure(figsize=(12, 8))
    # Filter for mid-segment and premium for better view
    subset = df[df['Locality_Tier'].isin(['Mid-Segment', 'Premium'])]
    
    # Scatter
    sns.scatterplot(data=subset, x='Locality', y='Price_Per_SqFt', hue='Price_Fairness', 
                    palette={'Undervalued (Great Deal)': 'green', 'Fair Value': 'grey', 'Overvalued (Premium)': 'red'},
                    alpha=0.6)
    
    plt.title('The "Deal Hunter" Matrix: Spotting Undervalued Gems', fontsize=16, fontweight='bold')
    plt.xticks([]) # Hide locality names as there are too many
    plt.xlabel('Localities (Anonymized for Clarity)')
    plt.ylabel('Price per SqFt')
    save_plot('deal_hunter_matrix.png')

def plot_transaction_premium(df):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Transaction_Type', y='Price_Per_SqFt', estimator='mean', palette='Paired')
    plt.title('New vs Resale: The "Freshness" Premium', fontsize=16, fontweight='bold')
    save_plot('transaction_premium.png')

def plot_emi_heatmap(df):
    # Create pivot table: Avg Price by BHK and Tier
    pivot = df.pivot_table(index='Locality_Tier', columns='BHK_Clean', values='Price_Lakhs', aggfunc='median')
    pivot = pivot[[1, 2, 3, 4]] # Reorder columns
    pivot = pivot.reindex(['Affordable', 'Mid-Segment', 'Premium', 'Luxury']) # Reorder rows
    
    # Calculate approx EMI (Rule of thumb: 850 per Lakh for 20 yrs @ 8.5%)
    emi_pivot = pivot * 850 
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(emi_pivot, annot=True, fmt='.0f', cmap='YlGnBu', cbar_kws={'label': 'Est. Monthly EMI (₹)'})
    plt.title('Affordability Heatmap: Estimated EMI by Segment', fontsize=16, fontweight='bold')
    save_plot('emi_heatmap.png')

# --- D. Product & Lifestyle ---

def plot_space_efficiency(df):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='Space_Efficiency', y='Price_Lakhs', hue='BHK_Clean', palette='deep', size='Luxury_Score', sizes=(20, 200), alpha=0.7)
    plt.title('Space Efficiency: Do Spacious Layouts Cost More?', fontsize=16, fontweight='bold')
    plt.xlabel('Space Efficiency (SqFt per Bedroom)')
    plt.ylabel('Price (Lakhs)')
    plt.xlim(100, 1000)
    plt.ylim(0, 300)
    save_plot('space_efficiency.png')

def plot_amenities_radar(df):
    # Avg Luxury Score by Tier
    tiers = ['Affordable', 'Mid-Segment', 'Premium', 'Luxury']
    scores = df.groupby('Locality_Tier')['Luxury_Score'].mean().reindex(tiers).values
    
    # Radar Chart setup
    angles = np.linspace(0, 2*np.pi, len(tiers), endpoint=False).tolist()
    scores = np.concatenate((scores, [scores[0]]))
    angles += angles[:1]
    tiers += tiers[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='gold', alpha=0.25)
    ax.plot(angles, scores, color='gold', linewidth=2)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tiers[:-1], fontsize=12, fontweight='bold')
    
    plt.title('Amenities Radar: Luxury Score by Market Segment', fontsize=16, fontweight='bold', y=1.1)
    save_plot('amenities_radar.png')

def plot_vastu_impact_enhanced(df):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['Vastu_Compliant']==1], x='Price_Per_SqFt', fill=True, color='green', label='Vastu Compliant')
    sns.kdeplot(data=df[df['Vastu_Compliant']==0], x='Price_Per_SqFt', fill=True, color='grey', label='Non-Compliant')
    plt.title('Vastu Impact: Price Density Comparison', fontsize=16, fontweight='bold')
    plt.xlim(0, 15000)
    plt.legend()
    save_plot('vastu_impact_enhanced.png')

def plot_furnishing_roi(df):
    plt.figure(figsize=(10, 6))
    # Calculate median price per sqft
    medians = df.groupby('Furnishing_Status_Clean')['Price_Per_SqFt'].median().sort_values()
    
    # Bar plot
    bars = plt.bar(medians.index, medians.values, color=['#e74c3c', '#f1c40f', '#2ecc71'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'₹{int(height)}',
                 ha='center', va='bottom', fontweight='bold')
                 
    plt.title('Furnishing ROI: Median Price/SqFt', fontsize=16, fontweight='bold')
    save_plot('furnishing_roi.png')

# --- E. Advanced ---
def plot_feature_importance(df):
    # Correlation with Price (Streamlined)
    # Only showing correlation for numeric features available
    cols = ['Area_SqFt_Clean', 'BHK_Clean', 'Luxury_Score', 'Carpet_Area_Est']
    # Note: Locality is categorical, so not in simple corr. 
    # Luxury_Score and Carpet_Area are derived but good for EDA context even if not in ML model.
    # If we want strictly what's in ML, it's just Area and BHK.
    # Let's keep Luxury_Score in EDA for context as it was interesting, but note it's not in ML.
    
def plot_feature_importance(df):
    # Correlation with Price (Streamlined)
    
    # 1. Encode Locality to measure its strength
    # We map each locality to its median price to see how strong the "Location Effect" is
    locality_prices = df.groupby('Locality')['Price_Lakhs'].median()
    df['Locality_Strength'] = df['Locality'].map(locality_prices)
    
    # 2. Select features to correlate
    cols = ['Area_SqFt_Clean', 'BHK_Clean', 'Luxury_Score', 'Locality_Strength']
    labels = ['Area (Size)', 'BHK (Config)', 'Luxury Score', 'Locality (Location)']
    
    # 3. Calculate Correlation
    corr = df[cols].apply(lambda x: x.corr(df['Price_Lakhs']))
    corr.index = labels # Rename for better chart
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=corr.values, y=corr.index, palette='Blues_d')
    plt.title('Feature Correlation with Price: What Matters Most?', fontsize=16, fontweight='bold')
    plt.xlabel('Correlation Coefficient (0 to 1)')
    save_plot('feature_importance.png')

def plot_pairplot_drivers(df):
    cols = ['Price_Lakhs', 'Area_SqFt_Clean', 'Luxury_Score', 'Locality_Tier']
    # Sample for speed if needed, but 2700 is fine
    sns.pairplot(df[cols], hue='Locality_Tier', palette='husl', height=2.5)
    save_plot('pairplot_drivers.png')
    
def plot_price_distribution_enhanced(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Price_Lakhs', hue='Locality_Tier', multiple='stack', bins=50, palette='Spectral')
    plt.title('Price Distribution Stacked by Locality Tier', fontsize=16, fontweight='bold')
    plt.xlim(0, 300)
    save_plot('price_dist_enhanced.png')

def main():
    print("Starting Advanced Creative EDA...")
    try:
        df = load_data()
        df = remove_outliers(df) # Apply Noise Reduction
        # df = engineer_features(df) # REMOVED
        setup_style()
        
        # A. Market Structure
        plot_affordability_pyramid(df)
        # plot_market_depth(df) # REMOVED per user request
        plot_locality_tier_boxen(df)
        
        # B. Buyer Psychology
        plot_upgrade_cost(df)
        plot_seller_psychology(df)
        
        # C. Investment
        plot_deal_hunter_matrix(df)
        # plot_transaction_premium(df) # REMOVED per user request
        plot_emi_heatmap(df)
        
        # D. Product
        plot_space_efficiency(df)
        plot_amenities_radar(df)
        plot_vastu_impact_enhanced(df)
        plot_furnishing_roi(df)
        # plot_extra_bathrooms_impact(df) # REMOVED
        
        # E. Advanced
        plot_feature_importance(df)
        # plot_pairplot_drivers(df) # REMOVED per user request
        plot_price_distribution_enhanced(df)
        
        print("Success! 15 Creative Visualizations saved to outputs/figures/eda/")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
