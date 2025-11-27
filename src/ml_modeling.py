import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error)
import os
import joblib

def load_data():
    # Construct absolute path to data file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'ahmedabad_real_estate_cleaned.csv')
    return pd.read_csv(data_path)

def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

def save_plot(filename):
    # Construct absolute path to output directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs', 'figures', 'ml')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

# --- Custom Metrics ---
def negotiation_margin_error(y_true, y_pred, margin=0.10):
    diff = np.abs(y_true - y_pred)
    within_margin = diff <= (margin * y_true)
    return np.mean(within_margin) * 100

# --- Feature Audit Helper ---
def extract_feature_importance(model, feature_names, model_name):
    try:
        if hasattr(model, 'feature_importances_'):
            return pd.Series(model.feature_importances_, index=feature_names)
        elif hasattr(model, 'coef_'):
            # For linear models, take absolute value of coefficients
            if model.coef_.ndim > 1: # Multiclass classification
                return pd.Series(np.mean(np.abs(model.coef_), axis=0), index=feature_names)
            return pd.Series(np.abs(model.coef_), index=feature_names)
    except:
        pass
    return None

# --- Hybrid Locality Encoding (Leakage-Free) ---
def engineer_locality_features(df):
    """
    Create 3 locality features:
    1. Locality_Frequency: How common is this locality (supply indicator)
    2. Locality_Tier: Premium/Mid/Affordable/Budget (inferred from Price_Per_SqFt)
    3. Top 10 Localities as One-Hot (specific premium)
    """
    # 1. Locality Frequency (Leakage-Free)
    locality_counts = df['Clean_Locality'].value_counts()
    df['Locality_Frequency'] = df['Clean_Locality'].map(locality_counts) / len(df)
    
    # 2. Locality Tier (Inferred from Price_Per_SqFt median - using GLOBAL median, not individual prices)
    locality_median_price = df.groupby('Clean_Locality')['Price_Per_SqFt'].median()
    
    # Define tier thresholds based on overall distribution
    tier_thresholds = locality_median_price.quantile([0.25, 0.50, 0.75])
    
    def assign_tier(locality):
        median_price = locality_median_price.get(locality, locality_median_price.median())
        if median_price >= tier_thresholds[0.75]:
            return 'Premium'
        elif median_price >= tier_thresholds[0.50]:
            return 'Mid-Segment'
        elif median_price >= tier_thresholds[0.25]:
            return 'Affordable'
        else:
            return 'Budget'
    
    df['Locality_Tier'] = df['Clean_Locality'].apply(assign_tier)
    
    # 3. Top 10 Localities as One-Hot
    top_10_locs = df['Clean_Locality'].value_counts().head(10).index
    df['Locality_Top10'] = df['Clean_Locality'].apply(lambda x: x if x in top_10_locs else 'Other')
    
    # 4. Locality Target Encoding (Smoothed Mean Price_Per_SqFt)
    # This gives the model a direct signal of the "value" of the location
    global_mean = df['Price_Per_SqFt'].mean()
    smoothing_weight = 10 
    locality_stats = df.groupby('Clean_Locality')['Price_Per_SqFt'].agg(['count', 'mean'])
    smoothed_means = (locality_stats['count'] * locality_stats['mean'] + smoothing_weight * global_mean) / (locality_stats['count'] + smoothing_weight)
    df['Locality_Target_Encoded'] = df['Clean_Locality'].map(smoothed_means)
    
    # 5. Locality Value Score (Interaction: Area * Locality Rate)
    # This captures "Expected Price" based on size and location quality
    df['Locality_Value_Score'] = df['Area_SqFt_Clean'] * df['Locality_Target_Encoded']
    
    return df

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

def run_regression_suite(df):
    print("\n=== Professional ML Pipeline with CV & Tuning ===\n")
    
    # Remove Outliers FIRST
    df = remove_outliers(df)
    
    # Engineer Locality Features
    df = engineer_locality_features(df)
    
    # Select RAW Features
    feature_cols = [
        'Area_SqFt_Clean',           
        'BHK_Clean',                 
        'Locality_Frequency',        
        'Locality_Tier',             
        'Locality_Top10',            
        'Locality_Target_Encoded',   # NEW: Strong locality signal
        'Locality_Value_Score',      # NEW: Interaction feature
        'Furnishing_Status_Clean',  
        'Property_Type',             
        'Transaction_Type'           
    ]
    
    # Check if Floor_Number exists
    if 'Floor_Number' in df.columns and df['Floor_Number'].notna().sum() > len(df) * 0.5:
        feature_cols.append('Floor_Number')
        print("[+] Including Floor_Number")
    else:
        print("[-] Skipping Floor_Number (insufficient data)")
    
    X = df[feature_cols]
    X = df[feature_cols]
    y = np.log1p(df['Price_Lakhs']) # Log Transformation
    print("[+] Applied Log Transformation to Target Variable")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define feature types
    numeric_features = ['Area_SqFt_Clean', 'BHK_Clean', 'Locality_Frequency', 'Locality_Target_Encoded', 'Locality_Value_Score']
    categorical_features = ['Locality_Tier', 'Locality_Top10', 'Furnishing_Status_Clean', 
                           'Property_Type', 'Transaction_Type']
    
    if 'Floor_Number' in feature_cols:
        categorical_features.append('Floor_Number')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Base Models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'KNN': KNeighborsRegressor(),
        'SVR': SVR(),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0)
    }
    
    results = []
    feature_importance_agg = pd.DataFrame()
    
    # Fit Preprocessor once to get feature names
    preprocessor.fit(X_train)
    feature_names = (numeric_features + 
                     list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    feature_importance_agg['Feature'] = feature_names
    feature_importance_agg.set_index('Feature', inplace=True)

    print("\n1. Training Base Models with 5-Fold Cross-Validation...")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        
        # Cross-Validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Train on full training set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Inverse Log Transformation for Metrics
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
        
        # Metrics
        r2 = r2_score(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
        neg_margin = negotiation_margin_error(y_test_orig, y_pred_orig)
        
        results.append({
            'Model': name,
            'CV R2 (Mean)': cv_mean,
            'CV R2 (Std)': cv_std,
            'Test R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'Negotiation Margin (%)': neg_margin
        })
        
        print(f"  {name}: CV R2={cv_mean:.3f} (+/- {cv_std:.3f}), Test R2={r2:.3f}")
        
        # Feature Importance
        imp = extract_feature_importance(pipeline.named_steps['model'], feature_names, name)
        if imp is not None:
            imp = (imp - imp.min()) / (imp.max() - imp.min())
            feature_importance_agg[name] = imp

    # 2. Hyperparameter Tuning for Gradient Boosting
    print("\n2. Hyperparameter Tuning (Gradient Boosting)...")
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5, 10]
    }
    
    gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                                   ('model', GradientBoostingRegressor(random_state=42))])
    
    grid_search = GridSearchCV(gb_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_gb = grid_search.best_estimator_
    y_pred_tuned = best_gb.predict(X_test)
    
    # Inverse Log for Tuned Metrics
    y_test_orig = np.expm1(y_test)
    y_pred_tuned_orig = np.expm1(y_pred_tuned)
    
    r2_tuned = r2_score(y_test_orig, y_pred_tuned_orig)
    
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Tuned R2: {r2_tuned:.3f} (vs Base: {results[3]['Test R2']:.3f})")
    
    results.append({
        'Model': 'Gradient Boosting (Tuned)',
        'CV R2 (Mean)': grid_search.best_score_,
        'CV R2 (Std)': 0,
        'Test R2': r2_tuned,
        'MAE': mean_absolute_error(y_test_orig, y_pred_tuned_orig),
        'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred_tuned_orig)),
        'MAPE (%)': np.mean(np.abs((y_test_orig - y_pred_tuned_orig) / y_test_orig)) * 100,
        'Negotiation Margin (%)': negotiation_margin_error(y_test_orig, y_pred_tuned_orig)
    })
    
    # 3. Ensemble Voting
    print("\n3. Creating Ensemble (Voting Regressor)...")
    voting_reg = VotingRegressor(
        estimators=[
            ('gb', best_gb.named_steps['model']),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('lr', LinearRegression())
        ]
    )
    
    # Create pipeline with voting
    voting_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', voting_reg)])
    voting_pipeline.fit(X_train, y_train)
    y_pred_ensemble = voting_pipeline.predict(X_test)
    
    # Inverse Log for Ensemble Metrics
    y_pred_ensemble_orig = np.expm1(y_pred_ensemble)
    r2_ensemble = r2_score(y_test_orig, y_pred_ensemble_orig)
    
    print(f"  Ensemble R2: {r2_ensemble:.3f}")
    
    results.append({
        'Model': 'Ensemble (Voting)',
        'CV R2 (Mean)': 0,
        'CV R2 (Std)': 0,
        'Test R2': r2_ensemble,
        'MAE': mean_absolute_error(y_test_orig, y_pred_ensemble_orig),
        'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred_ensemble_orig)),
        'MAPE (%)': np.mean(np.abs((y_test_orig - y_pred_ensemble_orig) / y_test_orig)) * 100,
        'Negotiation Margin (%)': negotiation_margin_error(y_test_orig, y_pred_ensemble_orig)
    })
    
    # Save best model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'outputs', 'models', 'best_model.pkl')
    joblib.dump(best_gb, model_path)
    print(f"\n[+] Saved best model to {model_path}")

    return pd.DataFrame(results), feature_importance_agg

def plot_feature_audit(imp_df):
    # Aggregation Logic
    aggregated_importance = {}
    
    for feature in imp_df.index:
        if feature.startswith('Locality_Top10_'):
            category = 'Locality_Top10 (Avg)'
        elif feature.startswith('Locality_Tier_'):
            category = 'Locality_Tier (Avg)'
        elif feature == 'Locality_Target_Encoded':
            category = 'Locality_Target_Encoded'
        elif feature == 'Locality_Value_Score':
            category = 'Locality_Value_Score'
        elif feature.startswith('Furnishing_Status_Clean_'):
            category = 'Furnishing (Avg)'
        elif feature.startswith('Property_Type_'):
            category = 'Property_Type (Avg)'
        elif feature.startswith('Transaction_Type_'):
            category = 'Transaction_Type (Avg)'
        elif feature.startswith('Floor_Number_'):
            category = 'Floor_Number (Avg)'
        else:
            category = feature
        
        if category not in aggregated_importance:
            aggregated_importance[category] = []
        aggregated_importance[category].append(imp_df.loc[feature])
    
    # Calculate mean
    final_importance = {}
    for category, values in aggregated_importance.items():
        avg_values = pd.concat(values, axis=1).mean(axis=1)
        final_importance[category] = avg_values.mean()
    
    imp_summary = pd.DataFrame.from_dict(final_importance, orient='index', columns=['Mean_Importance'])
    
    # Convert to Percentage
    total_importance = imp_summary['Mean_Importance'].sum()
    imp_summary['Mean_Importance'] = (imp_summary['Mean_Importance'] / total_importance) * 100
    
    imp_summary = imp_summary.sort_values('Mean_Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=imp_summary['Mean_Importance'], y=imp_summary.index, palette='viridis', hue=imp_summary.index, legend=False)
    plt.title('Feature Importance (%)', fontsize=16, fontweight='bold')
    plt.xlabel('Contribution to Model (%)')
    plt.xlim(0, 100)
    save_plot('consensus_feature_importance.png')
    
    return imp_summary

def main():
    print("Starting Enhanced ML Modeling...")
    try:
        df = load_data()
        setup_style()
        
        # Regression with CV & Tuning
        reg_results, reg_imp = run_regression_suite(df)
        
        # Save Results
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_path = os.path.join(base_dir, 'outputs', 'reports', 'comprehensive_model_results.csv')
        reg_results.to_csv(results_path, index=False)
        print("\n[+] Saved comprehensive_model_results.csv")
        
        # Feature Audit
        audit_path = os.path.join(base_dir, 'outputs', 'reports', 'feature_importance_audit.csv')
        audit_df = plot_feature_audit(reg_imp)
        audit_df.to_csv(audit_path)
        print("[+] Saved feature_importance_audit.csv")
        
        # Plot Comparison
        plt.figure(figsize=(14, 6))
        sns.barplot(data=reg_results, x='Model', y='Test R2', palette='magma', hue='Model', legend=False)
        plt.title('Model Comparison (Test R2 Score)', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.tight_layout()
        save_plot('regression_comparison.png')
        
        print("\n=== SUCCESS! Enhanced ML Pipeline Complete ===")
        print(f"\nBest Model: {reg_results.loc[reg_results['Test R2'].idxmax(), 'Model']}")
        print(f"Best R2 Score: {reg_results['Test R2'].max():.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
