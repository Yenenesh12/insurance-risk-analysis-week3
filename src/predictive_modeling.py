import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib

class InsurancePredictiveModeling:
    def __init__(self, data_path=r'C:\Users\admin\insurance-risk-analysis-week3\data\raw\insurance_data.csv'):
        """Initialize with insurance data"""
        self.df = pd.read_csv(data_path)
        print(f"Data loaded with shape: {self.df.shape}")
        
    def prepare_data_severity(self):
        """Prepare data for claim severity prediction (claims > 0)"""
        print("\nPreparing data for Claim Severity Prediction...")
        
        # Filter only policies with claims
        severity_df = self.df[self.df['TotalClaims'] > 0].copy()
        print(f"Severity dataset shape (claims only): {severity_df.shape}")
        
        # Target variable
        y = severity_df['TotalClaims']
        
        # Features for severity prediction
        severity_features = [
            'Province', 'VehicleType', 'Gender', 'Make', 'Model',
            'TotalPremium', 'CustomValueEstimate', 'SumInsured',
            'CalculatedPremiumPerTerm', 'CoverType', 'RegistrationYear',
            'NumberOfDoors', 'Bodytype', 'MaritalStatus'
        ]
        
        X = severity_df[severity_features].copy()
        
        # Feature engineering for severity
        X['vehicle_age'] = 2024 - X['RegistrationYear']  # Assuming current year is 2024
        X['premium_to_sum_insured_ratio'] = X['TotalPremium'] / (X['SumInsured'] + 1)
        X['value_estimate_ratio'] = X['CustomValueEstimate'] / (X['SumInsured'] + 1)
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        print(f"Categorical columns: {len(categorical_cols)}")
        print(f"Numerical columns: {len(numerical_cols)}")
        
        # Encode categorical variables
        self.label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        self.X_severity = pd.DataFrame(X_scaled, columns=X.columns)
        self.y_severity = y
        
        print(f"Final severity dataset shape: {self.X_severity.shape}")
        return self.X_severity, self.y_severity
    
    def prepare_data_premium(self):
        """Prepare data for premium prediction"""
        print("\nPreparing data for Premium Prediction...")
        
        # Use all data for premium prediction
        premium_df = self.df.copy()
        
        # Target variable - using TotalPremium
        y = premium_df['TotalPremium']
        
        # Features for premium prediction
        premium_features = [
            'Province', 'VehicleType', 'Gender', 'Make', 'Model',
            'CustomValueEstimate', 'SumInsured', 'CoverType',
            'RegistrationYear', 'NumberOfDoors', 'Bodytype',
            'MaritalStatus', 'TotalClaims'  
        ]
        
        X = premium_df[premium_features].copy()
        
        # Feature engineering
        X['vehicle_age'] = 2024 - X['RegistrationYear']
        X['has_claim'] = (premium_df['TotalClaims'] > 0).astype(int)
        X['claim_amount_ratio'] = premium_df['TotalClaims'] / (premium_df['TotalPremium'] + 1)
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        self.X_premium = pd.DataFrame(X_scaled, columns=X.columns)
        self.y_premium = y
        
        print(f"Final premium dataset shape: {self.X_premium.shape}")
        return self.X_premium, self.y_premium
    
    def prepare_data_claim_probability(self):
        """Prepare data for claim probability prediction"""
        print("\nPreparing data for Claim Probability Prediction...")
        
        claim_df = self.df.copy()
        
        # Target variable - binary classification for claim occurrence
        y = (claim_df['TotalClaims'] > 0).astype(int)
        
        # Features similar to premium prediction but excluding claim-related features
        claim_features = [
            'Province', 'VehicleType', 'Gender', 'Make', 'Model',
            'CustomValueEstimate', 'SumInsured', 'CoverType',
            'RegistrationYear', 'NumberOfDoors', 'Bodytype',
            'MaritalStatus', 'TotalPremium'
        ]
        
        X = claim_df[claim_features].copy()
        
        # Feature engineering
        X['vehicle_age'] = 2024 - X['RegistrationYear']
        X['premium_per_value'] = X['TotalPremium'] / (X['SumInsured'] + 1)
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        self.X_claim = pd.DataFrame(X_scaled, columns=X.columns)
        self.y_claim = y
        
        print(f"Final claim probability dataset shape: {self.X_claim.shape}")
        return self.X_claim, self.y_claim
    
    def build_severity_models(self):
        """Build and compare severity prediction models"""
        print("\n" + "="*60)
        print("CLAIM SEVERITY PREDICTION MODELS")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_severity, self.y_severity, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'R²': r2,
                'MAE': mae,
                'model': model
            }
            
            print(f"  RMSE: R{rmse:.2f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAE: R{mae:.2f}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T[['RMSE', 'R²', 'MAE']]
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(comparison_df.round(4))
        
        # Save best model
        best_model_name = comparison_df['R²'].idxmax()
        best_model = results[best_model_name]['model']
        
        joblib.dump(best_model, 'models/best_severity_model.pkl')
        print(f"\nBest model saved: {best_model_name}")
        
        return results, comparison_df
    
    def build_premium_models(self):
        """Build and compare premium prediction models"""
        print("\n" + "="*60)
        print("PREMIUM PREDICTION MODELS")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_premium, self.y_premium, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate percentage error
            percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            results[name] = {
                'RMSE': rmse,
                'R²': r2,
                'MAE': mae,
                'Mean % Error': percentage_error,
                'model': model
            }
            
            print(f"  RMSE: R{rmse:.2f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  Mean % Error: {percentage_error:.2f}%")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T[['RMSE', 'R²', 'MAE', 'Mean % Error']]
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(comparison_df.round(4))
        
        # Save best model
        best_model_name = comparison_df['R²'].idxmax()
        best_model = results[best_model_name]['model']
        
        joblib.dump(best_model, 'models/best_premium_model.pkl')
        print(f"\nBest model saved: {best_model_name}")
        
        return results, comparison_df
    
    def analyze_feature_importance(self, model, X, model_name):
        """Analyze feature importance using SHAP"""
        print(f"\nAnalyzing feature importance for {model_name}...")
        
        # Create SHAP explainer
        if hasattr(model, 'predict_proba'):
            explainer = shap.Explainer(model, X)
        else:
            explainer = shap.Explainer(model, X)
        
        # Calculate SHAP values
        shap_values = explainer(X)
        
        # Plot summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f'Feature Importance - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'results/shap_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Get top features
        shap_df = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        print(f"\nTop 10 Most Important Features for {model_name}:")
        print(shap_df.head(10).to_string(index=False))
        
        return shap_df
    
    def build_risk_based_pricing_framework(self):
        """Build comprehensive risk-based pricing framework"""
        print("\n" + "="*60)
        print("RISK-BASED PRICING FRAMEWORK")
        print("="*60)
        
        # Load or train models
        try:
            claim_prob_model = joblib.load(r'C:/Users/admin/insurance-risk-analysis-week3/models/best_claim_prob_model.pkl')
            severity_model = joblib.load(r'C:/Users/admin/insurance-risk-analysis-week3/models/best_severity_model.pkl')
        except:
            print("Training claim probability model...")
            # You would implement claim probability classification here
            claim_prob_model = None
            severity_model = None
        
        # Conceptual pricing formula
        print("\nPRICING FORMULA CONCEPT:")
        print("Premium = (Predicted Probability of Claim × Predicted Claim Severity) +")
        print("          Expense Loading + Profit Margin")
        
        # Calculate key metrics for pricing
        avg_claim_freq = self.df['has_claim'].mean() if 'has_claim' in self.df.columns else 0.1
        avg_severity = self.df['TotalClaims'].mean() if self.df['TotalClaims'].sum() > 0 else 10000
        avg_premium = self.df['TotalPremium'].mean()
        
        print(f"\nKey Pricing Metrics:")
        print(f"Average Claim Frequency: {avg_claim_freq:.4f}")
        print(f"Average Claim Severity: R{avg_severity:.2f}")
        print(f"Average Premium: R{avg_premium:.2f}")
        print(f"Expected Loss per Policy: R{avg_claim_freq * avg_severity:.2f}")
        
        # Create pricing recommendation table
        risk_segments = ['Low Risk', 'Medium Risk', 'High Risk']
        segment_multipliers = [0.7, 1.0, 1.5]  
        
        pricing_table = pd.DataFrame({
            'Risk Segment': risk_segments,
            'Risk Multiplier': segment_multipliers,
            'Base Premium (R)': [avg_premium * m for m in segment_multipliers],
            'Expected Claims (R)': [avg_claim_freq * avg_severity * m for m in segment_multipliers],
            'Expected Margin (R)': [avg_premium * m - avg_claim_freq * avg_severity * m for m in segment_multipliers]
        })
        
        print("\n" + "="*60)
        print("RISK-BASED PRICING RECOMMENDATIONS")
        print("="*60)
        print(pricing_table.round(2).to_string(index=False))
        
        # Save pricing framework
        pricing_table.to_csv(r'C:/Users/admin/insurance-risk-analysis-week3/results/risk_based_pricing_framework.csv', index=False)
        
        return pricing_table
    
    def evaluate_model_performance(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Build all models
        severity_results, severity_comparison = self.build_severity_models()
        premium_results, premium_comparison = self.build_premium_models()
        
        # Feature importance analysis for best models
        X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
            self.X_severity, self.y_severity, test_size=0.2, random_state=42
        )
        
        best_severity_model_name = severity_comparison['R²'].idxmax()
        best_severity_model = severity_results[best_severity_model_name]['model']
        
        # Train on full dataset for SHAP analysis
        best_severity_model.fit(self.X_severity, self.y_severity)
        
        # Analyze feature importance
        shap_df = self.analyze_feature_importance(
            best_severity_model, 
            self.X_severity, 
            f"{best_severity_model_name} - Severity Prediction"
        )
        
        # Business interpretation of top features
        print("\n" + "="*60)
        print("BUSINESS INTERPRETATION OF KEY FEATURES")
        print("="*60)
        
        top_features = shap_df.head(5)['feature'].tolist()
        interpretations = {
            'TotalPremium': "Higher premiums correlate with higher claim amounts, suggesting accurate risk assessment in current pricing.",
            'SumInsured': "Vehicles with higher sum insured have larger claim amounts, as expected for more valuable assets.",
            'CustomValueEstimate': "Custom valuation provides accurate risk assessment of vehicle value.",
            'vehicle_age': "Older vehicles may have higher claim amounts due to repair costs and safety features.",
            'CalculatedPremiumPerTerm': "Premium calculation reflects underlying risk factors accurately."
        }
        
        for feature in top_features:
            if feature in interpretations:
                print(f"\n{feature}:")
                print(f"  • {interpretations[feature]}")
                print(f"  • Impact: One standard deviation increase changes predicted claim by approximately R{shap_df[shap_df['feature']==feature]['mean_abs_shap'].values[0]:.2f}")
        
        # Generate final report
        self.generate_modeling_report(severity_comparison, premium_comparison, shap_df)
    
    def generate_modeling_report(self, severity_comparison, premium_comparison, shap_df):
        """Generate comprehensive modeling report"""
        print("\n" + "="*60)
        print("FINAL MODELING REPORT")
        print("="*60)
        
        print("\n1. CLAIM SEVERITY PREDICTION")
        print("Best Model:", severity_comparison['R²'].idxmax())
        print(f"Performance: R² = {severity_comparison['R²'].max():.4f}, RMSE = R{severity_comparison['RMSE'].min():.2f}")
        
        print("\n2. PREMIUM PREDICTION")
        print("Best Model:", premium_comparison['R²'].idxmax())
        print(f"Performance: R² = {premium_comparison['R²'].max():.4f}, RMSE = R{premium_comparison['RMSE'].min():.2f}")
        
        print("\n3. TOP 5 RISK DRIVERS (from SHAP analysis):")
        for i, (idx, row) in enumerate(shap_df.head(5).iterrows(), 1):
            print(f"   {i}. {row['feature']}: SHAP importance = {row['mean_abs_shap']:.4f}")
        
        print("\n4. RECOMMENDATIONS FOR RISK-BASED PRICING:")
        print("   • Implement the XGBoost model for severity prediction in underwriting")
        print("   • Use feature importance to refine rating factors")
        print("   • Develop risk segments based on top predictive features")
        print("   • Regularly retrain models with new data")
        print("   • Monitor model performance and feature stability over time")
        
        # Save results
        severity_comparison.to_csv(r'C:/Users/admin/insurance-risk-analysis-week3/results/severity_model_comparison.csv')
        premium_comparison.to_csv(r'C:/Users/admin/insurance-risk-analysis-week3/results/premium_model_comparison.csv')
        shap_df.to_csv('results/feature_importance_analysis.csv', index=False)
        
        print("\nResults saved to 'results/' directory")

# Main execution
if __name__ == "__main__":
    import os
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    # Initialize modeling
    modeler = InsurancePredictiveModeling(r'C:\Users\admin\insurance-risk-analysis-week3\data\raw\insurance_data.csv')
    
    # Prepare data for different modeling tasks
    modeler.prepare_data_severity()
    modeler.prepare_data_premium()
    modeler.prepare_data_claim_probability()
    
    # Build and evaluate models
    modeler.evaluate_model_performance()
    
    # Build risk-based pricing framework
    modeler.build_risk_based_pricing_framework()