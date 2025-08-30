import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, 
    f_classif, 
    chi2, 
    mutual_info_classif,
    SelectPercentile
)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class StatisticalFeatureSelector:
    """
    Statistical Feature Selection for Sales Prediction Dataset
    
    This class implements statistical methods for feature selection:
    - ANOVA F-test for numerical features
    - Chi-square test for categorical features
    - Mutual Information for both feature types
    
    The target variable is 'SalesSuccess' with categories:
    ['Top Seller', 'Good Seller', 'Moderate Seller', 'Slow Seller']
    """
    
    def __init__(self, target_column='SalesSuccess'):
        """
        Initialize the feature selector
        
        Args:
            target_column (str): Name of the target variable column
        """
        self.target_column = target_column
        self.numerical_features = []
        self.categorical_features = []
        self.selected_features = {}
        self.feature_scores = {}
        self.label_encoder = LabelEncoder()
        self.encoded_target = None
        
    def identify_feature_types(self, df):
        """
        Automatically identify numerical and categorical features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (numerical_features, categorical_features)
        """
        # Exclude target column and ID columns
        feature_columns = [col for col in df.columns 
                          if col != self.target_column 
                          and not col.endswith('ID')
                          and col not in ['ProductName', 'Month']]
        
        numerical_features = []
        categorical_features = []
        
        for col in feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (few unique values)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 20:
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                categorical_features.append(col)
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        print(f"Identified {len(numerical_features)} numerical features:")
        print(f"  {numerical_features}")
        print(f"\nIdentified {len(categorical_features)} categorical features:")
        print(f"  {categorical_features}")
        
        return numerical_features, categorical_features
    
    def prepare_target_variable(self, df):
        """
        Encode target variable for statistical tests
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            np.array: Encoded target variable
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe")
        
        # Encode target variable
        self.encoded_target = self.label_encoder.fit_transform(df[self.target_column])
        
        print(f"\nTarget variable encoding:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  {class_name} -> {i}")
        
        return self.encoded_target
    
    def anova_f_test(self, df, k_best=10):
        """
        Perform ANOVA F-test for numerical features
        
        Args:
            df (pd.DataFrame): Input dataframe
            k_best (int): Number of best features to select
            
        Returns:
            dict: Selected features with their F-scores and p-values
        """
        if not self.numerical_features:
            print("No numerical features found for ANOVA F-test")
            return {}
        
        print(f"\n=== ANOVA F-Test for Numerical Features ===")
        print(f"Testing {len(self.numerical_features)} numerical features")
        
        # Prepare data
        X_numerical = df[self.numerical_features].fillna(df[self.numerical_features].median())
        y = self.encoded_target
        
        # Perform F-test
        selector = SelectKBest(score_func=f_classif, k=min(k_best, len(self.numerical_features)))
        X_selected = selector.fit_transform(X_numerical, y)
        
        # Get scores and p-values
        feature_scores = selector.scores_
        p_values = selector.pvalues_
        selected_indices = selector.get_support(indices=True)
        
        # Create results dictionary
        anova_results = {}
        for i, feature_idx in enumerate(selected_indices):
            feature_name = self.numerical_features[feature_idx]
            anova_results[feature_name] = {
                'f_score': feature_scores[feature_idx],
                'p_value': p_values[feature_idx],
                'rank': i + 1
            }
        
        # Store results
        self.selected_features['anova_numerical'] = list(anova_results.keys())
        self.feature_scores['anova'] = anova_results
        
        # Print results
        print(f"\nTop {len(anova_results)} numerical features (ANOVA F-test):")
        for feature, stats in sorted(anova_results.items(), 
                                   key=lambda x: x[1]['f_score'], reverse=True):
            print(f"  {feature:25} | F-score: {stats['f_score']:.4f} | p-value: {stats['p_value']:.6f}")
        
        return anova_results
    
    def chi_square_test(self, df, k_best=10):
        """
        Perform Chi-square test for categorical features
        
        Args:
            df (pd.DataFrame): Input dataframe
            k_best (int): Number of best features to select
            
        Returns:
            dict: Selected features with their Chi-square scores and p-values
        """
        if not self.categorical_features:
            print("No categorical features found for Chi-square test")
            return {}
        
        print(f"\n=== Chi-Square Test for Categorical Features ===")
        print(f"Testing {len(self.categorical_features)} categorical features")
        
        # Prepare categorical data
        X_categorical = df[self.categorical_features].copy()
        
        # Encode categorical features
        label_encoders = {}
        for col in self.categorical_features:
            le = LabelEncoder()
            X_categorical[col] = le.fit_transform(X_categorical[col].astype(str).fillna('Unknown'))
            label_encoders[col] = le
        
        y = self.encoded_target
        
        # Perform Chi-square test
        selector = SelectKBest(score_func=chi2, k=min(k_best, len(self.categorical_features)))
        X_selected = selector.fit_transform(X_categorical, y)
        
        # Get scores and p-values
        feature_scores = selector.scores_
        p_values = selector.pvalues_
        selected_indices = selector.get_support(indices=True)
        
        # Create results dictionary
        chi2_results = {}
        for i, feature_idx in enumerate(selected_indices):
            feature_name = self.categorical_features[feature_idx]
            chi2_results[feature_name] = {
                'chi2_score': feature_scores[feature_idx],
                'p_value': p_values[feature_idx],
                'rank': i + 1
            }
        
        # Store results
        self.selected_features['chi2_categorical'] = list(chi2_results.keys())
        self.feature_scores['chi2'] = chi2_results
        
        # Print results
        print(f"\nTop {len(chi2_results)} categorical features (Chi-square test):")
        for feature, stats in sorted(chi2_results.items(), 
                                   key=lambda x: x[1]['chi2_score'], reverse=True):
            print(f"  {feature:25} | Chi2-score: {stats['chi2_score']:.4f} | p-value: {stats['p_value']:.6f}")
        
        return chi2_results
    
    def mutual_information_test(self, df, k_best=15):
        """
        Perform Mutual Information test for all features
        
        Args:
            df (pd.DataFrame): Input dataframe
            k_best (int): Number of best features to select
            
        Returns:
            dict: Selected features with their MI scores
        """
        print(f"\n=== Mutual Information Test for All Features ===")
        
        all_features = self.numerical_features + self.categorical_features
        print(f"Testing {len(all_features)} total features")
        
        # Prepare data
        X_all = df[all_features].copy()
        
        # Encode categorical features for MI
        for col in self.categorical_features:
            if col in X_all.columns:
                le = LabelEncoder()
                X_all[col] = le.fit_transform(X_all[col].astype(str).fillna('Unknown'))
        
        # Fill missing values
        X_all = X_all.fillna(X_all.median())
        y = self.encoded_target
        
        # Perform Mutual Information test
        mi_scores = mutual_info_classif(X_all, y, random_state=42)
        
        # Create results dictionary
        mi_results = {}
        feature_mi_pairs = list(zip(all_features, mi_scores))
        feature_mi_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature_name, mi_score) in enumerate(feature_mi_pairs[:k_best]):
            mi_results[feature_name] = {
                'mi_score': mi_score,
                'rank': i + 1,
                'feature_type': 'numerical' if feature_name in self.numerical_features else 'categorical'
            }
        
        # Store results
        self.selected_features['mutual_info'] = list(mi_results.keys())
        self.feature_scores['mutual_info'] = mi_results
        
        # Print results
        print(f"\nTop {len(mi_results)} features (Mutual Information):")
        for feature, stats in mi_results.items():
            feature_type = stats['feature_type']
            print(f"  {feature:25} | MI-score: {stats['mi_score']:.4f} | Type: {feature_type}")
        
        return mi_results
    
    def get_combined_feature_ranking(self, weights=None):
        """
        Combine rankings from all statistical tests
        
        Args:
            weights (dict): Weights for different methods
                          {'anova': 0.4, 'chi2': 0.4, 'mutual_info': 0.2}
        
        Returns:
            dict: Combined feature rankings
        """
        if weights is None:
            weights = {'anova': 0.4, 'chi2': 0.4, 'mutual_info': 0.2}
        
        print(f"\n=== Combined Feature Ranking ===")
        print(f"Weights: {weights}")
        
        # Collect all features and their scores
        all_features = set()
        for method_features in self.selected_features.values():
            all_features.update(method_features)
        
        combined_scores = {}
        
        for feature in all_features:
            score = 0
            methods_count = 0
            
            # ANOVA score (for numerical features)
            if feature in self.feature_scores.get('anova', {}):
                # Normalize F-score (higher is better)
                max_f_score = max([s['f_score'] for s in self.feature_scores['anova'].values()])
                normalized_score = self.feature_scores['anova'][feature]['f_score'] / max_f_score
                score += weights['anova'] * normalized_score
                methods_count += 1
            
            # Chi-square score (for categorical features)
            if feature in self.feature_scores.get('chi2', {}):
                # Normalize Chi2-score (higher is better)
                max_chi2_score = max([s['chi2_score'] for s in self.feature_scores['chi2'].values()])
                normalized_score = self.feature_scores['chi2'][feature]['chi2_score'] / max_chi2_score
                score += weights['chi2'] * normalized_score
                methods_count += 1
            
            # Mutual Information score
            if feature in self.feature_scores.get('mutual_info', {}):
                # Normalize MI-score (higher is better)
                max_mi_score = max([s['mi_score'] for s in self.feature_scores['mutual_info'].values()])
                if max_mi_score > 0:
                    normalized_score = self.feature_scores['mutual_info'][feature]['mi_score'] / max_mi_score
                    score += weights['mutual_info'] * normalized_score
                    methods_count += 1
            
            # Average score across applicable methods
            if methods_count > 0:
                combined_scores[feature] = score / methods_count
        
        # Sort by combined score
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create final ranking
        combined_ranking = {}
        for i, (feature, score) in enumerate(sorted_features):
            feature_type = 'numerical' if feature in self.numerical_features else 'categorical'
            combined_ranking[feature] = {
                'combined_score': score,
                'rank': i + 1,
                'feature_type': feature_type
            }
        
        self.selected_features['combined'] = list(combined_ranking.keys())
        self.feature_scores['combined'] = combined_ranking
        
        # Print results
        print(f"\nTop {min(15, len(combined_ranking))} features (Combined Ranking):")
        for i, (feature, stats) in enumerate(list(combined_ranking.items())[:15]):
            feature_type = stats['feature_type']
            print(f"  {i+1:2d}. {feature:25} | Score: {stats['combined_score']:.4f} | Type: {feature_type}")
        
        return combined_ranking
    
    def perform_complete_feature_selection(self, df, k_numerical=10, k_categorical=10, k_combined=15):
        """
        Perform complete feature selection pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            k_numerical (int): Number of numerical features to select
            k_categorical (int): Number of categorical features to select
            k_combined (int): Number of features in combined ranking
            
        Returns:
            dict: Complete feature selection results
        """
        print("=" * 80)
        print("STATISTICAL FEATURE SELECTION FOR SALES PREDICTION")
        print("=" * 80)
        
        # Step 1: Identify feature types
        self.identify_feature_types(df)
        
        # Step 2: Prepare target variable
        self.prepare_target_variable(df)
        
        # Step 3: ANOVA F-test for numerical features
        anova_results = self.anova_f_test(df, k_best=k_numerical)
        
        # Step 4: Chi-square test for categorical features
        chi2_results = self.chi_square_test(df, k_best=k_categorical)
        
        # Step 5: Mutual Information for all features
        mi_results = self.mutual_information_test(df, k_best=k_combined)
        
        # Step 6: Combined ranking
        combined_results = self.get_combined_feature_ranking()
        
        # Summary
        print(f"\n=== FEATURE SELECTION SUMMARY ===")
        print(f"Total features analyzed: {len(self.numerical_features) + len(self.categorical_features)}")
        print(f"  - Numerical features: {len(self.numerical_features)}")
        print(f"  - Categorical features: {len(self.categorical_features)}")
        print(f"\nSelected features:")
        print(f"  - ANOVA F-test: {len(anova_results)} numerical features")
        print(f"  - Chi-square test: {len(chi2_results)} categorical features")
        print(f"  - Mutual Information: {len(mi_results)} total features")
        print(f"  - Combined ranking: {len(combined_results)} total features")
        
        return {
            'anova': anova_results,
            'chi2': chi2_results,
            'mutual_info': mi_results,
            'combined': combined_results,
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores
        }
    
    def save_results(self, output_path='feature_selection_results.csv'):
        """
        Save feature selection results to CSV
        
        Args:
            output_path (str): Path to save results
        """
        if not self.feature_scores:
            print("No feature selection results to save. Run feature selection first.")
            return
        
        # Prepare results for saving
        results_data = []
        
        # Combined results (most comprehensive)
        if 'combined' in self.feature_scores:
            for feature, stats in self.feature_scores['combined'].items():
                results_data.append({
                    'feature': feature,
                    'method': 'combined',
                    'score': stats['combined_score'],
                    'rank': stats['rank'],
                    'feature_type': stats['feature_type']
                })
        
        # ANOVA results
        if 'anova' in self.feature_scores:
            for feature, stats in self.feature_scores['anova'].items():
                results_data.append({
                    'feature': feature,
                    'method': 'anova_f_test',
                    'score': stats['f_score'],
                    'rank': stats['rank'],
                    'feature_type': 'numerical',
                    'p_value': stats['p_value']
                })
        
        # Chi-square results
        if 'chi2' in self.feature_scores:
            for feature, stats in self.feature_scores['chi2'].items():
                results_data.append({
                    'feature': feature,
                    'method': 'chi_square_test',
                    'score': stats['chi2_score'],
                    'rank': stats['rank'],
                    'feature_type': 'categorical',
                    'p_value': stats['p_value']
                })
        
        # Mutual Information results
        if 'mutual_info' in self.feature_scores:
            for feature, stats in self.feature_scores['mutual_info'].items():
                results_data.append({
                    'feature': feature,
                    'method': 'mutual_information',
                    'score': stats['mi_score'],
                    'rank': stats['rank'],
                    'feature_type': stats['feature_type']
                })
        
        # Save to CSV
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_path, index=False)
        print(f"\nFeature selection results saved to: {output_path}")
        
        return results_df

# Example usage and testing
if __name__ == "__main__":
    # This will be used for testing the feature selection module
    print("Statistical Feature Selection Module")
    print("This module provides ANOVA F-test, Chi-square test, and Mutual Information")
    print("for feature selection in the sales prediction dataset.")
    
    # Example of how to use:
    # selector = StatisticalFeatureSelector()
    # results = selector.perform_complete_feature_selection(df)
    # selector.save_results('feature_selection_results.csv')