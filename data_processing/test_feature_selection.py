import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_selection import StatisticalFeatureSelector
from generate_data import generate_products, generate_stores, generate_sales_data

def create_sample_data(num_products=500, num_stores=10, num_months=6):
    """
    Create sample sales data for testing feature selection
    
    Args:
        num_products (int): Number of products to generate
        num_stores (int): Number of stores to generate
        num_months (int): Number of months of sales data
        
    Returns:
        pd.DataFrame: Sample sales data
    """
    print(f"Creating sample data: {num_products} products, {num_stores} stores, {num_months} months")
    
    # Generate sample data
    products_df = generate_products(num_products)
    stores_df = generate_stores(num_stores)
    sales_df = generate_sales_data(products_df, stores_df, num_months)
    
    print(f"Sample data created: {sales_df.shape[0]} rows, {sales_df.shape[1]} columns")
    print(f"Target variable distribution:")
    print(sales_df['SalesSuccess'].value_counts())
    
    return sales_df

def test_feature_identification(selector, df):
    """
    Test feature type identification
    
    Args:
        selector: StatisticalFeatureSelector instance
        df: Sample dataframe
    """
    print("\n" + "="*60)
    print("TEST 1: FEATURE TYPE IDENTIFICATION")
    print("="*60)
    
    numerical_features, categorical_features = selector.identify_feature_types(df)
    
    # Verify expected feature types
    expected_numerical = ['UnitsSold', 'TotalRevenue', 'Price', 'CompetitorPrice']
    expected_categorical = ['StoreLocation', 'StoreSize', 'ProductCategory', 
                          'ProductSubCategory', 'Brand', 'IsLocal', 'OnPromotion',
                          'PromotionType', 'CustomerSegment', 'ShelfPlacement']
    
    print(f"\nValidation Results:")
    print(f"Expected numerical features found: {len([f for f in expected_numerical if f in numerical_features])}/{len(expected_numerical)}")
    print(f"Expected categorical features found: {len([f for f in expected_categorical if f in categorical_features])}/{len(expected_categorical)}")
    
    return True

def test_target_encoding(selector, df):
    """
    Test target variable encoding
    
    Args:
        selector: StatisticalFeatureSelector instance
        df: Sample dataframe
    """
    print("\n" + "="*60)
    print("TEST 2: TARGET VARIABLE ENCODING")
    print("="*60)
    
    encoded_target = selector.prepare_target_variable(df)
    
    # Verify encoding
    unique_encoded = np.unique(encoded_target)
    unique_original = df['SalesSuccess'].unique()
    
    print(f"\nValidation Results:")
    print(f"Original classes: {len(unique_original)}")
    print(f"Encoded classes: {len(unique_encoded)}")
    print(f"Encoding successful: {len(unique_encoded) == len(unique_original)}")
    
    return len(unique_encoded) == len(unique_original)

def test_anova_f_test(selector, df):
    """
    Test ANOVA F-test for numerical features
    
    Args:
        selector: StatisticalFeatureSelector instance
        df: Sample dataframe
    """
    print("\n" + "="*60)
    print("TEST 3: ANOVA F-TEST FOR NUMERICAL FEATURES")
    print("="*60)
    
    anova_results = selector.anova_f_test(df, k_best=5)
    
    # Validate results
    print(f"\nValidation Results:")
    print(f"Number of features selected: {len(anova_results)}")
    print(f"All F-scores positive: {all(stats['f_score'] > 0 for stats in anova_results.values())}")
    print(f"All p-values between 0 and 1: {all(0 <= stats['p_value'] <= 1 for stats in anova_results.values())}")
    
    # Check if results are reasonable
    if anova_results:
        best_feature = max(anova_results.items(), key=lambda x: x[1]['f_score'])
        print(f"Best numerical feature: {best_feature[0]} (F-score: {best_feature[1]['f_score']:.4f})")
    
    return len(anova_results) > 0

def test_chi_square_test(selector, df):
    """
    Test Chi-square test for categorical features
    
    Args:
        selector: StatisticalFeatureSelector instance
        df: Sample dataframe
    """
    print("\n" + "="*60)
    print("TEST 4: CHI-SQUARE TEST FOR CATEGORICAL FEATURES")
    print("="*60)
    
    chi2_results = selector.chi_square_test(df, k_best=5)
    
    # Validate results
    print(f"\nValidation Results:")
    print(f"Number of features selected: {len(chi2_results)}")
    print(f"All Chi2-scores positive: {all(stats['chi2_score'] > 0 for stats in chi2_results.values())}")
    print(f"All p-values between 0 and 1: {all(0 <= stats['p_value'] <= 1 for stats in chi2_results.values())}")
    
    # Check if results are reasonable
    if chi2_results:
        best_feature = max(chi2_results.items(), key=lambda x: x[1]['chi2_score'])
        print(f"Best categorical feature: {best_feature[0]} (Chi2-score: {best_feature[1]['chi2_score']:.4f})")
    
    return len(chi2_results) > 0

def test_mutual_information(selector, df):
    """
    Test Mutual Information for all features
    
    Args:
        selector: StatisticalFeatureSelector instance
        df: Sample dataframe
    """
    print("\n" + "="*60)
    print("TEST 5: MUTUAL INFORMATION FOR ALL FEATURES")
    print("="*60)
    
    mi_results = selector.mutual_information_test(df, k_best=10)
    
    # Validate results
    print(f"\nValidation Results:")
    print(f"Number of features selected: {len(mi_results)}")
    print(f"All MI-scores non-negative: {all(stats['mi_score'] >= 0 for stats in mi_results.values())}")
    
    # Check feature type distribution
    numerical_count = sum(1 for stats in mi_results.values() if stats['feature_type'] == 'numerical')
    categorical_count = sum(1 for stats in mi_results.values() if stats['feature_type'] == 'categorical')
    
    print(f"Feature type distribution:")
    print(f"  Numerical: {numerical_count}")
    print(f"  Categorical: {categorical_count}")
    
    # Check if results are reasonable
    if mi_results:
        best_feature = max(mi_results.items(), key=lambda x: x[1]['mi_score'])
        print(f"Best feature overall: {best_feature[0]} (MI-score: {best_feature[1]['mi_score']:.4f})")
    
    return len(mi_results) > 0

def test_combined_ranking(selector, df):
    """
    Test combined feature ranking
    
    Args:
        selector: StatisticalFeatureSelector instance
        df: Sample dataframe
    """
    print("\n" + "="*60)
    print("TEST 6: COMBINED FEATURE RANKING")
    print("="*60)
    
    combined_results = selector.get_combined_feature_ranking()
    
    # Validate results
    print(f"\nValidation Results:")
    print(f"Number of features in combined ranking: {len(combined_results)}")
    print(f"All combined scores positive: {all(stats['combined_score'] > 0 for stats in combined_results.values())}")
    
    # Check ranking consistency
    sorted_features = sorted(combined_results.items(), key=lambda x: x[1]['combined_score'], reverse=True)
    ranks_consistent = all(i + 1 == stats['rank'] for i, (_, stats) in enumerate(sorted_features))
    print(f"Ranking consistency: {ranks_consistent}")
    
    # Show top 5 features
    print(f"\nTop 5 features from combined ranking:")
    for i, (feature, stats) in enumerate(sorted_features[:5]):
        print(f"  {i+1}. {feature} (Score: {stats['combined_score']:.4f}, Type: {stats['feature_type']})")
    
    return len(combined_results) > 0

def test_complete_pipeline(df):
    """
    Test the complete feature selection pipeline
    
    Args:
        df: Sample dataframe
    """
    print("\n" + "="*60)
    print("TEST 7: COMPLETE FEATURE SELECTION PIPELINE")
    print("="*60)
    
    # Initialize selector
    selector = StatisticalFeatureSelector(target_column='SalesSuccess')
    
    # Run complete pipeline
    results = selector.perform_complete_feature_selection(
        df, 
        k_numerical=5, 
        k_categorical=5, 
        k_combined=10
    )
    
    # Validate complete results
    print(f"\nPipeline Validation:")
    print(f"ANOVA results available: {'anova' in results}")
    print(f"Chi-square results available: {'chi2' in results}")
    print(f"Mutual Information results available: {'mutual_info' in results}")
    print(f"Combined results available: {'combined' in results}")
    
    # Test saving results
    output_file = 'test_feature_selection_results.csv'
    results_df = selector.save_results(output_file)
    
    # Verify file was created
    file_created = os.path.exists(output_file)
    print(f"Results file created: {file_created}")
    
    if file_created:
        print(f"Results file shape: {results_df.shape}")
        print(f"Methods in results: {results_df['method'].unique()}")
        
        # Clean up test file
        try:
            os.remove(output_file)
            print(f"Test file cleaned up: {output_file}")
        except:
            print(f"Could not remove test file: {output_file}")
    
    return all([results.get(key) for key in ['anova', 'chi2', 'mutual_info', 'combined']])

def run_comprehensive_tests():
    """
    Run all feature selection tests
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE FEATURE SELECTION TESTING")
    print("="*80)
    
    # Create sample data
    try:
        df = create_sample_data(num_products=300, num_stores=8, num_months=4)
    except Exception as e:
        print(f"Error creating sample data: {e}")
        return False
    
    # Initialize selector
    selector = StatisticalFeatureSelector(target_column='SalesSuccess')
    
    # Run individual tests
    test_results = []
    
    try:
        test_results.append(("Feature Identification", test_feature_identification(selector, df)))
        test_results.append(("Target Encoding", test_target_encoding(selector, df)))
        test_results.append(("ANOVA F-Test", test_anova_f_test(selector, df)))
        test_results.append(("Chi-Square Test", test_chi_square_test(selector, df)))
        test_results.append(("Mutual Information", test_mutual_information(selector, df)))
        test_results.append(("Combined Ranking", test_combined_ranking(selector, df)))
        test_results.append(("Complete Pipeline", test_complete_pipeline(df)))
    except Exception as e:
        print(f"Error during testing: {e}")
        return False
    
    # Print test summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:30} | {status}")
        if result:
            passed_tests += 1
    
    print(f"\nOverall Results: {passed_tests}/{len(test_results)} tests passed")
    
    if passed_tests == len(test_results):
        print("🎉 All tests passed! Feature selection module is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

def demonstrate_usage():
    """
    Demonstrate practical usage of the feature selection module
    """
    print("\n" + "="*80)
    print("PRACTICAL USAGE DEMONSTRATION")
    print("="*80)
    
    # Load or create data
    try:
        # Try to load existing data
        data_path = '../monthly_product_sales.csv'
        if os.path.exists(data_path):
            print(f"Loading existing data from {data_path}")
            df = pd.read_csv(data_path)
            # Limit data size for demonstration
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
                print(f"Sampled {len(df)} rows for demonstration")
        else:
            print("Creating sample data for demonstration")
            df = create_sample_data(num_products=400, num_stores=10, num_months=6)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample data instead")
        df = create_sample_data(num_products=400, num_stores=10, num_months=6)
    
    # Initialize and run feature selection
    print(f"\nRunning feature selection on {df.shape[0]} samples with {df.shape[1]} features")
    
    selector = StatisticalFeatureSelector(target_column='SalesSuccess')
    
    # Perform complete feature selection
    results = selector.perform_complete_feature_selection(
        df,
        k_numerical=8,
        k_categorical=8,
        k_combined=15
    )
    
    # Save results
    output_file = 'sales_prediction_feature_selection.csv'
    results_df = selector.save_results(output_file)
    
    print(f"\n📊 Feature selection completed successfully!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📈 Total features analyzed: {len(selector.numerical_features) + len(selector.categorical_features)}")
    print(f"🎯 Top features selected: {len(results['combined'])}")
    
    # Show practical recommendations
    print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
    print(f"   Use the top 10-15 features from the combined ranking for model training")
    print(f"   Focus on features with high statistical significance (low p-values)")
    print(f"   Consider both numerical and categorical features for balanced model")
    
    return True

if __name__ == "__main__":
    print("Statistical Feature Selection - Test Suite")
    print("This script tests all functionality of the feature selection module")
    
    # Run comprehensive tests
    tests_passed = run_comprehensive_tests()
    
    if tests_passed:
        # Demonstrate practical usage
        demonstrate_usage()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)