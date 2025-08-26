import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class ProductStoreSalesPredictor:
    def __init__(self):
        """Initialize the predictor by loading the trained model and encoders"""
        self.model = None
        self.label_encoders = None
        self.target_encoder = None
        self.feature_names = None
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """Load the trained model and all necessary encoders"""
        try:
            # Load the trained CatBoost model
            self.model = CatBoostClassifier()
            self.model.load_model('product_store_catboost_model.cbm')
            print("✓ Model loaded successfully")
            
            # Load label encoders
            with open('product_store_label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            print("✓ Label encoders loaded successfully")
            
            # Load target encoder
            with open('product_store_target_encoder.pkl', 'rb') as f:
                self.target_encoder = pickle.load(f)
            print("✓ Target encoder loaded successfully")
            
            # Get feature names from the model
            self.feature_names = self.model.feature_names_
            print(f"✓ Model expects {len(self.feature_names)} features")
            
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            raise
    
    def predict_single_product_store(self, product_data):
        """
        Predict sales success for a single product-store combination
        
        Args:
            product_data (dict): Dictionary containing product and store information
        
        Returns:
            dict: Prediction results with probabilities
        """
        try:
            # Create a DataFrame from the input data
            input_df = pd.DataFrame([product_data])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    # Set default values for missing features
                    if feature in self.label_encoders:
                        input_df[feature] = 'Unknown'
                    else:
                        input_df[feature] = 0
            
            # Reorder columns to match training data
            input_df = input_df[self.feature_names]
            
            # Encode categorical features
            input_encoded = input_df.copy()
            for col in self.label_encoders.keys():
                if col in input_encoded.columns:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    try:
                        input_encoded[col] = le.transform(input_encoded[col].astype(str))
                    except ValueError:
                        # If category not seen during training, use the most frequent class
                        input_encoded[col] = 0  # Default to first class
            
            # Make prediction
            prediction = self.model.predict(input_encoded)[0]
            prediction_proba = self.model.predict_proba(input_encoded)[0]
            
            # Convert prediction back to original label
            predicted_class = self.target_encoder.inverse_transform([prediction])[0]
            
            # Create probability dictionary
            class_probabilities = {}
            for i, class_name in enumerate(self.target_encoder.classes_):
                class_probabilities[class_name] = float(prediction_proba[i])
            
            return {
                'predicted_class': str(predicted_class),
                'confidence': float(max(prediction_proba)),
                'class_probabilities': class_probabilities
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def get_feature_requirements(self):
        """
        Return the required features and their types for prediction
        """
        feature_info = {}
        for feature in self.feature_names:
            if feature in self.label_encoders:
                feature_info[feature] = {
                    'type': 'categorical',
                    'possible_values': list(self.label_encoders[feature].classes_)
                }
            else:
                feature_info[feature] = {
                    'type': 'numerical',
                    'description': 'Numeric value'
                }
        return feature_info
    
    def predict_batch(self, products_data):
        """
        Predict sales success for multiple product-store combinations
        
        Args:
            products_data (list): List of dictionaries containing product and store information
        
        Returns:
            list: List of prediction results
        """
        results = []
        for product_data in products_data:
            result = self.predict_single_product_store(product_data)
            results.append(result)
        return results

# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("PRODUCT-STORE SALES PREDICTOR")
    print("=" * 80)
    
    # Initialize the predictor
    predictor = ProductStoreSalesPredictor()
    
    # Show required features
    print("\nRequired Features for Prediction:")
    print("-" * 50)
    feature_info = predictor.get_feature_requirements()
    for feature, info in feature_info.items():
        print(f"{feature}: {info['type']}")
        if info['type'] == 'categorical' and len(info['possible_values']) <= 10:
            print(f"  Possible values: {info['possible_values']}")
    
    # Example prediction
    print("\nExample Prediction:")
    print("-" * 50)
    
    # Sample product-store data
    sample_product = {
        'StoreID': 1001,
        'ProductCategory': 'Electronics',
        'ProductSubCategory': 'Smartphones',
        'Price': 599.99,
        'Margin': 0.25,
        'Pack_type': 'Box',
        'Format': 'Standard',
        'Size': 'Medium',
        'Location_cluster': 'Urban',
        'Basket_value': 150.0,
        'Category_contribution': 0.15,
        'SKU_count_in_subcategory': 25,
        'Brand_share': 0.12,
        'Month': 'January',
        'Shelf_life': 365,
        'Sales_velocity': 2.5,
        'Price_per_volume': 12.5
    }
    
    # Make prediction
    result = predictor.predict_single_product_store(sample_product)
    
    if result:
        print(f"Predicted Sales Category: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nClass Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")
    
    print("\n" + "=" * 80)
    print("PREDICTOR READY FOR USE!")
    print("Use predictor.predict_single_product_store(product_data) for predictions")
    print("=" * 80)