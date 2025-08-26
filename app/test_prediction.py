import requests
import json

# Test data for prediction
test_data = {
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

print("Testing prediction API...")
print(f"Test data: {json.dumps(test_data, indent=2)}")

try:
    response = requests.post(
        'http://127.0.0.1:5000/api/predict',
        json=test_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"\nResponse status: {response.status_code}")
    print(f"Response content: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print("\n✓ Prediction successful!")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']}")
        else:
            print(f"\n✗ Prediction failed: {result.get('error')}")
    else:
        print(f"\n✗ HTTP Error: {response.status_code}")
        
except Exception as e:
    print(f"\n✗ Error: {e}")