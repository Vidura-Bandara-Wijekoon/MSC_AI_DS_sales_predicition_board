import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("FEATURE ENGINEERING")
print("Multi-Class Classification for Product Sales Success")
print("=" * 60)

# Process data in smaller chunks to manage memory
print("\n1. PROCESSING DATA IN CHUNKS...")
chunk_size = 100000
processed_chunks = []
total_processed = 0

# Process each chunk separately
for i, chunk in enumerate(pd.read_csv('d:/Dataset_new/monthly_product_sales.csv', chunksize=chunk_size)):
    print(f"Processing chunk {i+1}... ({len(chunk):,} rows)")
    
    # Convert Month to datetime
    chunk['Month'] = pd.to_datetime(chunk['Month'])
    
    # Sort within chunk
    chunk = chunk.sort_values(['ProductID', 'StoreID', 'Month'])
    
    processed_chunks.append(chunk)
    total_processed += len(chunk)
    
    # Limit processing to manage memory (process first 1M rows)
    if total_processed >= 1000000:
        break

print(f"Processed {total_processed:,} rows in {len(processed_chunks)} chunks")

# Combine processed chunks
df = pd.concat(processed_chunks, ignore_index=True)
print(f"Final dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Date range: {df['Month'].min()} to {df['Month'].max()}")

print("\n2. CALCULATING 3-MONTH ROLLING AVERAGES...")

# Calculate 3-month rolling average for key metrics
# Group by ProductID and StoreID to calculate rolling averages
grouped = df.groupby(['ProductID', 'StoreID'])

# Calculate rolling averages
df['Monthly_Avg_Revenue_3M'] = grouped['TotalRevenue'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df['Monthly_Avg_Units_3M'] = grouped['UnitsSold'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df['Monthly_Avg_Price_3M'] = grouped['Price'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

print("3-month rolling averages calculated for Revenue, Units, and Price")

print("\n3. CREATING PRODUCT MASTER FEATURES...")

# Create product master with the features you specified
product_master = df.groupby('ProductID').agg({
    'ProductName': 'first',
    'ProductCategory': 'first', 
    'ProductSubCategory': 'first',
    'Brand': 'first',
    'Price': 'mean',  # Average selling price
    'Monthly_Avg_Revenue_3M': 'mean'
}).reset_index()

# Rename columns to match your specification
product_master.columns = ['SKU_Code', 'SKU_Name', 'Category', 'Sub_Category', 'Brand', 'Selling_Price', 'Avg_Revenue_3M']

# Calculate Margin % (assuming cost is 70% of selling price for demonstration)
product_master['Cost'] = product_master['Selling_Price'] * 0.7
product_master['Margin_Percent'] = ((product_master['Selling_Price'] - product_master['Cost']) / product_master['Selling_Price']) * 100

# Calculate Price per Volume (assuming volume data - using price per unit as proxy)
product_master['Price_per_Volume'] = product_master['Selling_Price']  # This would be price/volume if volume data available

print(f"Product master created with {len(product_master)} unique products")
print("\nProduct master columns:", list(product_master.columns))

print("\n4. CREATING SKU-STORE PERFORMANCE FEATURES...")

# Create SKU-Store combination features as you specified
sku_store_performance = df.groupby(['ProductID', 'StoreID']).agg({
    'Monthly_Avg_Revenue_3M': 'mean',
    'Monthly_Avg_Units_3M': 'mean',
    'SalesSuccess': lambda x: x.mode()[0] if not x.empty else 'Unknown'  # Most common sales success category
}).reset_index()

# Rename columns to match your specification
sku_store_performance.columns = ['SKU_Code', 'StoreID', 'Monthly_Average_Net_Sale_3M', 'Monthly_Average_Units_3M', 'Sales_Success_Category']

print(f"SKU-Store performance data created with {len(sku_store_performance)} combinations")
print("\nSKU-Store performance columns:", list(sku_store_performance.columns))

print("\n5. PROMOTION STATUS FEATURES...")

# Create promotion features
promotion_features = df.groupby(['ProductID', 'StoreID']).agg({
    'OnPromotion': 'mean',  # Percentage of time on promotion
    'PromotionType': lambda x: x.mode()[0] if not x.empty else 'None'  # Most common promotion type
}).reset_index()

promotion_features.columns = ['SKU_Code', 'StoreID', 'Promotion_Rate', 'Most_Common_Promotion_Type']

# Create promotion impact features
promo_impact = df.groupby(['ProductID', 'StoreID', 'OnPromotion'])['TotalRevenue'].mean().unstack(fill_value=0)
promo_impact.columns = ['Revenue_No_Promo', 'Revenue_With_Promo']
promo_impact['Promotion_Lift'] = ((promo_impact['Revenue_With_Promo'] - promo_impact['Revenue_No_Promo']) / promo_impact['Revenue_No_Promo']) * 100
promo_impact = promo_impact.reset_index()
promo_impact.columns = ['SKU_Code', 'StoreID', 'Revenue_No_Promo', 'Revenue_With_Promo', 'Promotion_Lift']

print(f"Promotion features created for {len(promotion_features)} SKU-Store combinations")

print("\n6. LOCATION'S CUSTOMER PROFILING FEATURES...")

# Create customer profiling features by store location
customer_profiling = df.groupby(['StoreID', 'StoreLocation']).agg({
    'CustomerSegment': lambda x: x.mode()[0] if not x.empty else 'Unknown',  # Dominant customer segment
    'TotalRevenue': 'mean',  # Average revenue per transaction
    'UnitsSold': 'mean',  # Average units per transaction
    'Price': 'mean'  # Average price point
}).reset_index()

customer_profiling.columns = ['StoreID', 'Store_Location', 'Dominant_Customer_Segment', 'Avg_Transaction_Revenue', 'Avg_Units_Per_Transaction', 'Avg_Price_Point']

# Calculate customer segment distribution by store
customer_distribution = df.groupby(['StoreID', 'CustomerSegment']).size().unstack(fill_value=0)
customer_distribution = customer_distribution.div(customer_distribution.sum(axis=1), axis=0) * 100
customer_distribution = customer_distribution.reset_index()

print(f"Customer profiling features created for {len(customer_profiling)} stores")
print("\nCustomer segments:", df['CustomerSegment'].unique())

print("\n7. ADDITIONAL CONTEXTUAL FEATURES...")

# Shelf placement features
shelf_features = df.groupby(['ProductID', 'StoreID']).agg({
    'ShelfPlacement': lambda x: x.mode()[0] if not x.empty else 'Unknown',
    'CompetitorPrice': 'mean'
}).reset_index()

shelf_features.columns = ['SKU_Code', 'StoreID', 'Primary_Shelf_Placement', 'Avg_Competitor_Price']

# Calculate price competitiveness
shelf_features = shelf_features.merge(
    product_master[['SKU_Code', 'Selling_Price']], 
    on='SKU_Code', 
    how='left'
)
shelf_features['Price_Competitiveness'] = ((shelf_features['Avg_Competitor_Price'] - shelf_features['Selling_Price']) / shelf_features['Avg_Competitor_Price']) * 100

print(f"Shelf and competitive features created")

print("\n8. SEASONALITY FEATURES...")

# Create seasonality features
df['Month_Number'] = df['Month'].dt.month
df['Quarter'] = df['Month'].dt.quarter
df['Year'] = df['Month'].dt.year

# Calculate seasonal performance
seasonal_performance = df.groupby(['ProductID', 'Quarter']).agg({
    'TotalRevenue': 'mean',
    'UnitsSold': 'mean'
}).reset_index()

seasonal_pivot = seasonal_performance.pivot(index='ProductID', columns='Quarter', values='TotalRevenue')
seasonal_pivot.columns = [f'Q{i}_Avg_Revenue' for i in seasonal_pivot.columns]
seasonal_pivot = seasonal_pivot.reset_index()
seasonal_pivot.columns.name = None

print(f"Seasonality features created")

print("\n9. SAVING ENGINEERED FEATURES...")

# Save all feature sets
product_master.to_csv('d:/Dataset_new/product_master_features.csv', index=False)
sku_store_performance.to_csv('d:/Dataset_new/sku_store_performance.csv', index=False)
promotion_features.to_csv('d:/Dataset_new/promotion_features.csv', index=False)
promo_impact.to_csv('d:/Dataset_new/promotion_impact_features.csv', index=False)
customer_profiling.to_csv('d:/Dataset_new/customer_profiling_features.csv', index=False)
customer_distribution.to_csv('d:/Dataset_new/customer_distribution_by_store.csv', index=False)
shelf_features.to_csv('d:/Dataset_new/shelf_competitive_features.csv', index=False)
seasonal_pivot.to_csv('d:/Dataset_new/seasonal_features.csv', index=False)

print("\nAll feature files saved successfully!")

print("\n10. CREATING FINAL MODELING DATASET...")

# Create the final dataset for modeling by merging all features
modeling_data = sku_store_performance.copy()

# Add product master features
modeling_data = modeling_data.merge(product_master, on='SKU_Code', how='left')

# Add promotion features
modeling_data = modeling_data.merge(promotion_features, on=['SKU_Code', 'StoreID'], how='left')
modeling_data = modeling_data.merge(promo_impact, on=['SKU_Code', 'StoreID'], how='left')

# Add customer profiling
modeling_data = modeling_data.merge(customer_profiling, on='StoreID', how='left')

# Add shelf and competitive features
modeling_data = modeling_data.merge(shelf_features[['SKU_Code', 'StoreID', 'Primary_Shelf_Placement', 'Price_Competitiveness']], 
                                   on=['SKU_Code', 'StoreID'], how='left')

# Add seasonal features
modeling_data = modeling_data.merge(seasonal_pivot, left_on='SKU_Code', right_on='ProductID', how='left')
modeling_data = modeling_data.drop('ProductID', axis=1, errors='ignore')

print(f"Final modeling dataset created with {len(modeling_data)} records and {len(modeling_data.columns)} features")

# Save final modeling dataset
modeling_data.to_csv('d:/Dataset_new/final_modeling_dataset.csv', index=False)

print("\n" + "="*60)
print("FEATURE ENGINEERING SUMMARY")
print("="*60)
print(f"1. Product Master Features: {len(product_master)} products")
print(f"   - SKU Code, Name, Category, Sub-Category, Brand")
print(f"   - Selling Price, Margin %, Price per Volume")
print(f"\n2. SKU-Store Performance: {len(sku_store_performance)} combinations")
print(f"   - Monthly Average Net Sale (3 Months)")
print(f"   - Sales Success Category")
print(f"\n3. Promotion Status Features: {len(promotion_features)} combinations")
print(f"   - Promotion Rate, Most Common Promotion Type")
print(f"   - Promotion Lift Analysis")
print(f"\n4. Customer Profiling Features: {len(customer_profiling)} stores")
print(f"   - Dominant Customer Segment by Location")
print(f"   - Average Transaction Metrics")
print(f"\n5. Additional Features:")
print(f"   - Shelf Placement and Competitive Analysis")
print(f"   - Seasonality Features")
print(f"\n6. Final Modeling Dataset: {len(modeling_data)} records, {len(modeling_data.columns)} features")

print("\nFeature engineering completed successfully!")
print("Ready for model training phase.")

# Display sample of final dataset
print("\nSample of final modeling dataset:")
print(modeling_data.head())
print("\nFinal dataset columns:")
print(list(modeling_data.columns))