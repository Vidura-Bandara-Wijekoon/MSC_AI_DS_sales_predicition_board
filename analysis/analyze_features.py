import pandas as pd

print('FEATURES OF MONTHLY_PRODUCT_SALES.CSV:')
print('=' * 50)

# Load a sample of the data
df = pd.read_csv('d:/Dataset_new/monthly_product_sales.csv', nrows=1000)

print(f'Total columns: {len(df.columns)}')
print(f'Sample size checked: {len(df)} rows')
print(f'Total file size: ~785 MB with 3.6M records')

print('\nColumn Names and Data Types:')
print('-' * 40)
for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
    print(f'{i:2d}. {col:<25} ({dtype})')

print('\nSample values for key categorical features:')
print('-' * 40)
print('ProductCategory unique values:', df['ProductCategory'].unique()[:5])
print('CustomerSegment unique values:', df['CustomerSegment'].unique())
print('StoreLocation unique values:', df['StoreLocation'].unique()[:5])
print('SalesSuccess unique values:', df['SalesSuccess'].unique())
print('PromotionType unique values:', df['PromotionType'].unique())
print('ShelfPlacement unique values:', df['ShelfPlacement'].unique())

print('\nNumerical features summary:')
print('-' * 40)
numerical_cols = ['UnitsSold', 'TotalRevenue', 'Price', 'CompetitorPrice']
for col in numerical_cols:
    print(f'{col}: Min={df[col].min():.2f}, Max={df[col].max():.2f}, Mean={df[col].mean():.2f}')

print('\nBoolean features:')
print('-' * 40)
print(f'IsLocal: {df["IsLocal"].value_counts().to_dict()}')
print(f'OnPromotion: {df["OnPromotion"].value_counts().to_dict()}')

print('\nFeature Categories:')
print('-' * 40)
print('1. Product Identifiers: ProductID, ProductName, SupplierID')
print('2. Store Information: StoreID, StoreLocation, StoreSize')
print('3. Time Information: Month')
print('4. Sales Metrics: UnitsSold, TotalRevenue, Price')
print('5. Product Attributes: ProductCategory, ProductSubCategory, Brand, IsLocal')
print('6. Promotion Features: OnPromotion, PromotionType')
print('7. Customer Context: CustomerSegment')
print('8. Competitive Context: CompetitorPrice, ShelfPlacement')
print('9. Target Variable: SalesSuccess (Top Seller, Good Performer, Average, Poor)')