import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("DATA EXPLORATION AND PREPROCESSING")
print("Multi-Class Classification for Product Sales Success")
print("=" * 60)

# Load the data in chunks to avoid memory issues
print("\n1. LOADING DATA...")
chunk_size = 100000  # Load 100k rows at a time
chunks = []
total_rows = 0

# Read file in chunks
for chunk in pd.read_csv('d:/Dataset_new/monthly_product_sales.csv', chunksize=chunk_size):
    chunks.append(chunk)
    total_rows += len(chunk)
    if len(chunks) >= 5:  # Limit to first 500k rows for exploration
        break

# Combine chunks
df = pd.concat(chunks, ignore_index=True)
print(f"Data sample loaded successfully!")
print(f"Sample dataset shape: {df.shape}")
print(f"Total rows in file: ~{total_rows:,} (estimated)")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Basic information about the dataset
print("\n2. DATASET OVERVIEW")
print("\nColumn Information:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# Check for missing values
print("\n3. MISSING VALUES ANALYSIS")
missing_values = df.isnull().sum()
print("Missing values per column:")
for col, missing in missing_values.items():
    if missing > 0:
        print(f"{col}: {missing} ({missing/len(df)*100:.2f}%)")
    else:
        print(f"{col}: 0 (0.00%)")

# Data types analysis
print("\n4. DATA TYPES ANALYSIS")
print("\nNumerical columns:")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(numerical_cols)

print("\nCategorical columns:")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(categorical_cols)

# Target variable analysis
print("\n5. TARGET VARIABLE ANALYSIS (SalesSuccess)")
print("\nSalesSuccess distribution:")
target_dist = df['SalesSuccess'].value_counts()
print(target_dist)
print("\nSalesSuccess percentages:")
print(df['SalesSuccess'].value_counts(normalize=True) * 100)

# Key features analysis
print("\n6. KEY FEATURES ANALYSIS")

# Product categories
print("\nProduct Categories:")
print(df['ProductCategory'].value_counts())

print("\nProduct Sub-Categories:")
print(df['ProductSubCategory'].value_counts())

# Brand analysis
print("\nTop 10 Brands:")
print(df['Brand'].value_counts().head(10))

# Store analysis
print("\nStore Locations:")
print(df['StoreLocation'].value_counts())

print("\nStore Sizes:")
print(df['StoreSize'].value_counts())

# PROMOTION STATUS ANALYSIS (as requested)
print("\n7. PROMOTION STATUS ANALYSIS")
print("\nPromotion Status Distribution:")
print(df['OnPromotion'].value_counts())
print("\nPromotion Status Percentages:")
print(df['OnPromotion'].value_counts(normalize=True) * 100)

print("\nPromotion Types:")
print(df['PromotionType'].value_counts())

# Cross-analysis: Promotion vs Sales Success
print("\nPromotion Impact on Sales Success:")
promo_success = pd.crosstab(df['OnPromotion'], df['SalesSuccess'], normalize='index') * 100
print(promo_success)

# CUSTOMER PROFILING ANALYSIS (as requested)
print("\n8. LOCATION'S CUSTOMER PROFILING ANALYSIS")
print("\nCustomer Segments:")
print(df['CustomerSegment'].value_counts())

print("\nCustomer Segment Percentages:")
print(df['CustomerSegment'].value_counts(normalize=True) * 100)

# Cross-analysis: Customer Segment vs Sales Success
print("\nCustomer Segment Impact on Sales Success:")
customer_success = pd.crosstab(df['CustomerSegment'], df['SalesSuccess'], normalize='index') * 100
print(customer_success)

# Location vs Customer Segment
print("\nLocation vs Customer Segment Distribution:")
location_customer = pd.crosstab(df['StoreLocation'], df['CustomerSegment'], normalize='index') * 100
print(location_customer.head(10))

# Numerical features analysis
print("\n9. NUMERICAL FEATURES ANALYSIS")
print("\nDescriptive statistics for numerical features:")
print(df[numerical_cols].describe())

# Price analysis
print("\nPrice Analysis:")
print(f"Average Price: {df['Price'].mean():.2f}")
print(f"Median Price: {df['Price'].median():.2f}")
print(f"Price Range: {df['Price'].min():.2f} - {df['Price'].max():.2f}")

# Revenue analysis
print("\nRevenue Analysis:")
print(f"Average Revenue: {df['TotalRevenue'].mean():.2f}")
print(f"Median Revenue: {df['TotalRevenue'].median():.2f}")
print(f"Revenue Range: {df['TotalRevenue'].min():.2f} - {df['TotalRevenue'].max():.2f}")

# Units sold analysis
print("\nUnits Sold Analysis:")
print(f"Average Units Sold: {df['UnitsSold'].mean():.2f}")
print(f"Median Units Sold: {df['UnitsSold'].median():.2f}")
print(f"Units Sold Range: {df['UnitsSold'].min():.2f} - {df['UnitsSold'].max():.2f}")

# Time series analysis
print("\n10. TIME SERIES ANALYSIS")
df['Month'] = pd.to_datetime(df['Month'])
print("\nMonthly data range:")
print(f"From: {df['Month'].min()}")
print(f"To: {df['Month'].max()}")
print(f"Total months: {df['Month'].nunique()}")

# Monthly sales trends
monthly_sales = df.groupby('Month')['TotalRevenue'].sum().reset_index()
print("\nMonthly total revenue:")
print(monthly_sales)

# Shelf placement analysis
print("\n11. SHELF PLACEMENT ANALYSIS")
print("\nShelf Placement Distribution:")
print(df['ShelfPlacement'].value_counts())

# Shelf placement vs Sales Success
print("\nShelf Placement Impact on Sales Success:")
shelf_success = pd.crosstab(df['ShelfPlacement'], df['SalesSuccess'], normalize='index') * 100
print(shelf_success)

# Competitive analysis
print("\n12. COMPETITIVE ANALYSIS")
print("\nPrice vs Competitor Price Analysis:")
df['PriceAdvantage'] = df['Price'] - df['CompetitorPrice']
print(f"Average Price Advantage: {df['PriceAdvantage'].mean():.2f}")
print(f"Median Price Advantage: {df['PriceAdvantage'].median():.2f}")

# Price advantage vs Sales Success
print("\nPrice Advantage Impact on Sales Success:")
df['PriceAdvantageCategory'] = pd.cut(df['PriceAdvantage'], 
                                     bins=[-np.inf, -50, 0, 50, np.inf], 
                                     labels=['Much Cheaper', 'Cheaper', 'More Expensive', 'Much More Expensive'])
price_adv_success = pd.crosstab(df['PriceAdvantageCategory'], df['SalesSuccess'], normalize='index') * 100
print(price_adv_success)

# Data quality checks
print("\n13. DATA QUALITY CHECKS")
print("\nChecking for duplicates:")
duplicates = df.duplicated().sum()
print(f"Total duplicate rows: {duplicates}")

print("\nChecking for negative values in numerical columns:")
for col in ['UnitsSold', 'TotalRevenue', 'Price']:
    negative_count = (df[col] < 0).sum()
    print(f"{col}: {negative_count} negative values")

print("\nChecking for zero values in key columns:")
for col in ['UnitsSold', 'TotalRevenue', 'Price']:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} zero values")

# Summary insights
print("\n" + "="*60)
print("SUMMARY INSIGHTS")
print("="*60)
print(f"1. Dataset contains {df.shape[0]:,} records across {df.shape[1]} features")
print(f"2. Time period: {df['Month'].nunique()} months from {df['Month'].min().strftime('%Y-%m')} to {df['Month'].max().strftime('%Y-%m')}")
print(f"3. Products: {df['ProductID'].nunique():,} unique products")
print(f"4. Stores: {df['StoreID'].nunique()} unique stores")
print(f"5. Product categories: {df['ProductCategory'].nunique()}")
print(f"6. Brands: {df['Brand'].nunique()}")
print(f"7. Customer segments: {df['CustomerSegment'].nunique()}")
print(f"8. Promotion rate: {(df['OnPromotion'] == True).mean()*100:.1f}%")
print(f"9. Target variable distribution: {dict(df['SalesSuccess'].value_counts())}")

print("\nData exploration completed successfully!")
print("Ready for feature engineering phase.")