import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker for generating fake data
fake = Faker()

# --- Configuration ---
NUM_PRODUCTS = 3000
NUM_STORES = 50  # Reduced for manageable file size, but can be set to 200
NUM_MONTHS = 24
OUTPUT_DIR = "d:\\Dataset_new"

# --- Data Definitions ---

# Product Categories and Sub-categories (relevant to Sri Lanka)
PRODUCT_CATEGORIES = {
    "Groceries": ["Rice & Grains", "Spices & Condiments", "Canned Goods", "Cooking Oils", "Flour & Baking"],
    "Fresh Produce": ["Vegetables", "Fruits", "Herbs"],
    "Dairy & Eggs": ["Milk", "Cheese", "Yogurt", "Butter & Margarine", "Eggs"],
    "Meat & Seafood": ["Chicken", "Beef", "Fish", "Prawns & Shellfish"],
    "Bakery": ["Bread", "Cakes & Pastries", "Biscuits"],
    "Beverages": ["Tea & Coffee", "Juices", "Soft Drinks", "Water"],
    "Snacks & Confectionery": ["Chips & Crisps", "Chocolates", "Sweets & Toffees", "Nuts & Seeds"],
    "Health & Beauty": ["Skincare", "Haircare", "Personal Hygiene", "Vitamins & Supplements"],
    "Household": ["Cleaning Supplies", "Laundry", "Paper Goods", "Pest Control"]
}

# Brands (mix of local and international)
BRANDS = ["Munchee", "Maliban", "Kist", "Elephant House", "Anchor", "Milo", "Prima", "Harischandra", "CIC", "Pelwatte", "Nestle", "Unilever", "Procter & Gamble", "Coca-Cola", "PepsiCo"]

# Store Locations (major cities in Sri Lanka)
STORE_LOCATIONS = ["Colombo", "Kandy", "Galle", "Jaffna", "Negombo", "Anuradhapura", "Ratnapura", "Trincomalee", "Matara", "Badulla"]

# --- Helper Functions ---

def generate_products(num_products):
    products = []
    for i in range(num_products):
        category = random.choice(list(PRODUCT_CATEGORIES.keys()))
        sub_category = random.choice(PRODUCT_CATEGORIES[category])
        products.append({
            "ProductID": f"PROD-{1000 + i}",
            "ProductName": f"{sub_category.replace(' & ', ' ')} Item {i}",
            "ProductCategory": category,
            "ProductSubCategory": sub_category,
            "Brand": random.choice(BRANDS),
            "IsLocal": random.choice([True, False])
        })
    return pd.DataFrame(products)

def generate_stores(num_stores):
    stores = []
    for i in range(num_stores):
        stores.append({
            "StoreID": f"STORE-{i+1}",
            "StoreLocation": random.choice(STORE_LOCATIONS),
            "StoreSize": random.choice(["Small", "Medium", "Large"])
        })
    return pd.DataFrame(stores)

def generate_sales_data(products_df, stores_df, num_months):
    sales_data = []
    start_date = datetime.now() - timedelta(days=num_months * 30)

    for _, store in stores_df.iterrows():
        for _, product in products_df.iterrows():
            for month in range(num_months):
                current_date = start_date + timedelta(days=month * 30)
                
                # Simulate seasonality (e.g., higher sales in April and December)
                seasonality_factor = 1.0
                if current_date.month == 4 or current_date.month == 12:
                    seasonality_factor = 1.5

                base_price = 100 + (hash(product["ProductID"]) % 1000)
                price = base_price * (1 + (hash(store["StoreID"]) % 100) / 500)
                
                base_units_sold = 50 + (hash(product["ProductID"]) % 200)
                units_sold = int(base_units_sold * seasonality_factor * (1 + (hash(store["StoreID"]) % 50) / 100))

                on_promotion = random.choice([True, False])
                if on_promotion:
                    units_sold *= 1.2 # Promotions increase sales
                    price *= 0.9 # Promotions decrease price

                sales_data.append({
                    "ProductID": product["ProductID"],
                    "ProductName": product["ProductName"],
                    "StoreID": store["StoreID"],
                    "StoreLocation": store["StoreLocation"],
                    "StoreSize": store["StoreSize"],
                    "Month": current_date.strftime("%Y-%m"),
                    "UnitsSold": units_sold,
                    "TotalRevenue": units_sold * price,
                    "Price": price,
                    "ProductCategory": product["ProductCategory"],
                    "ProductSubCategory": product["ProductSubCategory"],
                    "Brand": product["Brand"],
                    "IsLocal": product["IsLocal"],
                    "OnPromotion": on_promotion,
                    "PromotionType": "Discount" if on_promotion else "None",
                    "CustomerSegment": random.choice(["Families", "Young Professionals", "Budget Shoppers", "Health-Conscious"]),
                    "CompetitorPrice": price * random.uniform(0.95, 1.05),
                    "ShelfPlacement": random.choice(["Eye-Level", "Top-Shelf", "Bottom-Shelf"]),
                    "SupplierID": f"SUP-{product['Brand'][:3].upper()}-{random.randint(10, 99)}"
                })

    sales_df = pd.DataFrame(sales_data)

    # --- Create the Target Variable: SalesSuccess ---
    # Rank products within each category and store for the last month
    last_month = sales_df["Month"].max()
    last_month_sales = sales_df[sales_df["Month"] == last_month].copy()
    last_month_sales["RevenueRank"] = last_month_sales.groupby(["StoreID", "ProductCategory"])["TotalRevenue"].rank(pct=True)

    def assign_success(rank):
        if rank > 0.8:
            return "Top Seller"
        elif rank > 0.5:
            return "Good Seller"
        elif rank > 0.2:
            return "Moderate Seller"
        else:
            return "Slow Seller"

    last_month_sales["SalesSuccess"] = last_month_sales["RevenueRank"].apply(assign_success)
    
    # Merge the SalesSuccess back to the main dataframe
    sales_df = pd.merge(sales_df, last_month_sales[['ProductID', 'StoreID', 'SalesSuccess']], on=['ProductID', 'StoreID'], how='left')
    sales_df["SalesSuccess"] = sales_df.groupby(['ProductID', 'StoreID'])['SalesSuccess'].ffill().bfill()
    sales_df = sales_df.dropna(subset=['SalesSuccess'])

    return sales_df

def generate_new_products_data(num_new_products, stores_df):
    new_products = []
    for i in range(num_new_products):
        category = random.choice(list(PRODUCT_CATEGORIES.keys()))
        sub_category = random.choice(PRODUCT_CATEGORIES[category])
        store = stores_df.sample(1).iloc[0]
        price = 150 + (i * 50)

        new_products.append({
            "NewProductID": f"NEWPROD-{2000 + i}",
            "NewProductName": f"New {sub_category.replace(' & ', ' ')} Item {i}",
            "StoreID": store["StoreID"],
            "StoreLocation": store["StoreLocation"],
            "StoreSize": store["StoreSize"],
            "Price": price,
            "ProductCategory": category,
            "ProductSubCategory": sub_category,
            "Brand": random.choice(BRANDS),
            "IsLocal": random.choice([True, False]),
            "OnPromotion": True,
            "PromotionType": "Discount",
            "CustomerSegment": "Gourmet Foodies",
            "CompetitorPrice": price * random.uniform(0.95, 1.05),
            "ShelfPlacement": "Eye-Level",
            "SupplierID": f"SUP-NEW-{random.randint(10, 99)}"
        })
    return pd.DataFrame(new_products)


# --- Main Execution ---
if __name__ == "__main__":
    print("Generating product and store data...")
    products_df = generate_products(NUM_PRODUCTS)
    stores_df = generate_stores(NUM_STORES)

    print("Generating historical sales data (this may take a few minutes)...")
    historical_sales_df = generate_sales_data(products_df, stores_df, NUM_MONTHS)
    
    print("Generating new product prediction data...")
    new_products_df = generate_new_products_data(20, stores_df)

    # --- Save to CSV ---
    historical_sales_path = f'{OUTPUT_DIR}\\monthly_product_sales.csv'
    new_products_path = f'{OUTPUT_DIR}\\new_product_predictions.csv'

    print(f"Saving historical sales data to {historical_sales_path}...")
    historical_sales_df.to_csv(historical_sales_path, index=False)

    print(f"Saving new product data to {new_products_path}...")
    new_products_df.to_csv(new_products_path, index=False)

    print("\n--- Data Generation Complete ---")
    print(f"Historical data shape: {historical_sales_df.shape}")
    print(f"New product data shape: {new_products_df.shape}")

    print("\n--- Sample of historical_sales.csv ---")
    print(historical_sales_df.head())

    print("\n--- Sample of new_products.csv ---")
    print(new_products_df.head())