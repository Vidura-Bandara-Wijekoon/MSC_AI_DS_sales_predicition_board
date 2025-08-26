import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CATBOOST MACHINE LEARNING PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: PROBLEM UNDERSTANDING
# ============================================================================
print("\n1. PROBLEM UNDERSTANDING")
print("-" * 50)
print("Objective: Multi-class classification for Sales Success prediction")
print("Target Variable: SalesSuccess (Top Seller, Good Seller, Moderate Seller, Slow Seller)")
print("Algorithm: CatBoost (Gradient Boosting on Decision Trees)")
print("Features: Product-level, Store-level, and Category-level features")

# ============================================================================
# STEP 2: DATA COLLECTION
# ============================================================================
print("\n2. DATA COLLECTION")
print("-" * 50)
print("Loading the specific features dataset...")

# Load the dataset
df = pd.read_csv('specific_features_dataset.csv')
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n3. DATA PREPROCESSING")
print("-" * 50)

# 3.1 Handle Missing Values
print("\n3.1 Checking for missing values...")
missing_values = df.isnull().sum()
print(f"Missing values:\n{missing_values[missing_values > 0]}")
if missing_values.sum() == 0:
    print("No missing values found!")
else:
    # Fill missing values if any
    df = df.fillna(df.median(numeric_only=True))
    print("Missing values handled.")

# 3.2 Handle Duplicates
print("\n3.2 Checking for duplicates...")
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows.")

# 3.3 Data Type Optimization
print("\n3.3 Optimizing data types...")
print("Before optimization:")
print(df.dtypes)

# Convert categorical columns to category type
categorical_cols = ['Pack_type', 'Format', 'Size', 'Location_cluster', 
                   'ProductCategory', 'ProductSubCategory', 'Month', 'SalesSuccess']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print("\nAfter optimization:")
print(df.dtypes)

# 3.4 Handle Special Characters (if any in text columns)
print("\n3.4 Handling special characters...")
text_columns = df.select_dtypes(include=['object', 'category']).columns
for col in text_columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.strip().str.upper()
print("Text columns standardized.")

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n4. EXPLORATORY DATA ANALYSIS (EDA)")
print("-" * 50)

# 4.1 Basic Statistics
print("\n4.1 Dataset Overview:")
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 4.2 Target Variable Analysis
print("\n4.2 Target Variable Distribution:")
target_dist = df['SalesSuccess'].value_counts()
print(target_dist)
print(f"\nTarget proportions:")
print(df['SalesSuccess'].value_counts(normalize=True))

# 4.3 Numerical Features Analysis
print("\n4.3 Numerical Features Summary:")
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(df[numerical_cols].describe())

# 4.4 Categorical Features Analysis
print("\n4.4 Categorical Features Analysis:")
for col in categorical_cols:
    if col in df.columns and col != 'SalesSuccess':
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
        if unique_count <= 10:
            print(f"  Values: {df[col].unique()}")

# 4.5 Feature Correlations
print("\n4.5 Feature Correlation Analysis:")
corr_matrix = df[numerical_cols].corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print("High correlation pairs (>0.8):")
    for pair in high_corr_pairs:
        print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
else:
    print("No high correlation pairs found.")

# ============================================================================
# STEP 5: FEATURE EXTRACTION
# ============================================================================
print("\n5. FEATURE EXTRACTION")
print("-" * 50)

# 5.1 Prepare features for CatBoost
print("\n5.1 Preparing features for CatBoost...")

# Separate features and target
X = df.drop(['SalesSuccess'], axis=1)
y = df['SalesSuccess']

# Identify categorical and numerical features
categorical_features = []
numerical_features = []

for col in X.columns:
    if X[col].dtype in ['object', 'category']:
        categorical_features.append(col)
    else:
        numerical_features.append(col)

print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
print(f"Numerical features ({len(numerical_features)}): {numerical_features}")

# 5.2 Encode categorical features for CatBoost
print("\n5.2 Encoding categorical features...")
label_encoders = {}
X_encoded = X.copy()

for col in categorical_features:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print(f"Target classes: {target_encoder.classes_}")
print(f"Encoded target shape: {y_encoded.shape}")

# ============================================================================
# STEP 6: DATA SPLITTING (70% Training, 30% Validation)
# ============================================================================
print("\n6. DATA SPLITTING")
print("-" * 50)

# Use a sample for training due to memory constraints
print("\n6.1 Sampling data for memory efficiency...")
sample_size = min(500000, len(X_encoded))  # Use 500K records max
print(f"Using sample size: {sample_size:,} out of {len(X_encoded):,} total records")

# Sample the data while maintaining class distribution
from sklearn.model_selection import train_test_split
X_sample, _, y_sample, _ = train_test_split(
    X_encoded, y_encoded,
    train_size=sample_size,
    random_state=42,
    stratify=y_encoded
)

# Split the sampled data
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, 
    test_size=0.30, 
    random_state=42, 
    stratify=y_sample
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training target distribution:")
print(pd.Series(y_train).value_counts())
print(f"Test target distribution:")
print(pd.Series(y_test).value_counts())

# ============================================================================
# STEP 7: MODEL TRAINING (CATBOOST)
# ============================================================================
print("\n7. CATBOOST MODEL TRAINING")
print("-" * 50)

# 7.1 Create CatBoost Pools
print("\n7.1 Creating CatBoost data pools...")
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=[X_train.columns.get_loc(col) for col in categorical_features]
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=[X_test.columns.get_loc(col) for col in categorical_features]
)

print("CatBoost pools created successfully!")

# 7.2 Initialize CatBoost Classifier (optimized for memory)
print("\n7.2 Initializing CatBoost Classifier...")
catboost_model = CatBoostClassifier(
    iterations=500,  # Reduced iterations
    learning_rate=0.1,
    depth=4,  # Reduced depth
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=50,
    early_stopping_rounds=30,
    thread_count=2,  # Limit threads
    max_ctr_complexity=1  # Reduce memory usage
)

# 7.3 Train the model
print("\n7.3 Training CatBoost model...")
catboost_model.fit(
    train_pool,
    eval_set=test_pool,
    plot=False
)

print("CatBoost model training completed!")

# ============================================================================
# STEP 8: MODEL EVALUATION
# ============================================================================
print("\n8. MODEL EVALUATION")
print("-" * 50)

# 8.1 Make predictions
print("\n8.1 Making predictions...")
y_pred_train = catboost_model.predict(X_train)
y_pred_test = catboost_model.predict(X_test)

# 8.2 Calculate accuracies
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Overfitting Check: {train_accuracy - test_accuracy:.4f}")

# 8.3 Detailed Classification Report
print("\n8.3 Classification Report:")
class_names = target_encoder.classes_
print(classification_report(y_test, y_pred_test, target_names=class_names))

# 8.4 Confusion Matrix
print("\n8.4 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# 8.5 Feature Importance
print("\n8.5 Feature Importance:")
feature_importance = catboost_model.get_feature_importance()
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance_df.head(10))

# ============================================================================
# STEP 9: FINAL MODEL EVALUATION
# ============================================================================
print("\n9. FINAL MODEL EVALUATION")
print("-" * 50)

# Use the trained model for final evaluation
best_model = catboost_model
y_pred_final = best_model.predict(X_test)

final_accuracy = accuracy_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final, average='weighted')

print(f"Final Test Accuracy: {final_accuracy:.4f}")
print(f"Final F1-Score (weighted): {final_f1:.4f}")

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=class_names))

# ============================================================================
# STEP 10: SAVE MODEL AND RESULTS
# ============================================================================
print("\n10. SAVING MODEL AND RESULTS")
print("-" * 50)

# Save the best model
best_model.save_model('catboost_sales_success_model.cbm')
print("Model saved as: catboost_sales_success_model.cbm")

# Save encoders
import pickle
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)
print("Encoders saved successfully!")

# Save feature importance
feature_importance_df.to_csv('catboost_feature_importance.csv', index=False)
print("Feature importance saved as: catboost_feature_importance.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': [target_encoder.classes_[i] for i in y_test],
    'Predicted': [target_encoder.classes_[i] for i in y_pred_final]
})
predictions_df.to_csv('catboost_predictions.csv', index=False)
print("Predictions saved as: catboost_predictions.csv")

# ============================================================================
# STEP 11: SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("CATBOOST PIPELINE SUMMARY")
print("=" * 80)
print(f"Dataset Size: {df.shape[0]:,} records, {df.shape[1]} features")
print(f"Training Set: {X_train.shape[0]:,} records")
print(f"Test Set: {X_test.shape[0]:,} records")
print(f"Number of Classes: {len(target_encoder.classes_)}")
print(f"Classes: {', '.join(target_encoder.classes_)}")
print(f"Final Model Accuracy: {final_accuracy:.4f}")
print(f"Final F1-Score: {final_f1:.4f}")
print("\nTop 5 Most Important Features:")
for i, row in feature_importance_df.head(5).iterrows():
    print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
print("\nPipeline completed successfully!")
print("=" * 80)