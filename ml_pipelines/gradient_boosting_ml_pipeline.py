import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GRADIENT BOOSTING MACHINE LEARNING PIPELINE")
print("(Alternative to CatBoost with similar performance)")
print("=" * 80)

# ============================================================================
# STEP 1: PROBLEM UNDERSTANDING
# ============================================================================
print("\n1. PROBLEM UNDERSTANDING")
print("-" * 50)
print("Objective: Multi-class classification for Sales Success prediction")
print("Target Variable: SalesSuccess (Top Seller, Good Seller, Moderate Seller, Slow Seller)")
print("Algorithm: Gradient Boosting Classifier (similar to CatBoost)")
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

# 5.1 Prepare features for Gradient Boosting
print("\n5.1 Preparing features for Gradient Boosting...")

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

# 5.2 Encode categorical features
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

# 5.3 Feature Scaling
print("\n5.3 Scaling numerical features...")
scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])

print("Feature scaling completed.")

# ============================================================================
# STEP 6: DATA SPLITTING (70% Training, 30% Validation)
# ============================================================================
print("\n6. DATA SPLITTING")
print("-" * 50)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, 
    test_size=0.30, 
    random_state=42, 
    stratify=y_encoded
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training target distribution:")
print(pd.Series(y_train).value_counts())
print(f"Test target distribution:")
print(pd.Series(y_test).value_counts())

# ============================================================================
# STEP 7: MODEL TRAINING (GRADIENT BOOSTING)
# ============================================================================
print("\n7. GRADIENT BOOSTING MODEL TRAINING")
print("-" * 50)

# 7.1 Initialize Gradient Boosting Classifier
print("\n7.1 Initializing Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbose=1
)

# 7.2 Train the model
print("\n7.2 Training Gradient Boosting model...")
gb_model.fit(X_train, y_train)

print("Gradient Boosting model training completed!")

# ============================================================================
# STEP 8: MODEL EVALUATION
# ============================================================================
print("\n8. MODEL EVALUATION")
print("-" * 50)

# 8.1 Make predictions
print("\n8.1 Making predictions...")
y_pred_train = gb_model.predict(X_train)
y_pred_test = gb_model.predict(X_test)

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
feature_importance = gb_model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance_df.head(10))

# 8.6 Cross-validation
print("\n8.6 Cross-validation (5-fold):")
cv_scores = cross_val_score(gb_model, X_scaled, y_encoded, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# STEP 9: MODEL OPTIMIZATION
# ============================================================================
print("\n9. MODEL OPTIMIZATION")
print("-" * 50)

# 9.1 Hyperparameter tuning
print("\n9.1 Hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [4, 6, 8]
}

# Create a smaller sample for faster tuning
sample_size = min(10000, len(X_train))
sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train.iloc[sample_indices]
y_train_sample = y_train[sample_indices]

optimized_model = GradientBoostingClassifier(random_state=42)

grid_search = GridSearchCV(
    optimized_model, 
    param_grid, 
    cv=3, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("Starting hyperparameter tuning (using sample for speed)...")
grid_search.fit(X_train_sample, y_train_sample)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# ============================================================================
# STEP 10: FINAL MODEL TRAINING WITH BEST PARAMETERS
# ============================================================================
print("\n10. FINAL MODEL TRAINING")
print("-" * 50)

# Train final model with best parameters on full training set
final_model = GradientBoostingClassifier(
    **grid_search.best_params_,
    random_state=42,
    verbose=1
)

print("Training final model with best parameters on full training set...")
final_model.fit(X_train, y_train)

# ============================================================================
# STEP 11: FINAL MODEL EVALUATION
# ============================================================================
print("\n11. FINAL MODEL EVALUATION")
print("-" * 50)

# Make final predictions
y_pred_final = final_model.predict(X_test)

final_accuracy = accuracy_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final, average='weighted')

print(f"Final Test Accuracy: {final_accuracy:.4f}")
print(f"Final F1-Score (weighted): {final_f1:.4f}")

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=class_names))

# Final Feature Importance
final_feature_importance = final_model.feature_importances_
final_feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': final_feature_importance
}).sort_values('importance', ascending=False)

print("\nFinal Top 10 Most Important Features:")
print(final_feature_importance_df.head(10))

# ============================================================================
# STEP 12: SAVE MODEL AND RESULTS
# ============================================================================
print("\n12. SAVING MODEL AND RESULTS")
print("-" * 50)

# Save the model and preprocessing objects
import pickle

with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("Model saved as: gradient_boosting_model.pkl")

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Preprocessing objects saved successfully!")

# Save feature importance
final_feature_importance_df.to_csv('gradient_boosting_feature_importance.csv', index=False)
print("Feature importance saved as: gradient_boosting_feature_importance.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': [target_encoder.classes_[i] for i in y_test],
    'Predicted': [target_encoder.classes_[i] for i in y_pred_final]
})
predictions_df.to_csv('gradient_boosting_predictions.csv', index=False)
print("Predictions saved as: gradient_boosting_predictions.csv")

# Save model performance summary
performance_summary = {
    'Model': 'Gradient Boosting Classifier',
    'Dataset_Size': df.shape[0],
    'Features': df.shape[1] - 1,
    'Training_Size': X_train.shape[0],
    'Test_Size': X_test.shape[0],
    'Classes': len(target_encoder.classes_),
    'Final_Accuracy': final_accuracy,
    'Final_F1_Score': final_f1,
    'Best_Parameters': grid_search.best_params_,
    'CV_Score_Mean': cv_scores.mean(),
    'CV_Score_Std': cv_scores.std()
}

performance_df = pd.DataFrame([performance_summary])
performance_df.to_csv('model_performance_summary.csv', index=False)
print("Performance summary saved as: model_performance_summary.csv")

# ============================================================================
# STEP 13: SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("GRADIENT BOOSTING PIPELINE SUMMARY")
print("=" * 80)
print(f"Dataset Size: {df.shape[0]:,} records, {df.shape[1]} features")
print(f"Training Set: {X_train.shape[0]:,} records")
print(f"Test Set: {X_test.shape[0]:,} records")
print(f"Number of Classes: {len(target_encoder.classes_)}")
print(f"Classes: {', '.join(target_encoder.classes_)}")
print(f"Final Model Accuracy: {final_accuracy:.4f}")
print(f"Final F1-Score: {final_f1:.4f}")
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print("\nTop 5 Most Important Features:")
for i, row in final_feature_importance_df.head(5).iterrows():
    print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
print("\nFiles Generated:")
print("  - gradient_boosting_model.pkl")
print("  - label_encoders.pkl, target_encoder.pkl, scaler.pkl")
print("  - gradient_boosting_feature_importance.csv")
print("  - gradient_boosting_predictions.csv")
print("  - model_performance_summary.csv")
print("\nPipeline completed successfully!")
print("=" * 80)