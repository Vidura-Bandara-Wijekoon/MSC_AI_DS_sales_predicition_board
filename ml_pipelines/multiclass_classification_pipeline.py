import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MULTI-CLASS CLASSIFICATION PIPELINE")
print("Following Steps from Handwritten Process Flow")
print("=" * 80)

# ============================================================================
# STEP 1: PROBLEM UNDERSTANDING
# ============================================================================
print("\n" + "=" * 50)
print("STEP 1: PROBLEM UNDERSTANDING")
print("=" * 50)
print("Objective: Multi-class classification for Product Sales Success")
print("Target Variable: Sales_Success_Category (Top Seller, Good Performer, Average, Poor)")
print("Features: Product attributes, pricing, promotion status, customer profiling")
print("Business Goal: Predict sales performance to optimize product selection")

# ============================================================================
# STEP 2: DATA COLLECTION
# ============================================================================
print("\n" + "=" * 50)
print("STEP 2: DATA COLLECTION")
print("=" * 50)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('d:/Dataset_new/final_modeling_dataset.csv')
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display basic information
print("\nDataset Info:")
print(f"- Total records: {len(df):,}")
print(f"- Features: {df.shape[1] - 1}")
print(f"- Target classes: {df['Sales_Success_Category'].nunique()}")
print(f"- Date range: Available in engineered features")

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 50)
print("STEP 3: DATA PREPROCESSING")
print("=" * 50)

# 3.1 Handle Missing Values
print("\n3.1 Checking for missing values...")
missing_values = df.isnull().sum()
print(f"Missing values found: {missing_values.sum()}")
if missing_values.sum() > 0:
    print("Missing values by column:")
    print(missing_values[missing_values > 0])
    # Fill missing values if any
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])  # For categorical
    print("Missing values handled.")
else:
    print("No missing values found.")

# 3.2 Remove Duplicates
print("\n3.2 Checking for duplicates...")
duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. New shape: {df.shape}")

# 3.3 Data Type Optimization
print("\n3.3 Optimizing data types...")
# Convert categorical columns to category type for memory efficiency
categorical_cols = ['SKU_Name', 'Category', 'Sub_Category', 'Brand', 
                   'Most_Common_Promotion_Type', 'Store_Location', 
                   'Dominant_Customer_Segment', 'Primary_Shelf_Placement',
                   'Sales_Success_Category']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
        print(f"Converted {col} to category type")

# 3.4 Feature Engineering for Text Processing (if needed)
print("\n3.4 Text preprocessing (SKU names, categories)...")
# Clean and standardize text fields
text_cols = ['SKU_Name', 'Category', 'Sub_Category', 'Brand']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()
        print(f"Standardized text in {col}")

print(f"\nPreprocessed dataset shape: {df.shape}")

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 50)
print("STEP 4: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 50)

# 4.1 Target Variable Distribution
print("\n4.1 Target Variable Analysis")
print("Sales Success Category Distribution:")
target_dist = df['Sales_Success_Category'].value_counts()
print(target_dist)
print(f"\nClass Balance:")
for category, count in target_dist.items():
    percentage = (count / len(df)) * 100
    print(f"{category}: {count:,} ({percentage:.1f}%)")

# 4.2 Numerical Features Analysis
print("\n4.2 Numerical Features Analysis")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical features: {len(numerical_cols)}")
print("Statistical Summary:")
print(df[numerical_cols].describe())

# 4.3 Categorical Features Analysis
print("\n4.3 Categorical Features Analysis")
categorical_features = df.select_dtypes(include=['category', 'object']).columns.tolist()
if 'Sales_Success_Category' in categorical_features:
    categorical_features.remove('Sales_Success_Category')

print(f"Categorical features: {len(categorical_features)}")
for col in categorical_features[:5]:  # Show first 5 to avoid too much output
    print(f"\n{col} - Unique values: {df[col].nunique()}")
    print(df[col].value_counts().head())

# 4.4 Promotion Status Analysis
print("\n4.4 Promotion Status Analysis")
print(f"Average Promotion Rate: {df['Promotion_Rate'].mean():.3f}")
print(f"Promotion Types:")
print(df['Most_Common_Promotion_Type'].value_counts())
print(f"\nPromotion Impact:")
print(f"Average Promotion Lift: {df['Promotion_Lift'].mean():.2f}%")

# 4.5 Customer Profiling Analysis
print("\n4.5 Customer Profiling Analysis")
print("Store Locations:")
print(df['Store_Location'].value_counts())
print("\nCustomer Segments:")
print(df['Dominant_Customer_Segment'].value_counts())

# 4.6 Correlation Analysis
print("\n4.6 Feature Correlation Analysis")
corr_matrix = df[numerical_cols].corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((
                corr_matrix.columns[i], 
                corr_matrix.columns[j], 
                corr_matrix.iloc[i, j]
            ))

if high_corr_pairs:
    print("High correlation pairs (>0.8):")
    for col1, col2, corr in high_corr_pairs:
        print(f"{col1} - {col2}: {corr:.3f}")
else:
    print("No high correlation pairs found (>0.8)")

# ============================================================================
# STEP 5: FEATURE EXTRACTION
# ============================================================================
print("\n" + "=" * 50)
print("STEP 5: FEATURE EXTRACTION")
print("=" * 50)

# 5.1 Prepare features for modeling
print("\n5.1 Preparing features for modeling...")

# Separate features and target
X = df.drop(['Sales_Success_Category'], axis=1)
y = df['Sales_Success_Category']

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# 5.2 Encode categorical variables
print("\n5.2 Encoding categorical variables...")
le_dict = {}
X_encoded = X.copy()

# Label encode categorical features
for col in categorical_features:
    if col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        le_dict[col] = le
        print(f"Encoded {col}: {len(le.classes_)} unique values")

# 5.3 Feature scaling
print("\n5.3 Feature scaling...")
scaler = StandardScaler()
numerical_features = X_encoded.select_dtypes(include=[np.number]).columns.tolist()

X_scaled = X_encoded.copy()
X_scaled[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])
print(f"Scaled {len(numerical_features)} numerical features")

# 5.4 Feature selection
print("\n5.4 Feature selection...")
selector = SelectKBest(score_func=f_classif, k=20)  # Select top 20 features
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X_scaled.columns[selector.get_support()].tolist()

print(f"Selected {len(selected_features)} most important features:")
for i, feature in enumerate(selected_features[:10]):
    print(f"{i+1}. {feature}")

# ============================================================================
# STEP 6: MODEL TRAINING (70% Training, 30% Validation)
# ============================================================================
print("\n" + "=" * 50)
print("STEP 6: MODEL TRAINING")
print("=" * 50)

# 6.1 Split data (70% training, 30% validation as per image)
print("\n6.1 Splitting data (70% training, 30% validation)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X_selected)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X_selected)*100:.1f}%)")

# Check class distribution in splits
print("\nClass distribution in training set:")
print(pd.Series(y_train).value_counts(normalize=True))
print("\nClass distribution in validation set:")
print(pd.Series(y_val).value_counts(normalize=True))

# 6.2 Train multiple models
print("\n6.2 Training multiple classification models...")

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr'),
    'SVM': SVC(random_state=42, probability=True, kernel='rbf')
}

model_results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_f1 = f1_score(y_val, y_pred_val, average='weighted')
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    model_results[name] = {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'val_f1_score': val_f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred_val
    }
    
    trained_models[name] = model
    
    print(f"{name} Results:")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Validation F1-Score: {val_f1:.4f}")
    print(f"  CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# STEP 7: MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 50)
print("STEP 7: MODEL EVALUATION")
print("=" * 50)

# 7.1 Find best model
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['val_accuracy'])
best_model = trained_models[best_model_name]

print(f"\n7.1 Best Model: {best_model_name}")
print(f"Best Validation Accuracy: {model_results[best_model_name]['val_accuracy']:.4f}")
print(f"Best Validation F1-Score: {model_results[best_model_name]['val_f1_score']:.4f}")

# 7.2 Detailed evaluation
print(f"\n7.2 Detailed Evaluation of {best_model_name}")
y_pred_best = model_results[best_model_name]['predictions']

print("\nClassification Report:")
print(classification_report(y_val, y_pred_best))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred_best)
print(cm)

# 7.3 Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n7.3 Top 10 Most Important Features ({best_model_name}):")
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================
print("\n" + "=" * 50)
print("STEP 8: SAVING RESULTS")
print("=" * 50)

# Save models and preprocessing objects
print("\nSaving models and preprocessing objects...")
joblib.dump(best_model, 'd:/Dataset_new/best_multiclass_model.pkl')
joblib.dump(scaler, 'd:/Dataset_new/feature_scaler.pkl')
joblib.dump(le_dict, 'd:/Dataset_new/label_encoders.pkl')
joblib.dump(selector, 'd:/Dataset_new/feature_selector.pkl')
joblib.dump(selected_features, 'd:/Dataset_new/selected_features.pkl')

# Save model comparison results
results_df = pd.DataFrame(model_results).T
results_df.to_csv('d:/Dataset_new/multiclass_model_comparison.csv')

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_val,
    'Predicted': y_pred_best
})
predictions_df.to_csv('d:/Dataset_new/validation_predictions.csv', index=False)

print("Files saved:")
print("- best_multiclass_model.pkl")
print("- feature_scaler.pkl")
print("- label_encoders.pkl")
print("- feature_selector.pkl")
print("- selected_features.pkl")
print("- multiclass_model_comparison.csv")
print("- validation_predictions.csv")

print("\n" + "=" * 80)
print("MULTI-CLASS CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY!")
print("All steps from the handwritten process flow have been implemented.")
print("=" * 80)

# Summary
print("\nPIPELINE SUMMARY:")
print(f"✓ Problem Understanding: Multi-class classification for sales success")
print(f"✓ Data Collection: {df.shape[0]:,} records loaded")
print(f"✓ Data Preprocessing: Missing values, duplicates, encoding completed")
print(f"✓ EDA: Target distribution, feature analysis, correlation analysis")
print(f"✓ Feature Extraction: {len(selected_features)} features selected")
print(f"✓ Model Training: 70-30 split, {len(models)} models trained")
print(f"✓ Model Evaluation: Best model - {best_model_name} ({model_results[best_model_name]['val_accuracy']:.4f} accuracy)")
print(f"✓ Results Saved: Models and evaluation metrics saved")