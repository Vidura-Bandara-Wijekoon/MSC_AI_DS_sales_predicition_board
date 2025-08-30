<<<<<<< HEAD
# Sales Prediction Dashboard

A Flask-based web application for predicting sales success using CatBoost machine learning. This dashboard provides intelligent product-store combination analytics with interactive visualizations and real-time predictions.

## 🚀 Features

- **Interactive Dashboard**: View model performance and key insights
- **Real-time Prediction Interface**: Enter product and store details for instant predictions
- **Feature Importance Analysis**: Understand which factors drive sales success
- **Multiple ML Pipeline Implementations**: CatBoost, Gradient Boosting, and Multiclass Classification
- **Performance Metrics**: Detailed model accuracy and evaluation metrics
- **Beautiful UI**: Modern, responsive design with Bootstrap and Plotly visualizations

## 📁 Project Structure

```
sales-prediction-dashboard/
├── app/
│   ├── flask_app.py                 # Main Flask web application
│   ├── product_store_predictor.py   # ML predictor class
│   └── templates/                   # HTML templates
│       ├── base.html               # Base template
│       ├── dashboard.html          # Dashboard page
│       ├── predict.html            # Prediction interface
│       ├── performance.html        # Performance metrics
│       └── about.html              # About page
├── ml_pipelines/
│   ├── catboost_ml_pipeline.py     # CatBoost pipeline
│   ├── gradient_boosting_ml_pipeline.py # Gradient boosting pipeline
│   └── multiclass_classification_pipeline.py # Classification pipeline
├── data_processing/
│   ├── data_exploration.py         # Data analysis scripts
│   ├── feature_engineering.py      # Feature creation logic
│   └── generate_data.py            # Data generation scripts
├── analysis/
│   ├── analyze_features.py         # Feature analysis
│   └── business_analysis_report.py # Business insights
├── requirements.txt                 # Python dependencies
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sales-prediction-dashboard
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare model files**
   - Ensure you have the trained model files (*.cbm, *.pkl) in the project directory
   - Run the data generation scripts if needed:
     ```bash
     python data_processing/generate_data.py
     ```

## 🚀 Usage

### Running the Flask Application

```bash
cd app
python flask_app.py
```

The application will start on `http://localhost:5000`

### Available Pages

- **Dashboard** (`/`): Overview of model performance and key metrics
- **Predict** (`/predict`): Make predictions for new product-store combinations
- **Performance** (`/performance`): Detailed model performance analysis
- **About** (`/about`): Information about the system and technology stack

### API Endpoints

- **POST** `/api/predict`: Make predictions via API
  ```json
  {
    "ProductCategory": "Electronics",
    "StoreType": "Supermarket",
    "Price": 299.99,
    "CompetitorPrice": 319.99,
    "SeasonalFactor": 1.2,
    "PromotionActive": 1,
    "StoreSize": "Large",
    "StoreLocation": "Urban",
    "ShelfPlacement": "Eye Level",
    "InventoryLevel": "High",
    "DayOfWeek": "Friday",
    "Month": "December",
    "CustomerFootfall": 850,
    "LocalCompetition": "Medium",
    "WeatherCondition": "Clear",
    "EconomicIndicator": "Stable",
    "MarketingSpend": 5000
  }
  ```

## 🧠 Machine Learning Pipelines

### CatBoost Pipeline
- **File**: `ml_pipelines/catboost_ml_pipeline.py`
- **Features**: Gradient boosting with categorical feature support
- **Use Case**: Primary model for sales prediction

### Gradient Boosting Pipeline
- **File**: `ml_pipelines/gradient_boosting_ml_pipeline.py`
- **Features**: Traditional gradient boosting implementation
- **Use Case**: Alternative model comparison

### Multiclass Classification Pipeline
- **File**: `ml_pipelines/multiclass_classification_pipeline.py`
- **Features**: Multi-class classification approach
- **Use Case**: Categorical sales success prediction

## 📊 Data Processing

### Feature Engineering
- **File**: `data_processing/feature_engineering.py`
- **Purpose**: Create and transform features for model training
- **Features**: Price ratios, seasonal adjustments, categorical encoding

### Data Exploration
- **File**: `data_processing/data_exploration.py`
- **Purpose**: Analyze data patterns and distributions
- **Output**: Statistical summaries and visualizations

### Data Generation
- **File**: `data_processing/generate_data.py`
- **Purpose**: Generate synthetic training data
- **Features**: Realistic product-store combinations with sales outcomes

## 📈 Analysis Tools

### Feature Analysis
- **File**: `analysis/analyze_features.py`
- **Purpose**: Analyze feature importance and correlations
- **Output**: Feature importance rankings and visualizations

### Business Analysis
- **File**: `analysis/business_analysis_report.py`
- **Purpose**: Generate business insights and recommendations
- **Output**: Comprehensive business analysis reports

## 🔧 Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: CatBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Icons**: Font Awesome

## 📝 Model Features

The model uses the following features for prediction:

1. **Product Features**:
   - Product Category
   - Price
   - Competitor Price

2. **Store Features**:
   - Store Type
   - Store Size
   - Store Location
   - Customer Footfall

3. **Marketing Features**:
   - Promotion Active
   - Marketing Spend
   - Shelf Placement

4. **External Features**:
   - Seasonal Factor
   - Day of Week
   - Month
   - Weather Condition
   - Economic Indicator
   - Local Competition

5. **Inventory Features**:
   - Inventory Level

## 🎯 Model Performance

- **Algorithm**: CatBoost Classifier
- **Accuracy**: ~85-90% (varies by dataset)
- **Features**: 17 input features
- **Output**: Sales success probability (High/Medium/Low)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation in the `/docs` folder (if available)

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Advanced forecasting models
- [ ] A/B testing framework
- [ ] Mobile-responsive design improvements
- [ ] API rate limiting and authentication
- [ ] Docker containerization
- [ ] Cloud deployment guides

---

**Built with ❤️ using Flask, CatBoost, and modern web technologies**
=======
# MSC_AI_DS_sales_predicition_board
This is the official repository for the sales prediction project.
>>>>>>> origin/main
