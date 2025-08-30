from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from product_store_predictor import ProductStoreSalesPredictor
from eda_visualizations import SalesPredictionEDA
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store data
predictor = None
feature_importance = None
predictions = None
eda_analyzer = None

def load_data():
    """Load model data and predictor"""
    global predictor, feature_importance, predictions, eda_analyzer
    
    try:
        # Load predictor
        predictor = ProductStoreSalesPredictor()
        
        # Load feature importance
        feature_importance = pd.read_csv('product_store_feature_importance.csv')
        
        # Load predictions
        predictions = pd.read_csv('product_store_predictions.csv')
        
        # Clean the predicted values - remove the list brackets and quotes
        predictions['Predicted'] = predictions['Predicted'].str.replace("['", "").str.replace("']", "")
        
        # Initialize EDA analyzer
        eda_analyzer = SalesPredictionEDA()
        # Try to load real data, fallback to sample data if not available
        try:
            eda_analyzer.load_data('../data_processing/monthly_product_sales.csv')
        except:
            print("Using sample data for EDA visualizations")
            eda_analyzer._generate_sample_data()
        
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

# Chart creation functions removed - no longer needed

def create_prediction_confidence_chart(class_probabilities):
    """Create prediction confidence visualization"""
    classes = list(class_probabilities.keys())
    probabilities = list(class_probabilities.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
            marker_color=['#ff7f0e' if p == max(probabilities) else '#1f77b4' for p in probabilities]
        )
    ])
    
    fig.update_layout(
        title='Prediction Confidence by Category',
        xaxis_title='Sales Category',
        yaxis_title='Probability',
        height=400,
        title_font_size=16,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def dashboard():
    """Main dashboard page"""
    if not load_data():
        return render_template('error.html', error="Failed to load model data")
    
    # Calculate key metrics
    overall_accuracy = (predictions['Actual'] == predictions['Predicted']).mean()
    total_predictions = len(predictions)
    top_feature = feature_importance.iloc[0]['feature'] if not feature_importance.empty else "N/A"
    
    return render_template('dashboard.html',
                         overall_accuracy=f"{overall_accuracy:.1%}",
                         total_predictions=total_predictions,
                         top_feature=top_feature)

@app.route('/predict')
def predict_page():
    """Prediction interface page"""
    if not load_data():
        return render_template('error.html', error="Failed to load model data")
    
    # Get feature requirements
    feature_info = predictor.get_feature_requirements()
    
    return render_template('predict.html', feature_info=feature_info)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for making predictions"""
    try:
        data = request.get_json()
        
        # Make prediction
        result = predictor.predict_single_product_store(data)
        
        if result:
            # Create confidence chart
            confidence_chart = create_prediction_confidence_chart(result['class_probabilities'])
            
            return jsonify({
                'success': True,
                'predicted_class': result['predicted_class'],
                'confidence': f"{result['confidence']:.1%}",
                'class_probabilities': result['class_probabilities'],
                'confidence_chart': confidence_chart
            })
        else:
            return jsonify({'success': False, 'error': 'Prediction failed'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Performance route removed

@app.route('/metrics')
def metrics_page():
    """Model metrics comparison page"""
    try:
        # Load model metrics comparison data
        metrics_df = pd.read_csv('model_metrics_comparison.csv')
        
        # Charts removed - only showing table data
        
        # Convert metrics to records for table display
        metrics_data = metrics_df.to_dict('records')
        
        return render_template('metrics.html',
                             metrics_data=metrics_data)
    
    except Exception as e:
        return render_template('error.html', error=f"Failed to load metrics data: {str(e)}")

@app.route('/about')
def about_page():
    """About page"""
    return render_template('about.html')

@app.route('/test-js')
def test_js_page():
    """JavaScript test page"""
    return render_template('test_js.html')

@app.route('/test-predict')
def test_predict():
    return render_template('test_predict.html')

@app.route('/eda')
def eda_page():
    """EDA Analysis page with three key visualizations"""
    if not load_data():
        return render_template('error.html', error="Failed to load model data")
    
    try:
        # Generate all EDA visualizations
        visualizations = eda_analyzer.generate_all_visualizations()
        
        # Get insights summary
        insights = eda_analyzer.get_insights_summary()
        
        return render_template('eda.html',
                             price_sales_chart=visualizations['price_sales'],
                             promotion_impact_chart=visualizations['promotion_impact'],
                             location_customer_chart=visualizations['location_customer'],
                             insights=insights)
    
    except Exception as e:
        return render_template('error.html', error=f"Failed to generate EDA visualizations: {str(e)}")

@app.route('/debug')
def debug_page():
    return render_template('debug.html')

if __name__ == '__main__':
    # Load data on startup
    if not load_data():
        print("Failed to load model data. Exiting.")
        exit(1)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)