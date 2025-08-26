from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from product_store_predictor import ProductStoreSalesPredictor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store data
predictor = None
feature_importance = None
predictions = None

def load_data():
    """Load model data and predictor"""
    global predictor, feature_importance, predictions
    
    try:
        # Load predictor
        predictor = ProductStoreSalesPredictor()
        
        # Load feature importance
        feature_importance = pd.read_csv('product_store_feature_importance.csv')
        
        # Load predictions
        predictions = pd.read_csv('product_store_predictions.csv')
        
        # Clean the predicted values - remove the list brackets and quotes
        predictions['Predicted'] = predictions['Predicted'].str.replace("['", "").str.replace("']", "")
        
        return True
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def create_feature_importance_chart():
    """Create feature importance visualization"""
    top_features = feature_importance.head(10)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features',
        labels={'importance': 'Importance (%)', 'feature': 'Features'},
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        title_font_size=16,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_model_performance_chart():
    """Create model performance visualization"""
    # Calculate accuracy by class
    class_accuracy = []
    for class_name in predictions['Actual'].unique():
        class_data = predictions[predictions['Actual'] == class_name]
        accuracy = (class_data['Actual'] == class_data['Predicted']).mean()
        class_accuracy.append({'Class': class_name, 'Accuracy': accuracy})
    
    accuracy_df = pd.DataFrame(class_accuracy)
    
    fig = px.bar(
        accuracy_df,
        x='Class',
        y='Accuracy',
        title='Model Accuracy by Sales Category',
        labels={'Accuracy': 'Accuracy (%)', 'Class': 'Sales Category'},
        color='Accuracy',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
    fig.update_layout(
        height=400, 
        title_font_size=16,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_class_distribution_chart():
    """Create class distribution pie chart"""
    class_dist = predictions['Actual'].value_counts()
    
    fig = px.pie(
        values=class_dist.values,
        names=class_dist.index,
        title='Sales Category Distribution in Test Set',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        height=400, 
        title_font_size=16,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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
    top_feature = feature_importance.iloc[0]['feature']
    top_importance = feature_importance.iloc[0]['importance']
    
    # Create charts
    feature_chart = create_feature_importance_chart()
    performance_chart = create_model_performance_chart()
    distribution_chart = create_class_distribution_chart()
    
    return render_template('dashboard.html',
                         overall_accuracy=f"{overall_accuracy:.1%}",
                         total_predictions=f"{total_predictions:,}",
                         top_feature=top_feature,
                         top_importance=f"{top_importance:.1f}%",
                         feature_chart=feature_chart,
                         performance_chart=performance_chart,
                         distribution_chart=distribution_chart)

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

@app.route('/performance')
def performance_page():
    """Model performance analysis page"""
    if not load_data():
        return render_template('error.html', error="Failed to load model data")
    
    # Calculate detailed performance metrics
    performance_data = []
    for class_name in predictions['Actual'].unique():
        class_data = predictions[predictions['Actual'] == class_name]
        accuracy = (class_data['Actual'] == class_data['Predicted']).mean()
        total_samples = len(class_data)
        correct_predictions = (class_data['Actual'] == class_data['Predicted']).sum()
        
        performance_data.append({
            'Sales Category': class_name,
            'Accuracy': f"{accuracy:.1%}",
            'Correct Predictions': correct_predictions,
            'Total Samples': total_samples
        })
    
    # Create charts
    feature_chart = create_feature_importance_chart()
    performance_chart = create_model_performance_chart()
    distribution_chart = create_class_distribution_chart()
    
    return render_template('performance.html',
                         performance_data=performance_data,
                         feature_importance=feature_importance.to_dict('records'),
                         feature_chart=feature_chart,
                         performance_chart=performance_chart,
                         distribution_chart=distribution_chart)

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

@app.route('/debug')
def debug_page():
    return render_template('debug.html')

if __name__ == '__main__':
    # Load data on startup
    if not load_data():
        print("Failed to load model data. Exiting.")
        exit(1)
    
    app.run(debug=False, host='0.0.0.0', port=5000)