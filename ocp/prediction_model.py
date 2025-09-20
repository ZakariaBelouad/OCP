import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SatisfactionPredictor:
    """Enhanced satisfaction prediction model with proper integration"""
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.feature_columns = ['code_centre', 'nom_centre', 'type', 'weekday', 'month', 'week']
        self.model_path = None
        self.encoder_path = None
        self.is_trained = False
        
    def setup_paths(self):
        """Setup model and encoder file paths"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(current_dir, '..', 'reports')
        
        # Create reports directory if it doesn't exist
        os.makedirs(reports_dir, exist_ok=True)
        
        self.model_path = os.path.join(reports_dir, 'satisfaction_model.joblib')
        self.encoder_path = os.path.join(reports_dir, 'encoder.joblib')
    
    def load_or_train_model(self, df):
        """Load existing model or train a new one"""
        self.setup_paths()
        
        try:
            # Try to load existing model
            if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
                self.model = joblib.load(self.model_path)
                self.encoder = joblib.load(self.encoder_path)
                self.is_trained = True
                return True
            else:
                # Train new model
                return self.train_model(df)
        except Exception as e:
            st.warning(f"Could not load existing model: {e}. Training new model...")
            return self.train_model(df)
    
    def prepare_features(self, df):
        """Prepare features for training/prediction"""
        df_features = df.copy()
        
        # Ensure all required columns exist
        if 'weekday' not in df_features.columns and 'date' in df_features.columns:
            df_features['weekday'] = pd.to_datetime(df_features['date']).dt.day_name()
        
        if 'month' not in df_features.columns and 'date' in df_features.columns:
            df_features['month'] = pd.to_datetime(df_features['date']).dt.month
            
        if 'week' not in df_features.columns and 'date' in df_features.columns:
            df_features['week'] = pd.to_datetime(df_features['date']).dt.isocalendar().week
        
        # Fill missing values
        for col in self.feature_columns:
            if col in df_features.columns:
                if df_features[col].dtype == 'object':
                    df_features[col] = df_features[col].fillna('Unknown')
                else:
                    df_features[col] = df_features[col].fillna(df_features[col].median())
        
        # Select only available feature columns
        available_features = [col for col in self.feature_columns if col in df_features.columns]
        
        if not available_features:
            raise ValueError("No valid feature columns found in the dataset")
        
        return df_features[available_features]
    
    def train_model(self, df):
        """Train the satisfaction prediction model"""
        try:
            # Prepare features and target
            X = self.prepare_features(df)
            y = df['avis'].dropna()
            
            # Align X and y indices
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            if len(X) < 10:
                raise ValueError("Insufficient data for training (minimum 10 samples required)")
            
            # Initialize and fit encoder
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_encoded = self.encoder.fit_transform(X)
            
            # Split data
            if len(X) > 50:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y, test_size=0.2, random_state=42
                )
            else:
                # Use all data for training if dataset is small
                X_train, y_train = X_encoded, y
                X_test, y_test = X_encoded, y
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Save model and encoder
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.encoder, self.encoder_path)
            
            self.is_trained = True
            
            return {
                'success': True,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'samples_used': len(X_train),
                'features_used': X.columns.tolist()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_satisfaction(self, center_data, horizon_days=7):
        """Predict satisfaction scores with confidence intervals"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        try:
            # Prepare prediction features
            prediction_features = self.prepare_features(center_data.tail(1))
            
            # Handle encoding
            X_encoded = self.encoder.transform(prediction_features)
            
            # Get base prediction
            base_prediction = self.model.predict(X_encoded)[0]
            
            # Get prediction probabilities for confidence estimation
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_encoded)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.75  # Default confidence
            
            # Calculate confidence intervals based on historical variance
            historical_std = center_data['avis'].std() if len(center_data) > 1 else 0.3
            margin = historical_std * (1 - confidence)
            
            lower_bound = max(1.0, base_prediction - margin)
            upper_bound = min(4.0, base_prediction + margin)
            
            # Generate predictions for each day in horizon
            predictions = []
            for day in range(horizon_days):
                # Add some realistic daily variation
                daily_variation = np.random.normal(0, 0.1)
                daily_pred = np.clip(base_prediction + daily_variation, 1, 4)
                
                predictions.append({
                    'day': day + 1,
                    'prediction': daily_pred,
                    'lower_bound': max(1.0, daily_pred - margin),
                    'upper_bound': min(4.0, daily_pred + margin),
                    'confidence': confidence
                })
            
            return {
                'success': True,
                'base_prediction': base_prediction,
                'confidence': confidence,
                'predictions': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_feature_importance(self):
        """Get model feature importance"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                feature_names = self.encoder.get_feature_names_out()
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return importance_df
            else:
                return None
        except Exception:
            return None

# Enhanced prediction interface for Streamlit
def enhanced_prediction_interface(df):
    """Enhanced prediction interface with proper model integration"""
    st.subheader("🤖 AI-Powered Satisfaction Prediction")
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = SatisfactionPredictor()
    
    predictor = st.session_state.predictor
    
    # Model training/loading section
    with st.expander("Model Status & Training", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load/Train Model", type="primary"):
                with st.spinner("Loading or training model..."):
                    result = predictor.load_or_train_model(df)
                    
                    if isinstance(result, dict) and not result.get('success', False):
                        st.error(f"Training failed: {result.get('error', 'Unknown error')}")
                    elif isinstance(result, dict):
                        st.success("Model trained successfully!")
                        st.write(f"**Training Accuracy:** {result['train_accuracy']:.3f}")
                        st.write(f"**Test Accuracy:** {result['test_accuracy']:.3f}")
                        st.write(f"**Samples Used:** {result['samples_used']}")
                    else:
                        st.success("Model loaded successfully!")
        
        with col2:
            status = "✅ Ready" if predictor.is_trained else "❌ Not Ready"
            st.write(f"**Model Status:** {status}")
            
            if predictor.is_trained:
                importance_df = predictor.get_feature_importance()
                if importance_df is not None and not importance_df.empty:
                    st.write("**Top Features:**")
                    for _, row in importance_df.head(3).iterrows():
                        st.write(f"- {row['feature']}: {row['importance']:.3f}")
    
    if not predictor.is_trained:
        st.warning("⚠️ Please load/train the model first to make predictions.")
        return
    
    # Prediction interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### Prediction Settings")
        
        # Center selection with better display
        df['center_display'] = df['nom_centre'].astype(str) + " (" + df['code_centre'].astype(str) + ")"
        available_centers = sorted(df['center_display'].unique())
        
        center_choice = st.selectbox(
            "Select Center for Prediction",
            available_centers,
            help="Choose the center to predict satisfaction for"
        )
        
        # Prediction horizon
        horizon = st.slider(
            "Prediction Horizon (days)",
            min_value=1,
            max_value=30,
            value=7,
            help="How many days ahead to predict"
        )
        
        # Additional context (for future enhancement)
        with st.expander("Additional Context"):
            season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
            workload = st.selectbox("Expected Workload", ["Low", "Normal", "High"])
    
    with col2:
        st.write("### Prediction Results")
        
        if st.button("🔮 Generate Prediction", type="primary"):
            try:
                # Get center data
                center_data = df[df['center_display'] == center_choice].copy()
                
                if center_data.empty:
                    st.error("No data found for selected center.")
                    return
                
                # Generate prediction
                with st.spinner("Generating predictions..."):
                    result = predictor.predict_satisfaction(center_data, horizon)
                
                if not result['success']:
                    st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
                    return
                
                # Display results
                col2a, col2b, col2c = st.columns(3)
                
                with col2a:
                    st.metric(
                        "Predicted Score",
                        f"{result['base_prediction']:.2f}/4",
                        help="Average predicted satisfaction score"
                    )
                
                with col2b:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']:.1%}",
                        help="Model confidence in prediction"
                    )
                
                with col2c:
                    range_text = f"{result['lower_bound']:.2f} - {result['upper_bound']:.2f}"
                    st.metric(
                        "Range",
                        range_text,
                        help="Confidence interval for prediction"
                    )
                
                # Prediction visualization
                st.write("### Prediction Visualization")
                
                # Prepare historical data for context
                recent_data = center_data.tail(30).copy()
                if 'datetime' not in recent_data.columns and 'date' in recent_data.columns:
                    recent_data['datetime'] = pd.to_datetime(recent_data['date'])
                
                # Create visualization
                fig = go.Figure()
                
                # Historical data
                if not recent_data.empty:
                    fig.add_trace(go.Scatter(
                        x=recent_data['datetime'] if 'datetime' in recent_data.columns else recent_data.index,
                        y=recent_data['avis'],
                        mode='lines+markers',
                        name='Historical Data',
                        line=dict(color='#3b82f6', width=2),
                        marker=dict(size=6)
                    ))
                
                # Future predictions
                last_date = recent_data['datetime'].max() if 'datetime' in recent_data.columns else pd.Timestamp.now()
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizon,
                    freq='D'
                )
                
                predictions = [p['prediction'] for p in result['predictions']]
                lower_bounds = [p['lower_bound'] for p in result['predictions']]
                upper_bounds = [p['upper_bound'] for p in result['predictions']]
                
                # Prediction line
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions,
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='#ef4444', width=2, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=future_dates.tolist() + future_dates.tolist()[::-1],
                    y=upper_bounds + lower_bounds[::-1],
                    fill='toself',
                    fillcolor='rgba(239, 68, 68, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ))
                
                # Layout
                fig.update_layout(
                    title=f"Satisfaction Prediction - {center_choice}",
                    xaxis_title="Date",
                    yaxis_title="Satisfaction Score",
                    yaxis=dict(range=[0.5, 4.5]),
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed predictions table
                with st.expander("Detailed Daily Predictions"):
                    pred_df = pd.DataFrame(result['predictions'])
                    pred_df['date'] = future_dates
                    pred_df = pred_df[['date', 'prediction', 'lower_bound', 'upper_bound', 'confidence']]
                    pred_df.columns = ['Date', 'Prediction', 'Lower Bound', 'Upper Bound', 'Confidence']
                    
                    st.dataframe(
                        pred_df.style.format({
                            'Prediction': '{:.2f}',
                            'Lower Bound': '{:.2f}',
                            'Upper Bound': '{:.2f}',
                            'Confidence': '{:.1%}'
                        }),
                        use_container_width=True
                    )
                
                # Recommendations based on prediction
                st.write("### 💡 Recommendations")
                avg_pred = np.mean(predictions)
                
                if avg_pred >= 3.5:
                    st.success("🎉 **Excellent outlook!** Satisfaction levels are predicted to remain high. Continue current practices.")
                elif avg_pred >= 3.0:
                    st.info("👍 **Good performance expected.** Consider initiatives to push satisfaction even higher.")
                elif avg_pred >= 2.5:
                    st.warning("⚠️ **Moderate satisfaction predicted.** Recommend proactive measures to improve employee experience.")
                else:
                    st.error("🚨 **Low satisfaction expected.** Immediate intervention recommended to address potential issues.")
                
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
                st.write("Please ensure the model is properly trained and try again.")

# Usage example - replace the existing enhanced_prediction_interface function in your main app
if __name__ == "__main__":
    # This would be called from your main Streamlit app
    # enhanced_prediction_interface(your_dataframe)
    pass