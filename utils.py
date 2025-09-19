"""
Utility functions for the IoT Intrusion Detection System
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Data preprocessing utilities"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform the data
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Separate features and target
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df
        
        # Encode categorical variables
        categorical_cols = ['device_name', 'protocol_type', 'service', 'flag']
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        self.is_fitted = True
        return X_scaled, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        X = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['device_name', 'protocol_type', 'service', 'flag']
        for col in categorical_cols:
            if col in X.columns and col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        # Scale numerical features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled

class ModelEvaluator:
    """Model evaluation utilities"""
    
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: List[str] = None, title: str = "Confusion Matrix"):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            title: Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = ['Normal', 'Attack']
        
        fig = px.imshow(cm, 
                       text_auto=True,
                       title=title,
                       labels=dict(x="Predicted", y="Actual"),
                       x=labels,
                       y=labels,
                       color_continuous_scale='Blues')
        
        return fig
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      model_name: str = "Model") -> go.Figure:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
        """
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, 
                                mode='lines',
                                name=f'{model_name} (AUC = {roc_auc:.2f})',
                                line=dict(color='blue', width=2)))
        
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random Classifier',
                                line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        
        return fig

class DataVisualizer:
    """Data visualization utilities"""
    
    @staticmethod
    def plot_attack_distribution(df: pd.DataFrame, device_col: str = 'device_name', 
                                attack_col: str = 'attack_type') -> go.Figure:
        """
        Plot attack distribution by device
        
        Args:
            df: Input DataFrame
            device_col: Device column name
            attack_col: Attack type column name
            
        Returns:
            Plotly figure
        """
        attack_by_device = df.groupby([device_col, attack_col]).size().reset_index(name='count')
        
        fig = px.bar(attack_by_device, x=device_col, y='count', color=attack_col,
                    title="Attack Distribution by Device Type",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_xaxis(tickangle=45)
        fig.update_layout(height=500)
        
        return fig
    
    @staticmethod
    def plot_feature_distribution(df: pd.DataFrame, feature: str, 
                                by_attack: bool = True) -> go.Figure:
        """
        Plot feature distribution
        
        Args:
            df: Input DataFrame
            feature: Feature name to plot
            by_attack: Whether to separate by attack type
            
        Returns:
            Plotly figure
        """
        if by_attack and 'attack_type' in df.columns:
            fig = px.histogram(df, x=feature, color='attack_type',
                             title=f"Distribution of {feature} by Attack Type",
                             marginal="box",
                             color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            fig = px.histogram(df, x=feature, title=f"Distribution of {feature}")
        
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, title: str = "Feature Correlation Matrix") -> go.Figure:
        """
        Plot correlation matrix
        
        Args:
            df: Input DataFrame
            title: Plot title
            
        Returns:
            Plotly figure
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       title=title,
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        
        fig.update_layout(height=600)
        return fig
    
    @staticmethod
    def plot_time_series(df: pd.DataFrame, time_col: str = None, 
                        value_cols: List[str] = None, 
                        title: str = "Time Series Analysis") -> go.Figure:
        """
        Plot time series data
        
        Args:
            df: Input DataFrame
            time_col: Time column name
            value_cols: List of value columns to plot
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if time_col is None:
            time_col = df.index
        
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns[:5]  # First 5 numeric columns
        
        fig = make_subplots(
            rows=len(value_cols), cols=1,
            subplot_titles=value_cols,
            vertical_spacing=0.05
        )
        
        for i, col in enumerate(value_cols, 1):
            fig.add_trace(
                go.Scatter(x=time_col, y=df[col], mode='lines', name=col),
                row=i, col=1
            )
        
        fig.update_layout(height=200 * len(value_cols), title=title)
        return fig

class AlertSystem:
    """Alert system for intrusion detection"""
    
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            'high_attack_rate': 0.2,
            'high_packet_rate': 100,
            'high_failed_logins': 10,
            'suspicious_activity': 0.8
        }
        self.alerts = []
    
    def check_alerts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for alert conditions
        
        Args:
            data: Dictionary containing current metrics
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        # Check attack rate
        if 'attack_rate' in data and data['attack_rate'] > self.thresholds['high_attack_rate']:
            alerts.append({
                'type': 'HIGH_ATTACK_RATE',
                'message': f"High attack rate detected: {data['attack_rate']:.2%}",
                'severity': 'HIGH',
                'timestamp': pd.Timestamp.now()
            })
        
        # Check packet rate
        if 'packet_rate' in data and data['packet_rate'] > self.thresholds['high_packet_rate']:
            alerts.append({
                'type': 'HIGH_PACKET_RATE',
                'message': f"High packet rate detected: {data['packet_rate']:.1f} packets/sec",
                'severity': 'MEDIUM',
                'timestamp': pd.Timestamp.now()
            })
        
        # Check failed logins
        if 'failed_logins' in data and data['failed_logins'] > self.thresholds['high_failed_logins']:
            alerts.append({
                'type': 'HIGH_FAILED_LOGINS',
                'message': f"High number of failed logins: {data['failed_logins']}",
                'severity': 'HIGH',
                'timestamp': pd.Timestamp.now()
            })
        
        # Check suspicious activity
        if 'confidence' in data and data['confidence'] > self.thresholds['suspicious_activity']:
            alerts.append({
                'type': 'SUSPICIOUS_ACTIVITY',
                'message': f"Suspicious activity detected with confidence: {data['confidence']:.2%}",
                'severity': 'CRITICAL',
                'timestamp': pd.Timestamp.now()
            })
        
        self.alerts.extend(alerts)
        return alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent alerts
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample data for testing
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    data = {
        'device_name': np.random.choice(['Device_A', 'Device_B', 'Device_C'], n_samples),
        'packet_size': np.random.normal(1000, 300, n_samples),
        'packet_count': np.random.poisson(50, n_samples),
        'bytes_sent': np.random.exponential(5000, n_samples),
        'packet_rate': np.random.normal(10, 3, n_samples),
        'attack_type': np.random.choice(['Normal', 'DoS', 'DDoS'], n_samples, p=[0.8, 0.1, 0.1]),
        'label': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test the utilities
    print("Testing utility functions...")
    
    # Test data preprocessor
    sample_data = create_sample_data(100)
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(sample_data)
    print(f"Preprocessed data shape: {X.shape}")
    
    # Test model evaluator
    y_true = np.random.choice([0, 1], 100)
    y_pred = np.random.choice([0, 1], 100)
    metrics = ModelEvaluator.evaluate_model(y_true, y_pred)
    print(f"Model metrics: {metrics}")
    
    # Test alert system
    alert_system = AlertSystem()
    test_data = {'attack_rate': 0.25, 'packet_rate': 150, 'failed_logins': 15}
    alerts = alert_system.check_alerts(test_data)
    print(f"Generated alerts: {len(alerts)}")
    
    print("Utility functions tested successfully!")
