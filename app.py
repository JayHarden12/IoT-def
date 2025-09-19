import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.decomposition import PCA
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="IoT Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class IoTIntrusionDetector:
    def __init__(self):
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def load_sample_data(self):
        """Load sample data for demonstration purposes"""
        # Generate synthetic IoT data based on N-BaIoT characteristics
        np.random.seed(42)
        n_samples = 10000
        
        # Simulate IoT device features
        data = {
            'device_id': np.random.choice(['Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell', 
                                        'Philips_B120N10_Baby_Monitor', 'Provision_PT_737E_Security_Camera',
                                        'Provision_PT_838_Security_Camera', 'Samsung_SNH_1011_N_Webcam',
                                        'SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera'], n_samples),
            'packet_size': np.random.normal(1000, 300, n_samples),
            'packet_count': np.random.poisson(50, n_samples),
            'bytes_sent': np.random.exponential(5000, n_samples),
            'bytes_received': np.random.exponential(3000, n_samples),
            'packet_rate': np.random.normal(10, 3, n_samples),
            'connection_duration': np.random.exponential(300, n_samples),
            'protocol_type': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
            'service': np.random.choice(['http', 'https', 'ftp', 'ssh', 'telnet'], n_samples),
            'flag': np.random.choice(['SF', 'S0', 'S1', 'REJ', 'RSTR', 'RSTO'], n_samples),
            'src_bytes': np.random.exponential(1000, n_samples),
            'dst_bytes': np.random.exponential(2000, n_samples),
            'land': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'wrong_fragment': np.random.poisson(0.1, n_samples),
            'urgent': np.random.poisson(0.05, n_samples),
            'hot': np.random.poisson(1, n_samples),
            'num_failed_logins': np.random.poisson(0.1, n_samples),
            'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'num_compromised': np.random.poisson(0.05, n_samples),
            'root_shell': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
            'su_attempted': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'num_root': np.random.poisson(0.1, n_samples),
            'num_file_creations': np.random.poisson(0.5, n_samples),
            'num_shells': np.random.poisson(0.01, n_samples),
            'num_access_files': np.random.poisson(0.1, n_samples),
            'num_outbound_cmds': np.random.poisson(0.01, n_samples),
            'is_host_login': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'is_guest_login': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'count': np.random.poisson(10, n_samples),
            'srv_count': np.random.poisson(5, n_samples),
            'serror_rate': np.random.beta(2, 5, n_samples),
            'srv_serror_rate': np.random.beta(2, 5, n_samples),
            'rerror_rate': np.random.beta(2, 5, n_samples),
            'srv_rerror_rate': np.random.beta(2, 5, n_samples),
            'same_srv_rate': np.random.beta(5, 2, n_samples),
            'diff_srv_rate': np.random.beta(2, 5, n_samples),
            'srv_diff_host_rate': np.random.beta(2, 5, n_samples),
            'dst_host_count': np.random.poisson(15, n_samples),
            'dst_host_srv_count': np.random.poisson(8, n_samples),
            'dst_host_same_srv_rate': np.random.beta(5, 2, n_samples),
            'dst_host_diff_srv_rate': np.random.beta(2, 5, n_samples),
            'dst_host_same_src_port_rate': np.random.beta(5, 2, n_samples),
            'dst_host_srv_diff_host_rate': np.random.beta(2, 5, n_samples),
            'dst_host_serror_rate': np.random.beta(2, 5, n_samples),
            'dst_host_srv_serror_rate': np.random.beta(2, 5, n_samples),
            'dst_host_rerror_rate': np.random.beta(2, 5, n_samples),
            'dst_host_srv_rerror_rate': np.random.beta(2, 5, n_samples)
        }
        
        # Create labels (0: Normal, 1: Attack)
        # Simulate different attack types
        attack_prob = 0.15  # 15% of data is attacks
        labels = np.random.choice([0, 1], n_samples, p=[1-attack_prob, attack_prob])
        
        # Add attack patterns
        attack_indices = np.where(labels == 1)[0]
        for idx in attack_indices:
            # Simulate different attack patterns
            attack_type = np.random.choice(['DoS', 'DDoS', 'Botnet', 'Reconnaissance'])
            
            if attack_type == 'DoS':
                data['packet_rate'][idx] *= 10
                data['bytes_sent'][idx] *= 5
            elif attack_type == 'DDoS':
                data['packet_rate'][idx] *= 20
                data['count'][idx] *= 3
            elif attack_type == 'Botnet':
                data['num_failed_logins'][idx] *= 10
                data['num_compromised'][idx] *= 5
            elif attack_type == 'Reconnaissance':
                data['num_access_files'][idx] *= 8
                data['srv_count'][idx] *= 2
        
        df = pd.DataFrame(data)
        df['label'] = labels
        df['attack_type'] = 'Normal'
        df.loc[attack_indices, 'attack_type'] = np.random.choice(['DoS', 'DDoS', 'Botnet', 'Reconnaissance'], len(attack_indices))
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for machine learning"""
        # Encode categorical variables
        categorical_cols = ['device_id', 'protocol_type', 'service', 'flag']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = self.label_encoder.fit_transform(df[col].astype(str))
        
        # Select features for training
        feature_cols = [col for col in df.columns if col not in ['label', 'attack_type']]
        X = df[feature_cols]
        y = df['label'] if 'label' in df.columns else None
        
        return X, y, feature_cols
    
    def train_models(self, X, y):
        """Train multiple machine learning models"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Isolation Forest': IsolationForest(contamination=0.1, random_state=42)
        }
        
        # Train models
        for name, model in models.items():
            if name == 'Isolation Forest':
                model.fit(X_train_scaled)
                y_pred = model.predict(X_test_scaled)
                y_pred = np.where(y_pred == -1, 1, 0)  # Convert to binary (1 = attack/outlier)
                # Use negative decision function as anomaly score (higher = more likely attack)
                if hasattr(model, 'decision_function'):
                    scores = -model.decision_function(X_test_scaled)
                    y_proba = scores  # raw scores acceptable for AUC metrics
                elif hasattr(model, 'score_samples'):
                    scores = -model.score_samples(X_test_scaled)
                    y_proba = scores
                else:
                    y_proba = None
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                elif hasattr(model, 'decision_function'):
                    scores = model.decision_function(X_test_scaled)
                    y_proba = scores
                else:
                    y_proba = None
            
            self.models[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': (roc_auc_score(y_test, y_proba) if y_proba is not None else None),
                'pr_auc': (average_precision_score(y_test, y_proba) if y_proba is not None else None),
                'predictions': y_pred,
                'probabilities': y_proba,
            }
        
        self.is_trained = True
        return X_test, y_test
    
    def predict_attack(self, data_point):
        """Predict if a data point is an attack"""
        if not self.is_trained:
            return "Model not trained yet"
        
        # Preprocess the data point
        data_point_scaled = self.scaler.transform([data_point])
        
        predictions = {}
        for name, model_info in self.models.items():
            if name == 'Isolation Forest':
                pred = model_info['model'].predict(data_point_scaled)
                pred = 1 if pred[0] == -1 else 0
            else:
                pred = model_info['model'].predict(data_point_scaled)[0]
            
            predictions[name] = pred
        
        # Majority voting
        attack_votes = sum(predictions.values())
        is_attack = attack_votes > len(predictions) / 2
        
        return {
            'is_attack': is_attack,
            'confidence': attack_votes / len(predictions),
            'individual_predictions': predictions
        }

def main():
    # Initialize the detector
    if 'detector' not in st.session_state:
        st.session_state.detector = IoTIntrusionDetector()
    
    detector = st.session_state.detector
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ IoT Intrusion Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Web-based IoT intrusion detection for home and small office networks using N-BaIoT")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Dashboard", 
        "Data Analysis", 
        "Model Training", 
        "Real-time Monitoring",
        "Attack Detection",
        "Reports"
    ])
    
    if page == "Dashboard":
        show_dashboard(detector)
    elif page == "Data Analysis":
        show_data_analysis(detector)
    elif page == "Model Training":
        show_model_training(detector)
    elif page == "Real-time Monitoring":
        show_realtime_monitoring(detector)
    elif page == "Attack Detection":
        show_attack_detection(detector)
    elif page == "Reports":
        show_reports(detector)

def show_dashboard(detector):
    """Display the main dashboard"""
    st.header("📊 System Overview")
    
    # Load sample data if not already loaded
    if detector.data is None:
        with st.spinner("Loading sample data..."):
            detector.data = detector.load_sample_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Devices", len(detector.data['device_id'].unique()))
    
    with col2:
        total_packets = detector.data['packet_count'].sum()
        st.metric("Total Packets", f"{total_packets:,}")
    
    with col3:
        attack_count = detector.data['label'].sum()
        st.metric("Detected Attacks", attack_count)
    
    with col4:
        attack_rate = (attack_count / len(detector.data)) * 100
        st.metric("Attack Rate", f"{attack_rate:.2f}%")
    
    # Attack distribution
    st.subheader("Attack Distribution by Device")
    attack_by_device = detector.data.groupby('device_id')['label'].sum().reset_index()
    attack_by_device.columns = ['Device', 'Attack Count']
    
    fig = px.bar(attack_by_device, x='Device', y='Attack Count', 
                 title="Number of Attacks by Device Type")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Network Activity")
    recent_data = detector.data.tail(100)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Packet Rate Over Time', 'Bytes Transferred', 
                       'Connection Duration', 'Failed Login Attempts'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Packet rate
    fig.add_trace(
        go.Scatter(y=recent_data['packet_rate'], mode='lines', name='Packet Rate'),
        row=1, col=1
    )
    
    # Bytes transferred
    fig.add_trace(
        go.Scatter(y=recent_data['bytes_sent'], mode='lines', name='Bytes Sent'),
        row=1, col=2
    )
    
    # Connection duration
    fig.add_trace(
        go.Scatter(y=recent_data['connection_duration'], mode='lines', name='Connection Duration'),
        row=2, col=1
    )
    
    # Failed logins
    fig.add_trace(
        go.Scatter(y=recent_data['num_failed_logins'], mode='lines', name='Failed Logins'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_data_analysis(detector):
    """Display data analysis page"""
    st.header("📈 Data Analysis")
    
    if detector.data is None:
        with st.spinner("Loading sample data..."):
            detector.data = detector.load_sample_data()
    
    # Data overview
    st.subheader("Dataset Overview")
    st.write(f"Total samples: {len(detector.data):,}")
    st.write(f"Features: {len(detector.data.columns)}")
    st.write(f"Attack samples: {detector.data['label'].sum():,}")
    st.write(f"Normal samples: {(detector.data['label'] == 0).sum():,}")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(detector.data.head(10))
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(detector.data.describe())
    
    # Correlation matrix
    st.subheader("Feature Correlation Matrix")
    numeric_cols = detector.data.select_dtypes(include=[np.number]).columns
    corr_matrix = detector.data[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    # Attack type distribution
    st.subheader("Attack Type Distribution")
    attack_types = detector.data['attack_type'].value_counts()
    
    fig = px.pie(values=attack_types.values, 
                 names=attack_types.index,
                 title="Distribution of Attack Types")
    st.plotly_chart(fig, use_container_width=True)

def show_model_training(detector):
    """Display model training page"""
    st.header("🤖 Model Training")
    
    if detector.data is None:
        with st.spinner("Loading sample data..."):
            detector.data = detector.load_sample_data()
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models..."):
            X, y, feature_cols = detector.preprocess_data(detector.data)
            X_test, y_test = detector.train_models(X, y)
            
            st.success("Models trained successfully!")
            
            # Display model performance
            st.subheader("Model Performance")
            
            for name, model_info in detector.models.items():
                st.write(f"**{name}**")
                mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
                with mcol1:
                    st.metric("Accuracy", f"{model_info['accuracy']:.3f}")
                with mcol2:
                    st.metric("Precision", f"{(model_info.get('precision') or 0):.3f}")
                with mcol3:
                    st.metric("Recall", f"{(model_info.get('recall') or 0):.3f}")
                with mcol4:
                    st.metric("F1", f"{(model_info.get('f1') or 0):.3f}")
                with mcol5:
                    roc_val = model_info.get('roc_auc')
                    st.metric("ROC-AUC", f"{roc_val:.3f}" if roc_val is not None else "N/A")
                with mcol6:
                    pr_val = model_info.get('pr_auc')
                    st.metric("PR-AUC", f"{pr_val:.3f}" if pr_val is not None else "N/A")
            
            # Confusion matrices for all models
            st.subheader("Confusion Matrices")
            for name, model_info in detector.models.items():
                y_pred = model_info['predictions']
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    title=f"Confusion Matrix - {name}",
                    labels=dict(x="Predicted", y="Actual"),
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)

            # ROC curves (combined)
            st.subheader("ROC Curves")
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))
            added_any = False
            for name, model_info in detector.models.items():
                y_score = model_info.get('probabilities')
                if y_score is None:
                    continue
                fpr, tpr, _ = roc_curve(y_test, y_score)
                auc_val = roc_auc_score(y_test, y_score)
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc_val:.3f})"))
                added_any = True
            roc_fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=400)
            if added_any:
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.info("No probability scores available to plot ROC curves.")

            # Precision-Recall curves (combined)
            st.subheader("Precision-Recall Curves")
            pr_fig = go.Figure()
            added_any_pr = False
            # Baseline: positive class ratio
            pos_ratio = (y_test == 1).mean() if hasattr(y_test, 'mean') else float(np.mean(y_test))
            pr_fig.add_trace(go.Scatter(x=[0, 1], y=[pos_ratio, pos_ratio], mode='lines', name='Baseline', line=dict(dash='dash', color='gray')))
            for name, model_info in detector.models.items():
                y_score = model_info.get('probabilities')
                if y_score is None:
                    continue
                precision, recall, _ = precision_recall_curve(y_test, y_score)
                ap = average_precision_score(y_test, y_score)
                pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f"{name} (AP={ap:.3f})"))
                added_any_pr = True
            pr_fig.update_layout(xaxis_title='Recall', yaxis_title='Precision', height=400)
            if added_any_pr:
                st.plotly_chart(pr_fig, use_container_width=True)
            else:
                st.info("No probability scores available to plot PR curves.")

def show_realtime_monitoring(detector):
    """Display real-time monitoring page"""
    st.header("🔍 Real-time Monitoring")
    
    if not detector.is_trained:
        st.warning("Please train the models first in the Model Training page.")
        return
    
    st.subheader("Live Network Traffic")
    
    # Simulate real-time data
    if st.button("Start Monitoring", type="primary"):
        placeholder = st.empty()
        
        for i in range(50):  # Simulate 50 data points
            # Generate random data point
            sample_data = detector.data.sample(1).iloc[0]
            X, _, _ = detector.preprocess_data(sample_data.to_frame().T)
            
            # Make prediction
            prediction = detector.predict_attack(X.iloc[0])
            
            # Display result
            with placeholder.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Device", sample_data['device_id'])
                
                with col2:
                    status = "🚨 ATTACK" if prediction['is_attack'] else "✅ NORMAL"
                    st.metric("Status", status)
                
                with col3:
                    st.metric("Confidence", f"{prediction['confidence']:.2f}")
                
                # Show individual model predictions
                st.write("**Model Predictions:**")
                for model_name, pred in prediction['individual_predictions'].items():
                    st.write(f"- {model_name}: {'Attack' if pred else 'Normal'}")
                
                st.write("---")
            
            # Small delay to simulate real-time
            import time
            time.sleep(0.5)

def show_attack_detection(detector):
    """Display attack detection page"""
    st.header("🎯 Attack Detection")
    
    if not detector.is_trained:
        st.warning("Please train the models first in the Model Training page.")
        return
    
    st.subheader("Manual Attack Detection")
    
    # Input form for manual testing
    with st.form("attack_detection_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            device_id = st.selectbox("Device ID", detector.data['device_id'].unique())
            packet_size = st.number_input("Packet Size", min_value=0, value=1000)
            packet_count = st.number_input("Packet Count", min_value=0, value=50)
            bytes_sent = st.number_input("Bytes Sent", min_value=0, value=5000)
            bytes_received = st.number_input("Bytes Received", min_value=0, value=3000)
        
        with col2:
            packet_rate = st.number_input("Packet Rate", min_value=0.0, value=10.0)
            connection_duration = st.number_input("Connection Duration", min_value=0, value=300)
            protocol_type = st.selectbox("Protocol Type", ['TCP', 'UDP', 'ICMP'])
            service = st.selectbox("Service", ['http', 'https', 'ftp', 'ssh', 'telnet'])
            num_failed_logins = st.number_input("Failed Logins", min_value=0, value=0)
        
        submitted = st.form_submit_button("Analyze", type="primary")
        
        if submitted:
            # Create data point
            data_point = {
                'device_id': device_id,
                'packet_size': packet_size,
                'packet_count': packet_count,
                'bytes_sent': bytes_sent,
                'bytes_received': bytes_received,
                'packet_rate': packet_rate,
                'connection_duration': connection_duration,
                'protocol_type': protocol_type,
                'service': service,
                'num_failed_logins': num_failed_logins
            }
            
            # Fill missing features with sensible defaults
            for col in detector.data.columns:
                if col not in data_point and col not in ['label', 'attack_type']:
                    if pd.api.types.is_numeric_dtype(detector.data[col]):
                        data_point[col] = float(detector.data[col].mean())
                    else:
                        mode_series = detector.data[col].mode()
                        data_point[col] = mode_series.iloc[0] if not mode_series.empty else str(detector.data[col].iloc[0])
            
            # Preprocess and predict
            X, _, _ = detector.preprocess_data(pd.DataFrame([data_point]))
            prediction = detector.predict_attack(X.iloc[0])
            
            # Display results
            if prediction['is_attack']:
                st.error("🚨 **ATTACK DETECTED!**")
                st.write(f"Confidence: {prediction['confidence']:.2%}")
            else:
                st.success("✅ **NORMAL TRAFFIC**")
                st.write(f"Confidence: {prediction['confidence']:.2%}")
            
            # Show individual model predictions
            st.subheader("Individual Model Predictions")
            for model_name, pred in prediction['individual_predictions'].items():
                status = "Attack" if pred else "Normal"
                st.write(f"**{model_name}**: {status}")

def show_reports(detector):
    """Display reports page"""
    st.header("📋 Reports")
    
    if detector.data is None:
        with st.spinner("Loading sample data..."):
            detector.data = detector.load_sample_data()
    
    # Generate reports
    st.subheader("Security Report")
    
    # Attack statistics
    total_attacks = detector.data['label'].sum()
    total_samples = len(detector.data)
    attack_rate = (total_attacks / total_samples) * 100
    
    st.write(f"**Total Samples Analyzed**: {total_samples:,}")
    st.write(f"**Total Attacks Detected**: {total_attacks:,}")
    st.write(f"**Attack Rate**: {attack_rate:.2f}%")
    
    # Device-wise attack analysis
    st.subheader("Device-wise Attack Analysis")
    device_attacks = detector.data.groupby('device_id').agg({
        'label': ['count', 'sum'],
        'packet_count': 'mean',
        'bytes_sent': 'mean'
    }).round(2)
    
    device_attacks.columns = ['Total_Connections', 'Attack_Count', 'Avg_Packets', 'Avg_Bytes_Sent']
    device_attacks['Attack_Rate'] = (device_attacks['Attack_Count'] / device_attacks['Total_Connections'] * 100).round(2)
    
    st.dataframe(device_attacks)
    
    # Export functionality
    st.subheader("Export Report")
    
    if st.button("Generate CSV Report"):
        csv = device_attacks.to_csv()
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="iot_security_report.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
