"""
Configuration file for the IoT Intrusion Detection System
"""

# Application settings
APP_TITLE = "IoT Intrusion Detection System"
APP_ICON = "🛡️"
APP_LAYOUT = "wide"

# Dataset settings
DATASET_PATH = "N-BaIoT.zip"
EXTRACTED_PATH = "N-BaIoT"
SAMPLE_SIZE = 10000  # For demo purposes

# Device names in the N-BaIoT dataset
DEVICE_NAMES = [
    'Danmini_Doorbell',
    'Ecobee_Thermostat', 
    'Ennio_Doorbell',
    'Philips_B120N10_Baby_Monitor',
    'Provision_PT_737E_Security_Camera',
    'Provision_PT_838_Security_Camera',
    'Samsung_SNH_1011_N_Webcam',
    'SimpleHome_XCS7_1002_WHT_Security_Camera',
    'SimpleHome_XCS7_1003_WHT_Security_Camera'
]

# Attack types
ATTACK_TYPES = [
    'Normal',
    'DoS',
    'DDoS', 
    'Botnet',
    'Reconnaissance'
]

# Feature columns for machine learning
FEATURE_COLUMNS = [
    'packet_size',
    'packet_count', 
    'bytes_sent',
    'bytes_received',
    'packet_rate',
    'connection_duration',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

# Categorical columns that need encoding
CATEGORICAL_COLUMNS = [
    'device_name',
    'protocol_type',
    'service',
    'flag'
]

# Model parameters
MODEL_PARAMS = {
    'Random Forest': {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': 10
    },
    'SVM': {
        'kernel': 'rbf',
        'random_state': 42,
        'C': 1.0
    },
    'Neural Network': {
        'hidden_layer_sizes': (100, 50),
        'random_state': 42,
        'max_iter': 500,
        'learning_rate': 'adaptive'
    },
    'Isolation Forest': {
        'contamination': 0.1,
        'random_state': 42,
        'n_estimators': 100
    }
}

# Visualization settings
PLOT_COLORS = {
    'normal': '#2E8B57',  # Sea Green
    'attack': '#DC143C',  # Crimson
    'primary': '#1f77b4',  # Blue
    'secondary': '#ff7f0e',  # Orange
    'accent': '#2ca02c'  # Green
}

# Alert thresholds
ALERT_THRESHOLDS = {
    'high_attack_rate': 0.2,  # 20%
    'high_packet_rate': 100,  # packets per second
    'high_failed_logins': 10,  # failed login attempts
    'suspicious_activity': 0.8  # confidence threshold
}

# File paths
MODEL_SAVE_PATH = "models/"
REPORT_SAVE_PATH = "reports/"
DATA_SAVE_PATH = "data/"

# Page titles
PAGE_TITLES = {
    'dashboard': '📊 Dashboard',
    'data_analysis': '📈 Data Analysis', 
    'model_training': '🤖 Model Training',
    'realtime_monitoring': '🔍 Real-time Monitoring',
    'attack_detection': '🎯 Attack Detection',
    'reports': '📋 Reports'
}
