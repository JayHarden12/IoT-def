# IoT Intrusion Detection System

A web-based IoT intrusion detection system for home and small office networks using the N-BaIoT dataset. This application provides real-time monitoring, attack detection, and comprehensive analysis of IoT network traffic.

## Features

- **Real-time Monitoring**: Live monitoring of IoT network traffic
- **Attack Detection**: ML models (Random Forest, K-Nearest Neighbors, Isolation Forest)
- **Data Analysis**: Comprehensive analysis of network traffic patterns
- **Interactive Dashboard**: User-friendly interface for monitoring and analysis
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC; per-model confusion matrices; combined ROC and PR curves
- **Report Generation**: Automated security reports and alerts

## Installation

1. Clone or download this repository
2. (Optional) Create and activate a virtual environment
   - Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\\venv\\Scripts\\Activate.ps1
     ```
   - macOS/Linux:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

## Pages

### Dashboard
- System overview with key metrics
- Attack distribution by device
- Recent network activity visualization

### Data Analysis
- Dataset overview and statistics
- Feature correlation analysis
- Attack type distribution

### Model Training
- Train three ML models (Random Forest, K-Nearest Neighbors, Isolation Forest)
- Model performance comparison: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Confusion matrices for all models
- Combined ROC and Precision-Recall curve plots

### Real-time Monitoring
- Live network traffic monitoring
- Real-time attack detection
- Model prediction confidence scores

### Attack Detection
- Manual attack detection interface
- Custom parameter input
- Individual model predictions

### Reports
- Security report generation
- Device-wise attack analysis
- CSV export functionality

## Machine Learning Models

The system uses three machine learning models for robust attack detection:

1. **Random Forest**: Ensemble method for classification
2. **K-Nearest Neighbors (KNN)**: Distance-based classification
3. **Isolation Forest**: Unsupervised anomaly detection

### Evaluation Metrics and Plots
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Visuals: Confusion matrices (per model), ROC curves, Precision-Recall curves (combined across models)
- Isolation Forest AUCs are computed from anomaly scores (negative decision_function/score_samples), which are suitable for ranking-based metrics like ROC-AUC and PR-AUC.

## Dataset

The application uses the N-BaIoT dataset, which contains network traffic data from various IoT devices including:
- Danmini Doorbell
- Ecobee Thermostat
- Ennio Doorbell
- Philips Baby Monitor
- Provision Security Cameras
- Samsung Webcam
- SimpleHome Security Cameras

## Attack Types Detected

- **DoS (Denial of Service)**: High packet rate attacks
- **DDoS (Distributed DoS)**: Distributed high-volume attacks
- **Botnet**: Compromised device attacks
- **Reconnaissance**: Network scanning and probing

## Technical Requirements

- Python 3.8+ (3.9 recommended)
- 4GB RAM minimum
- Modern web browser

## Deployment (Streamlit Community Cloud)

1. Push the following to GitHub:
   - Required: `app.py`, `requirements.txt`, `README.md`
   - Optional: `.streamlit/config.toml` for theme/settings
   - Avoid pushing large data files (CSVs, zips) and local envs (`venv/`, `__pycache__/`)
2. In Streamlit Community Cloud, create a new app and connect the repo
3. Set Main file path to `app.py`
4. Deploy

Tip: if you need secrets or environment variables, add them in the app’s Settings → Secrets.

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests.
