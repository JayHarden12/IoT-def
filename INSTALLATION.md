# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 4GB RAM minimum
- 2GB free disk space

## Installation Methods

### Method 1: Quick Start (Recommended)

1. **Download the project files** to your local machine

2. **Open a terminal/command prompt** in the project directory

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```
   or
   ```bash
   streamlit run app.py
   ```

5. **Open your web browser** and go to `http://localhost:8501`

### Method 2: Using the Run Script

**For Windows:**
```cmd
run.bat
```

**For Linux/Mac:**
```bash
python run.py
```

### Method 3: Using Docker

1. **Build the Docker image:**
   ```bash
   docker build -t iot-detection .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 iot-detection
   ```

3. **Or use Docker Compose:**
   ```bash
   docker-compose up
   ```

### Method 4: Development Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd n-balot
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   **Windows:**
   ```cmd
   venv\Scripts\activate
   ```
   
   **Linux/Mac:**
   ```bash
   source venv/bin/activate
   ```

4. **Install in development mode:**
   ```bash
   pip install -e .
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Troubleshooting

### Common Issues

1. **Python not found:**
   - Install Python from https://python.org
   - Make sure Python is added to your PATH

2. **Permission denied (Linux/Mac):**
   ```bash
   chmod +x run.py
   ```

3. **Port already in use:**
   - Change the port in the command: `streamlit run app.py --server.port 8502`
   - Or kill the process using port 8501

4. **Package installation fails:**
   - Update pip: `python -m pip install --upgrade pip`
   - Try installing packages individually

5. **Memory issues:**
   - Reduce the sample size in `config.py`
   - Close other applications to free up memory

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.9+ |
| RAM | 4GB | 8GB+ |
| Storage | 2GB | 5GB+ |
| CPU | 2 cores | 4+ cores |

## Configuration

### Environment Variables

You can set these environment variables to customize the application:

- `STREAMLIT_SERVER_PORT`: Port number (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)
- `DATASET_PATH`: Path to the N-BaIoT dataset
- `MODEL_SAVE_PATH`: Path to save trained models

### Configuration File

Edit `config.py` to modify:
- Dataset settings
- Model parameters
- Alert thresholds
- Visualization settings

## Data Setup

### Using Sample Data

The application includes sample data generation for demonstration purposes. No additional setup is required.

### Using Real N-BaIoT Dataset

1. **Download the N-BaIoT dataset** from the official source
2. **Place the dataset file** in the project directory
3. **Update the dataset path** in `config.py`
4. **Restart the application**

## Security Considerations

- The application is designed for local/private network use
- Do not expose the application to public networks without proper security measures
- Regularly update dependencies for security patches
- Use HTTPS in production environments

## Performance Optimization

### For Large Datasets

1. **Increase memory allocation:**
   ```bash
   export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
   ```

2. **Use data sampling:**
   - Modify `SAMPLE_SIZE` in `config.py`
   - Use data filtering options

3. **Enable caching:**
   - The application uses Streamlit's built-in caching
   - Models are cached after training

### For Production Deployment

1. **Use a reverse proxy** (nginx, Apache)
2. **Enable HTTPS**
3. **Set up monitoring and logging**
4. **Use a production WSGI server**

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs in the terminal
3. Check the Streamlit documentation
4. Open an issue in the project repository

## Deployment (Streamlit Community Cloud)

1. Push to GitHub with at least:
   - `app.py`
   - `requirements.txt`
   - `README.md` (optional but recommended)
   - Optional: `.streamlit/config.toml` for theme/settings
2. Avoid pushing large datasets, local environments, or artifacts:
   - Exclude: `venv/`, `__pycache__/`, `*.csv`, `*.zip`, `models/`, `reports/`, `data/`
3. In Streamlit Community Cloud:
   - Create a new app and connect your repo
   - Set Main file path to `app.py`
   - Deploy
4. If you need secrets or environment variables, add them in the app’s Settings → Secrets

### Windows PowerShell activation note

If `venv\Scripts\activate` fails in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\\venv\\Scripts\\Activate.ps1
```

## Updates

To update the application:

1. **Pull the latest changes:**
   ```bash
   git pull origin main
   ```

2. **Update dependencies:**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Restart the application**

## Uninstallation

To remove the application:

1. **Stop the application** (Ctrl+C)

2. **Remove the project directory:**
   ```bash
   rm -rf n-balot
   ```

3. **Remove virtual environment** (if used):
   ```bash
   rm -rf venv
   ```

4. **Uninstall packages** (optional):
   ```bash
   pip uninstall -r requirements.txt
   ```
