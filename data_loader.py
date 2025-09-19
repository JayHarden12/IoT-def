import pandas as pd
import numpy as np
import os
import zipfile
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class NBalotDataLoader:
    """
    Data loader for the N-BaIoT dataset
    """
    
    def __init__(self, data_path: str = "N-BaIoT.zip"):
        self.data_path = data_path
        self.extracted_path = "N-BaIoT"
        self.data = None
        
    def extract_dataset(self) -> bool:
        """Extract the N-BaIoT dataset from zip file"""
        try:
            if not os.path.exists(self.data_path):
                print(f"Dataset file {self.data_path} not found!")
                return False
                
            with zipfile.ZipFile(self.data_path, 'r') as zip_ref:
                zip_ref.extractall(self.extracted_path)
            print(f"Dataset extracted to {self.extracted_path}")
            return True
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            return False
    
    def load_device_data(self, device_name: str) -> Optional[pd.DataFrame]:
        """
        Load data for a specific device
        
        Args:
            device_name: Name of the device (e.g., 'Danmini_Doorbell')
            
        Returns:
            DataFrame containing the device data
        """
        try:
            # Look for CSV files in the extracted directory
            device_files = []
            for root, dirs, files in os.walk(self.extracted_path):
                for file in files:
                    if file.endswith('.csv') and device_name in file:
                        device_files.append(os.path.join(root, file))
            
            if not device_files:
                print(f"No data files found for device: {device_name}")
                return None
            
            # Load the first matching file
            df = pd.read_csv(device_files[0])
            print(f"Loaded {len(df)} samples for device: {device_name}")
            return df
            
        except Exception as e:
            print(f"Error loading data for {device_name}: {e}")
            return None
    
    def load_all_devices(self) -> pd.DataFrame:
        """
        Load data from all available devices
        
        Returns:
            Combined DataFrame with all device data
        """
        if not os.path.exists(self.extracted_path):
            if not self.extract_dataset():
                return self._generate_sample_data()
        
        all_data = []
        device_names = [
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
        
        for device in device_names:
            device_data = self.load_device_data(device)
            if device_data is not None:
                device_data['device_name'] = device
                all_data.append(device_data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Loaded combined dataset with {len(combined_data)} samples")
            return combined_data
        else:
            print("No device data found, generating sample data...")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """
        Generate sample data for demonstration purposes
        This is used when the actual dataset is not available
        """
        print("Generating sample N-BaIoT dataset...")
        
        np.random.seed(42)
        n_samples = 50000
        
        # Device names
        devices = [
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
        
        # Generate synthetic data based on N-BaIoT characteristics
        data = {
            'device_name': np.random.choice(devices, n_samples),
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
        attack_prob = 0.15  # 15% of data is attacks
        labels = np.random.choice([0, 1], n_samples, p=[1-attack_prob, attack_prob])
        
        # Add attack patterns
        attack_indices = np.where(labels == 1)[0]
        attack_types = ['DoS', 'DDoS', 'Botnet', 'Reconnaissance']
        
        for idx in attack_indices:
            attack_type = np.random.choice(attack_types)
            
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
        df.loc[attack_indices, 'attack_type'] = np.random.choice(attack_types, len(attack_indices))
        
        print(f"Generated sample dataset with {len(df)} samples")
        return df
    
    def get_device_list(self) -> list:
        """Get list of available devices"""
        return [
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
    
    def get_attack_types(self) -> list:
        """Get list of attack types"""
        return ['Normal', 'DoS', 'DDoS', 'Botnet', 'Reconnaissance']

if __name__ == "__main__":
    # Test the data loader
    loader = NBalotDataLoader()
    data = loader.load_all_devices()
    print(f"Loaded dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Attack distribution: {data['attack_type'].value_counts()}")
