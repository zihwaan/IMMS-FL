import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WADIDataset(Dataset):
    def __init__(self, X, y, sequence_length=60):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.sequence_length]
        y_label = self.y[idx + self.sequence_length - 1]
        return torch.FloatTensor(x_seq), torch.LongTensor([y_label])

class WADIDataLoader:
    def __init__(self, data_path: str = "WADI_FINAL_DATASET.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.attack_types = {
            'flow_pressure': ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', 
                             '1_AIT_005_PV', '1_FIT_001_PV', '1_LT_001_PV'],
            'sensor_hidden': ['2_DPIT_001_PV', '2_FIC_101_PV', '2_FIC_201_PV', '2_FIC_301_PV',
                            '2_FIC_401_PV', '2_FIC_501_PV', '2_FIC_601_PV', '2_FIT_001_PV'],
            'quality_pressure': ['2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', '2A_AIT_004_PV',
                               '2B_AIT_001_PV', '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV']
        }
        
    def load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess WADI dataset"""
        logger.info(f"Loading WADI dataset from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Dataset shape: {df.shape}")
        
        # Remove datetime columns
        df = df.drop(['Date', 'Time'], axis=1)
        
        # Get feature columns (all except Label)
        self.feature_columns = [col for col in df.columns if col != 'Label']
        logger.info(f"Number of features: {len(self.feature_columns)}")
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Separate features and labels
        X = df[self.feature_columns].values
        y = df['Label'].values
        
        # Convert to binary classification (0=normal, 1=attack)
        y = (y > 0).astype(int)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Normal samples: {np.sum(y == 0)}, Anomaly samples: {np.sum(y == 1)}")
        
        return X_scaled, y
    
    def create_federated_split(self, X, y, test_size=0.2, val_size=0.1) -> Dict:
        """Create federated data split based on attack types"""
        logger.info("Creating federated data split...")
        
        # First split: train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Second split: train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, stratify=y_train, random_state=42
        )
        
        # Get normal data for pretraining (first 14 days worth)
        normal_indices = np.where(y_train == 0)[0]
        pretrain_size = min(len(normal_indices), 20160)  # 14 days * 24 hours * 60 minutes
        X_pretrain = X_train[normal_indices[:pretrain_size]]
        y_pretrain = y_train[normal_indices[:pretrain_size]]
        
        # Create federated splits
        edges_data = self._create_edge_splits(X_train, y_train)
        
        return {
            'pretrain': {'X': X_pretrain, 'y': y_pretrain},
            'edges': edges_data,
            'val': {'X': X_val, 'y': y_val},
            'test': {'X': X_test, 'y': y_test}
        }
    
    def _create_edge_splits(self, X, y) -> Dict:
        """Create edge-specific data splits"""
        edges_data = {}
        
        # Get indices for normal and attack samples
        normal_indices = np.where(y == 0)[0]
        attack_indices = np.where(y == 1)[0]
        
        # Split normal data equally among edges
        normal_per_edge = len(normal_indices) // 3
        
        # For attack data, we'll simulate different attack patterns per edge
        # In real scenario, this would be based on actual attack types
        attack_per_edge = len(attack_indices) // 3
        
        for edge_id in range(3):
            # Normal data split
            start_normal = edge_id * normal_per_edge
            end_normal = (edge_id + 1) * normal_per_edge if edge_id < 2 else len(normal_indices)
            edge_normal_indices = normal_indices[start_normal:end_normal]
            
            # Attack data split
            start_attack = edge_id * attack_per_edge
            end_attack = (edge_id + 1) * attack_per_edge if edge_id < 2 else len(attack_indices)
            edge_attack_indices = attack_indices[start_attack:end_attack]
            
            # Combine indices
            edge_indices = np.concatenate([edge_normal_indices, edge_attack_indices])
            np.random.shuffle(edge_indices)
            
            edges_data[f'edge_{edge_id}'] = {
                'X': X[edge_indices],
                'y': y[edge_indices],
                'attack_type': list(self.attack_types.keys())[edge_id]
            }
            
            logger.info(f"Edge {edge_id}: {len(edge_indices)} samples "
                       f"(Normal: {len(edge_normal_indices)}, Attack: {len(edge_attack_indices)})")
        
        return edges_data
    
    def create_dataloaders(self, data_dict: Dict, batch_size=32, sequence_length=60) -> Dict:
        """Create PyTorch DataLoaders for all splits"""
        dataloaders = {}
        
        for split_name, data in data_dict.items():
            if split_name == 'edges':
                dataloaders[split_name] = {}
                for edge_name, edge_data in data.items():
                    dataset = WADIDataset(edge_data['X'], edge_data['y'], sequence_length)
                    dataloaders[split_name][edge_name] = DataLoader(
                        dataset, batch_size=batch_size, shuffle=True
                    )
            else:
                dataset = WADIDataset(data['X'], data['y'], sequence_length)
                dataloaders[split_name] = DataLoader(
                    dataset, batch_size=batch_size, shuffle=False
                )
        
        return dataloaders
    
    def get_data_stats(self, data_dict: Dict) -> Dict:
        """Get statistics about the data splits"""
        stats = {}
        
        for split_name, data in data_dict.items():
            if split_name == 'edges':
                stats[split_name] = {}
                for edge_name, edge_data in data.items():
                    X, y = edge_data['X'], edge_data['y']
                    stats[split_name][edge_name] = {
                        'total_samples': len(X),
                        'normal_samples': np.sum(y == 0),
                        'attack_samples': np.sum(y == 1),
                        'attack_ratio': np.sum(y == 1) / len(y),
                        'attack_type': edge_data['attack_type']
                    }
            else:
                X, y = data['X'], data['y']
                stats[split_name] = {
                    'total_samples': len(X),
                    'normal_samples': np.sum(y == 0),
                    'attack_samples': np.sum(y == 1),
                    'attack_ratio': np.sum(y == 1) / len(y)
                }
        
        return stats

def calculate_wasserstein_distance(data1: np.ndarray, data2: np.ndarray) -> float:
    """Calculate Wasserstein distance between two data distributions"""
    from scipy.stats import wasserstein_distance
    
    # Flatten data if needed
    if data1.ndim > 1:
        data1 = data1.flatten()
    if data2.ndim > 1:
        data2 = data2.flatten()
    
    return wasserstein_distance(data1, data2)

def simulate_distribution_shift(X: np.ndarray, shift_type: str = 'scale', 
                              intensity: float = 2.0) -> np.ndarray:
    """Simulate distribution shift in data"""
    X_shifted = X.copy()
    
    if shift_type == 'scale':
        X_shifted = X_shifted * intensity
    elif shift_type == 'offset':
        X_shifted = X_shifted + intensity
    elif shift_type == 'noise':
        noise = np.random.normal(0, intensity * np.std(X), X.shape)
        X_shifted = X_shifted + noise
    
    return X_shifted

if __name__ == "__main__":
    # Test the data loader
    loader = WADIDataLoader()
    X, y = loader.load_and_preprocess()
    data_splits = loader.create_federated_split(X, y)
    stats = loader.get_data_stats(data_splits)
    
    print("Data Statistics:")
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        if isinstance(split_stats, dict) and 'edges' not in split_name:
            for key, value in split_stats.items():
                print(f"  {key}: {value}")
        elif split_name == 'edges':
            for edge_name, edge_stats in split_stats.items():
                print(f"  {edge_name}:")
                for key, value in edge_stats.items():
                    print(f"    {key}: {value}")