# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class WADIDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, window_size: int = 100):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx):
        window_data = self.data[idx:idx + self.window_size]
        label = self.labels[idx + self.window_size - 1]
        return torch.FloatTensor(window_data), torch.LongTensor([label])

def load_wadi_dataset(csv_path: str = 'WADI_FINAL_DATASET.csv') -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess WADI dataset"""
    df = pd.read_csv(csv_path)
    
    # Remove Date, Time columns and separate Label
    feature_cols = [col for col in df.columns if col not in ['Date', 'Time', 'Label']]
    labels = df['Label'].values
    features = df[feature_cols].values
    
    # Handle NaN values
    features = np.nan_to_num(features, nan=0.0)
    
    return features, labels, feature_cols

def create_attack_type_mapping() -> Dict[str, List[str]]:
    """Attack type to sensor group mapping"""
    return {
        'flow_pressure': [  # Edge 0: Flow/Pressure related
            '1_FIT_001_PV', '2_FIT_001_PV', '2_FIT_002_PV', '2_FIT_003_PV', '3_FIT_001_PV',
            '2_DPIT_001_PV', '2_PIT_001_PV', '2_PIT_002_PV', '2_PIT_003_PV',
            '2_FIC_101_PV', '2_FIC_201_PV', '2_FIC_301_PV', '2_FIC_401_PV', '2_FIC_501_PV', '2_FIC_601_PV'
        ],
        'sensor_hidden': [  # Edge 1: Sensor/Hidden related
            '1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', '1_AIT_005_PV',
            '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', '2A_AIT_004_PV',
            '2B_AIT_001_PV', '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV',
            '3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_004_PV', '3_AIT_005_PV'
        ],
        'quality_water': [  # Edge 2: Quality/Water pressure related
            '1_LT_001_PV', '2_LT_001_PV', '2_LT_002_PV', '3_LT_001_PV',
            '1_LS_001_AL', '1_LS_002_AL', '3_LS_001_AL',
            'LEAK_DIFF_PRESSURE', 'TOTAL_CONS_REQUIRED_FLOW'
        ]
    }

def distribute_data_by_attack_type(features: np.ndarray, labels: np.ndarray, 
                                 feature_cols: List[str], normal_ratio: float = 0.7) -> Dict[int, Dict]:
    """Distribute data across 3 edges by attack type"""
    
    attack_mapping = create_attack_type_mapping()
    
    # Find sensor indices for each attack type
    edge_sensor_indices = {}
    for edge_id, (attack_type, sensor_names) in enumerate(attack_mapping.items()):
        indices = []
        for sensor in sensor_names:
            if sensor in feature_cols:
                indices.append(feature_cols.index(sensor))
        edge_sensor_indices[edge_id] = indices
    
    # Separate normal/attack data
    normal_mask = (labels == 0)
    attack_mask = (labels == 1)
    
    normal_features = features[normal_mask]
    attack_features = features[attack_mask]
    
    edge_data = {}
    
    for edge_id in range(3):
        # Distribute normal data equally across all edges
        normal_per_edge = len(normal_features) // 3
        start_idx = edge_id * normal_per_edge
        end_idx = (edge_id + 1) * normal_per_edge if edge_id < 2 else len(normal_features)
        
        edge_normal_features = normal_features[start_idx:end_idx]
        edge_normal_labels = np.zeros(len(edge_normal_features))
        
        # Filter attack data by sensor variance for this edge
        sensor_indices = edge_sensor_indices[edge_id]
        if len(sensor_indices) > 0:
            # Select attack samples with high variance in relevant sensors
            sensor_variance = np.var(attack_features[:, sensor_indices], axis=1)
            top_attack_indices = np.argsort(sensor_variance)[-len(attack_features)//3:]
            edge_attack_features = attack_features[top_attack_indices]
            edge_attack_labels = np.ones(len(edge_attack_features))
        else:
            # Equal distribution if no sensor mapping
            attack_per_edge = len(attack_features) // 3
            start_idx = edge_id * attack_per_edge
            end_idx = (edge_id + 1) * attack_per_edge if edge_id < 2 else len(attack_features)
            edge_attack_features = attack_features[start_idx:end_idx]
            edge_attack_labels = np.ones(len(edge_attack_features))
        
        # Combine edge data
        edge_features = np.vstack([edge_normal_features, edge_attack_features])
        edge_labels = np.concatenate([edge_normal_labels, edge_attack_labels])
        
        # Shuffle
        indices = np.random.permutation(len(edge_features))
        edge_features = edge_features[indices]
        edge_labels = edge_labels[indices]
        
        edge_data[edge_id] = {
            'features': edge_features,
            'labels': edge_labels,
            'sensor_indices': sensor_indices,
            'attack_type': list(attack_mapping.keys())[edge_id]
        }
    
    return edge_data

def preprocess_for_models(features: np.ndarray, model_type: str = 'duogat') -> np.ndarray:
    """Model-specific preprocessing"""
    
    # Standardization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    if model_type == 'duogat':
        # DuoGAT uses difference features as well
        diff_features = np.diff(features_scaled, axis=0, prepend=features_scaled[0:1])
        return features_scaled, diff_features
    elif model_type == 'anomaly_transformer':
        # Anomaly Transformer uses normalized features only
        return features_scaled
    else:
        return features_scaled

def create_federated_dataloaders(edge_data: Dict[int, Dict], model_type: str = 'duogat', 
                               window_size: int = 100, batch_size: int = 32, 
                               test_split: float = 0.2) -> Dict[int, Dict]:
    """Create federated learning DataLoaders"""
    
    federated_loaders = {}
    
    for edge_id, data in edge_data.items():
        features = data['features']
        labels = data['labels']
        
        # Model-specific preprocessing
        if model_type == 'duogat':
            features_processed, diff_features = preprocess_for_models(features, model_type)
            # Combine original and difference for DuoGAT
            features_combined = np.concatenate([features_processed, diff_features], axis=1)
        else:
            features_combined = preprocess_for_models(features, model_type)
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            features_combined, labels, test_size=test_split, 
            stratify=labels, random_state=42
        )
        
        # Create datasets
        train_dataset = WADIDataset(X_train, y_train, window_size)
        test_dataset = WADIDataset(X_test, y_test, window_size)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        federated_loaders[edge_id] = {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'feature_dim': features_combined.shape[1],
            'attack_type': data['attack_type'],
            'sensor_indices': data['sensor_indices'],
            'data_size': len(X_train)
        }
    
    return federated_loaders

def prepare_federated_data(csv_path: str = 'WADI_FINAL_DATASET.csv', 
                         model_type: str = 'duogat',
                         window_size: int = 100, 
                         batch_size: int = 32) -> Dict[int, Dict]:
    """Complete federated data preparation pipeline"""
    
    print("Loading WADI dataset...")
    features, labels, feature_cols = load_wadi_dataset(csv_path)
    
    print(f"Dataset shape: {features.shape}")
    print(f"Normal samples: {np.sum(labels == 0)}, Attack samples: {np.sum(labels == 1)}")
    
    print("Distributing data across edges...")
    edge_data = distribute_data_by_attack_type(features, labels, feature_cols)
    
    print("Creating federated dataloaders...")
    federated_loaders = create_federated_dataloaders(
        edge_data, model_type, window_size, batch_size
    )
    
    # Print data distribution
    for edge_id, loader_info in federated_loaders.items():
        print(f"Edge {edge_id} ({loader_info['attack_type']}): "
              f"{loader_info['data_size']} samples, "
              f"feature_dim: {loader_info['feature_dim']}")
    
    return federated_loaders

if __name__ == "__main__":
    # Test execution
    federated_data = prepare_federated_data(
        csv_path='WADI_FINAL_DATASET.csv',
        model_type='duogat',
        window_size=100,
        batch_size=32
    )
    
    print("\nFederated data preparation completed!")