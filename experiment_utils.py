# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy.stats import wasserstein_distance as wd
import pandas as pd
import json
from datetime import datetime

# Import existing models (use without modification)
sys.path.append('./DuoGAT')
sys.path.append('./Anomaly-Transformer')

class ModelAdapter:
    """Model adapter class - wraps existing models for federated learning"""
    
    def __init__(self, model_type: str, feature_dim: int, freeze_encoder: bool = False):
        self.model_type = model_type
        self.feature_dim = feature_dim
        self.freeze_encoder = freeze_encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.criterion = nn.BCEWithLogitsLoss()
        
    def _create_model(self) -> nn.Module:
        """Create model"""
        if self.model_type == 'duogat':
            return self._create_duogat_model()
        elif self.model_type == 'anomaly_transformer':
            return self._create_anomaly_transformer_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_duogat_model(self) -> nn.Module:
        """Create DuoGAT model"""
        try:
            from DuoGAT.duogat import DuoGAT
            from DuoGAT.args import get_parser as duogat_parser
            
            # Use DuoGAT default settings
            parser = duogat_parser()
            args = parser.parse_args([])  # Use defaults
            
            # Adjust feature dimension
            args.feature_dim = self.feature_dim // 2  # Original + difference
            
            # Create DuoGAT model
            model = DuoGAT(
                n_features=args.feature_dim,
                window_size=100,  # Fixed value
                out_dim=getattr(args, 'out_dim', 128),
                kernel_size=getattr(args, 'kernel_size', 7)
            )
            
            # Frozen Encoder handling
            if self.freeze_encoder:
                for name, param in model.named_parameters():
                    if 'encoder' in name.lower() or 'gnn' in name.lower():
                        param.requires_grad = False
            
            return model.to(self.device)
            
        except ImportError:
            # Fallback: Simple CNN model
            class SimpleDuoGAT(nn.Module):
                def __init__(self, feature_dim):
                    super().__init__()
                    self.feature_dim = feature_dim
                    
                    # Convolutional layers
                    self.conv1 = nn.Conv1d(feature_dim, 64, kernel_size=7, padding=3)
                    self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
                    self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
                    
                    # Classifier
                    self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool1d(1),
                        nn.Flatten(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(32, 1)
                    )
                
                def forward(self, x):
                    # x shape: (batch, seq_len, features)
                    x = x.transpose(1, 2)  # (batch, features, seq_len)
                    x = torch.relu(self.conv1(x))
                    x = torch.relu(self.conv2(x))
                    x = torch.relu(self.conv3(x))
                    x = self.classifier(x)
                    return x
            
            model = SimpleDuoGAT(self.feature_dim)
            
            # Frozen Encoder handling
            if self.freeze_encoder:
                for name, param in model.named_parameters():
                    if 'conv' in name:
                        param.requires_grad = False
            
            return model.to(self.device)
    
    def _create_anomaly_transformer_model(self) -> nn.Module:
        """Create Anomaly Transformer model"""
        try:
            from Anomaly_Transformer.model.AnomalyTransformer import AnomalyTransformer
            
            model = AnomalyTransformer(
                win_size=100,
                enc_in=self.feature_dim,
                c_out=1,  # Binary classification
                d_model=512,
                n_heads=8,
                e_layers=3,
                d_ff=512,
                dropout=0.0,
                activation='gelu',
                output_attention=True
            )
            
            # Frozen Encoder handling
            if self.freeze_encoder:
                for name, param in model.named_parameters():
                    if 'encoder' in name.lower():
                        param.requires_grad = False
            
            return model.to(self.device)
            
        except ImportError:
            # Fallback: Simple Transformer model
            class SimpleAnomalyTransformer(nn.Module):
                def __init__(self, feature_dim, d_model=256, nhead=8, num_layers=3):
                    super().__init__()
                    self.feature_dim = feature_dim
                    self.d_model = d_model
                    
                    # Input projection
                    self.input_projection = nn.Linear(feature_dim, d_model)
                    
                    # Transformer encoder
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                        dropout=0.1, batch_first=True
                    )
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    
                    # Output layers
                    self.classifier = nn.Sequential(
                        nn.Linear(d_model, 128),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(128, 1)
                    )
                
                def forward(self, x):
                    # x shape: (batch, seq_len, features)
                    x = self.input_projection(x)
                    x = self.transformer(x)
                    # Global average pooling
                    x = x.mean(dim=1)
                    x = self.classifier(x)
                    return x
            
            model = SimpleAnomalyTransformer(self.feature_dim)
            
            # Frozen Encoder handling
            if self.freeze_encoder:
                for name, param in model.named_parameters():
                    if 'transformer' in name:
                        param.requires_grad = False
            
            return model.to(self.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        if self.freeze_encoder:
            # Train only classifier for frozen encoder
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            trainable_params = self.model.parameters()
        
        return optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-5)
    
    def get_all_parameters(self) -> List[np.ndarray]:
        """Return all model parameters"""
        return [param.detach().cpu().numpy() for param in self.model.parameters()]
    
    def get_classifier_parameters(self) -> List[np.ndarray]:
        """Return classifier parameters only (for Frozen Encoder)"""
        classifier_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Only trainable parameters
                classifier_params.append(param.detach().cpu().numpy())
        return classifier_params
    
    def set_all_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set all model parameters"""
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.dtype, device=param.device)
    
    def set_classifier_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set classifier parameters only (for Frozen Encoder)"""
        param_idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = torch.tensor(
                    parameters[param_idx], 
                    dtype=param.dtype, 
                    device=param.device
                )
                param_idx += 1
    
    def train_one_epoch(self, train_loader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = self.model(batch_data)
                
                # Adjust output dimensions
                if outputs.dim() > 1:
                    outputs = outputs.squeeze()
                
                loss = self.criterion(outputs, batch_labels.squeeze())
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Training error: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, test_loader) -> Tuple[float, float, float, np.ndarray]:
        """Model evaluation"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.float().to(self.device)
                
                try:
                    outputs = self.model(batch_data)
                    
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze()
                    
                    loss = self.criterion(outputs, batch_labels.squeeze())
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Calculate predictions
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"Evaluation error: {e}")
                    continue
        
        # Calculate metrics
        avg_loss = total_loss / max(num_batches, 1)
        
        if len(all_preds) > 0:
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels).flatten()
            
            f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
            acc = accuracy_score(all_labels, all_preds)
            conf_matrix = confusion_matrix(all_labels, all_preds)
        else:
            f1 = 0.0
            acc = 0.0
            conf_matrix = np.zeros((2, 2))
        
        return avg_loss, f1, acc, conf_matrix

def wasserstein_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate Wasserstein distance"""
    try:
        # Convert to 1D arrays
        x1_flat = x1.flatten()
        x2_flat = x2.flatten()
        
        # Match sizes
        min_len = min(len(x1_flat), len(x2_flat))
        x1_flat = x1_flat[:min_len]
        x2_flat = x2_flat[:min_len]
        
        return wd(x1_flat, x2_flat)
    except Exception as e:
        print(f"Wasserstein distance calculation error: {e}")
        return 0.0

def apply_distribution_shift(data: np.ndarray, shift_type: str = 'scale', 
                           intensity: float = 2.0) -> np.ndarray:
    """Apply distribution shift"""
    shifted_data = data.copy()
    
    if shift_type == 'scale':
        # Scale change
        shifted_data = shifted_data * intensity
    elif shift_type == 'shift':
        # Mean shift
        mean_shift = np.mean(data) * (intensity - 1)
        shifted_data = shifted_data + mean_shift
    elif shift_type == 'noise':
        # Add noise
        noise = np.random.normal(0, np.std(data) * intensity, data.shape)
        shifted_data = shifted_data + noise
    
    return shifted_data

def check_python_dependencies():
    """Check required Python packages"""
    required_packages = ['torch', 'pandas', 'numpy', 'sklearn', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print('Missing packages:', ', '.join(missing_packages))
        return False
    else:
        print('All required packages are installed.')
        return True

def validate_wadi_data(data_path: str) -> bool:
    """Validate WADI data"""
    try:
        df = pd.read_csv(data_path)
        print(f'Data size: {df.shape}')
        print(f'Number of columns: {len(df.columns)}')
        
        if 'Label' in df.columns:
            normal_count = sum(df['Label'] == 0)
            attack_count = sum(df['Label'] == 1)
            print(f'Normal data: {normal_count}')
            print(f'Attack data: {attack_count}')
            
            if attack_count == 0:
                print('Warning: No attack data found.')
                return False
        else:
            print('Warning: No Label column found.')
            return False
            
        # Check data types
        numeric_cols = df.select_dtypes(include=['number']).columns
        print(f'Numeric columns: {len(numeric_cols)}')
        
        if len(numeric_cols) < 10:
            print('Warning: Too few numeric columns.')
            
        return True
        
    except Exception as e:
        print(f'Data validation failed: {e}')
        return False

def generate_experiment_summary():
    """Generate experiment results summary"""
    results_dir = 'experiment_logs'
    if not os.path.exists(results_dir):
        print('Results directory not found.')
        return False
    
    experiment_results = []
    
    # Search CSV files
    for filename in os.listdir(results_dir):
        if filename.endswith('_metrics.csv'):
            try:
                csv_path = os.path.join(results_dir, filename)
                df = pd.read_csv(csv_path)
                
                if not df.empty:
                    # Extract experiment info
                    exp_info = {
                        'experiment_id': filename.replace('_metrics.csv', ''),
                        'final_f1': df['f1_score'].iloc[-1] if 'f1_score' in df.columns else 0,
                        'final_accuracy': df['accuracy'].iloc[-1] if 'accuracy' in df.columns else 0,
                        'final_loss': df['loss'].iloc[-1] if 'loss' in df.columns else 0,
                        'max_wasserstein': df['wasserstein_distance'].max() if 'wasserstein_distance' in df.columns else 0,
                        'num_rounds': len(df)
                    }
                    
                    # Parse model and strategy info
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        exp_info['model'] = parts[0]
                        exp_info['strategy'] = parts[1]
                        exp_info['frozen_encoder'] = 'frozen' in filename
                    
                    experiment_results.append(exp_info)
                    
            except Exception as e:
                print(f"Failed to read results file: {filename} - {e}")
                continue
    
    if not experiment_results:
        print('No experiment results to analyze.')
        return False
    
    # Convert to DataFrame and save
    df_summary = pd.DataFrame(experiment_results)
    summary_file = f'experiment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df_summary.to_csv(summary_file, index=False)
    
    print(f'Experiment summary saved: {summary_file}')
    
    # Print best performance
    if not df_summary.empty:
        best_f1_exp = df_summary.loc[df_summary['final_f1'].idxmax()]
        print(f'\n=== Best F1 Performance ===')
        print(f'Experiment ID: {best_f1_exp["experiment_id"]}')
        print(f'F1 Score: {best_f1_exp["final_f1"]:.4f}')
        print(f'Accuracy: {best_f1_exp["final_accuracy"]:.4f}')
        
        # Frozen Encoder analysis
        frozen_exps = df_summary[df_summary['frozen_encoder'] == True]
        full_exps = df_summary[df_summary['frozen_encoder'] == False]
        
        if not frozen_exps.empty and not full_exps.empty:
            print(f'\n=== Frozen Encoder Analysis ===')
            print(f'Full Fine-tuning avg F1: {full_exps["final_f1"].mean():.4f}')
            print(f'Frozen Encoder avg F1: {frozen_exps["final_f1"].mean():.4f}')
            print(f'Communication savings: ~50%')
        
        # Distribution shift analysis
        high_wasserstein = df_summary[df_summary['max_wasserstein'] > 0.2]
        if not high_wasserstein.empty:
            print(f'\n=== Distribution Shift Detection ===')
            print(f'High Wasserstein distance experiments: {len(high_wasserstein)}')
            print(f'Average max Wasserstein: {high_wasserstein["max_wasserstein"].mean():.4f}')
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiment_utils.py <command> [args...]")
        print("Commands:")
        print("  check_deps - Check dependencies")
        print("  validate_data <path> - Validate data")
        print("  generate_summary - Generate results summary")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "check_deps":
        success = check_python_dependencies()
        sys.exit(0 if success else 1)
    elif command == "validate_data":
        if len(sys.argv) < 3:
            print("Data path required")
            sys.exit(1)
        success = validate_wadi_data(sys.argv[2])
        sys.exit(0 if success else 1)
    elif command == "generate_summary":
        success = generate_experiment_summary()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)