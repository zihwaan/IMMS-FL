import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from anomaly_transformer import AnomalyTransformer, AnomalyLoss
from fl_algorithms import CommunicationTracker
from data_loader import calculate_wasserstein_distance, simulate_distribution_shift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetectionClient(fl.client.NumPyClient):
    """Flower client for anomaly detection with Anomaly Transformer"""
    
    def __init__(self, 
                 client_id: int,
                 model: AnomalyTransformer,
                 trainloader: DataLoader,
                 valloader: DataLoader,
                 device: torch.device,
                 frozen_encoder: bool = False,
                 algorithm: str = 'fedavg',
                 mu: float = 0.01):  # For FedProx
        
        self.client_id = client_id
        self.model = model.to(device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.frozen_encoder = frozen_encoder
        self.algorithm = algorithm.lower()
        self.mu = mu
        
        # Training components
        self.criterion = AnomalyLoss()
        self.optimizer = self._create_optimizer()
        
        # Communication tracking
        self.comm_tracker = CommunicationTracker()
        
        # Distribution tracking for shift detection
        self.baseline_distribution = None
        self.current_distribution = None
        
        logger.info(f"Client {client_id} initialized with {algorithm} algorithm")
        
    def _create_optimizer(self):
        """Create optimizer based on frozen encoder setting"""
        if self.frozen_encoder:
            # Only optimize detection head parameters
            params = self.model.get_detection_head_params()
            logger.info(f"Client {self.client_id}: Training detection head only ({sum(p.numel() for p in params)} params)")
        else:
            # Optimize all parameters
            params = self.model.parameters()
            logger.info(f"Client {self.client_id}: Training full model ({sum(p.numel() for p in params)} params)")
            
        return torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters as numpy arrays"""
        if self.frozen_encoder:
            # Only return detection head parameters
            params = [val.cpu().numpy() for val in self.model.detection_head.state_dict().values()]
        else:
            # Return all parameters
            params = [val.cpu().numpy() for val in self.model.state_dict().values()]
        
        # Log communication
        model_dict = self.model.detection_head.state_dict() if self.frozen_encoder else self.model.state_dict()
        self.comm_tracker.log_upload(model_dict)
        
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays"""
        if self.frozen_encoder:
            # Only set detection head parameters
            params_dict = zip(self.model.detection_head.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.detection_head.load_state_dict(state_dict, strict=True)
        else:
            # Set all parameters
            params_dict = zip(self.model.state_dict().keys(), parameters)  
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        
        # Log communication
        model_dict = self.model.detection_head.state_dict() if self.frozen_encoder else self.model.state_dict()
        self.comm_tracker.log_download(model_dict)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on local data"""
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Get training configuration
        epochs = config.get('epochs', 1)
        round_num = config.get('round', 0)
        
        # Apply distribution shift if specified
        if config.get('apply_shift', False) and self.client_id == 2:
            self._apply_distribution_shift(config)
        
        # Train model
        train_loss, train_metrics = self._train(epochs, round_num)
        
        # Get updated parameters  
        updated_params = self.get_parameters({})
        
        # Return results
        num_examples = len(self.trainloader.dataset)
        metrics = {
            'train_loss': train_loss,
            'train_f1': train_metrics.get('f1', 0.0),
            'train_accuracy': train_metrics.get('accuracy', 0.0),
            'client_id': self.client_id,
            'round': round_num
        }
        
        return updated_params, num_examples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model on local validation data"""
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Evaluate model
        val_loss, val_metrics = self._evaluate()
        
        num_examples = len(self.valloader.dataset)
        metrics = {
            'val_f1': val_metrics.get('f1', 0.0),
            'val_accuracy': val_metrics.get('accuracy', 0.0),
            'client_id': self.client_id
        }
        
        return val_loss, num_examples, metrics
    
    def _train(self, epochs: int, round_num: int) -> Tuple[float, Dict]:
        """Training loop"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in self.trainloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y, batch_x)
                
                # Add proximal term for FedProx
                if self.algorithm == 'fedprox' and hasattr(self, 'global_weights'):
                    proximal_loss = self._calculate_proximal_loss()
                    loss += proximal_loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for metrics
                predictions = torch.argmax(outputs['logits'], dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.squeeze().cpu().numpy())
            
            total_loss += epoch_loss / num_batches
        
        avg_loss = total_loss / epochs
        
        # Calculate metrics
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        accuracy = accuracy_score(all_targets, all_predictions)
        
        metrics = {
            'f1': f1,
            'accuracy': accuracy,
            'loss': avg_loss
        }
        
        logger.info(f"Client {self.client_id} Round {round_num}: "
                   f"Loss={avg_loss:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")
        
        return avg_loss, metrics
    
    def _evaluate(self) -> Tuple[float, Dict]:
        """Evaluation loop"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.valloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y, batch_x)
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs['logits'], dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.squeeze().cpu().numpy())
        
        avg_loss = total_loss / len(self.valloader)
        
        # Calculate metrics
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        accuracy = accuracy_score(all_targets, all_predictions)
        
        metrics = {
            'f1': f1,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix(all_targets, all_predictions).tolist()
        }
        
        return avg_loss, metrics
    
    def _calculate_proximal_loss(self) -> torch.Tensor:
        """Calculate proximal loss for FedProx"""
        if not hasattr(self, 'global_weights'):
            return torch.tensor(0.0).to(self.device)
        
        proximal_loss = 0.0
        
        if self.frozen_encoder:
            current_weights = self.model.detection_head.state_dict()
            global_weights = self.global_weights
        else:
            current_weights = self.model.state_dict()
            global_weights = self.global_weights
        
        for key in current_weights.keys():
            if key in global_weights:
                proximal_loss += torch.norm(current_weights[key] - global_weights[key]) ** 2
        
        return self.mu / 2 * proximal_loss
    
    def _apply_distribution_shift(self, config: Dict):
        """Apply distribution shift to training data"""
        shift_intensity = config.get('shift_intensity', 2.0)
        
        logger.info(f"Client {self.client_id}: Applying distribution shift with intensity {shift_intensity}")
        
        # This is a simplified simulation - in practice, you'd modify the actual data
        # Here we just log the shift for demonstration
        if self.baseline_distribution is None:
            # Store baseline distribution
            sample_batch = next(iter(self.trainloader))[0].numpy()
            self.baseline_distribution = sample_batch.flatten()
        
        # Simulate shifted distribution
        shifted_data = simulate_distribution_shift(
            self.baseline_distribution.reshape(-1, 1), 
            'scale', 
            shift_intensity
        )
        
        # Calculate Wasserstein distance
        wd = calculate_wasserstein_distance(self.baseline_distribution, shifted_data.flatten())
        
        if wd > 0.2:  # Threshold from requirements
            logger.warning(f"Client {self.client_id}: Distribution shift detected! "
                          f"Wasserstein Distance = {wd:.4f} > 0.2")
            # In practice, you would trigger rebalancing here
        
        self.current_distribution = shifted_data.flatten()
    
    def get_communication_stats(self) -> Dict:
        """Get communication statistics"""
        return self.comm_tracker.get_communication_stats()

def create_client(client_id: int,
                 model: AnomalyTransformer,
                 trainloader: DataLoader,
                 valloader: DataLoader,
                 device: torch.device,
                 config: Dict) -> AnomalyDetectionClient:
    """Create a federated learning client"""
    
    return AnomalyDetectionClient(
        client_id=client_id,
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        device=device,
        frozen_encoder=config.get('frozen_encoder', False),
        algorithm=config.get('algorithm', 'fedavg'),
        mu=config.get('mu', 0.01)
    )

if __name__ == "__main__":
    # Test client creation (this would be used in actual training)
    from anomaly_transformer import create_model
    from data_loader import WADIDataLoader, WADIDataset
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_loader = WADIDataLoader()  
    X, y = data_loader.load_and_preprocess()
    data_splits = data_loader.create_federated_split(X, y)
    
    # Create model
    input_dim = X.shape[1]
    model = create_model(input_dim)
    
    # Create dummy dataloaders for testing
    edge_data = data_splits['edges']['edge_0']
    dataset = WADIDataset(edge_data['X'], edge_data['y'])
    trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
    valloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Create client
    config = {'frozen_encoder': False, 'algorithm': 'fedavg'}
    client = create_client(0, model, trainloader, valloader, device, config)
    
    print(f"Client created successfully with {len(trainloader.dataset)} training samples")