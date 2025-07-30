import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import OrderedDict
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedAggregator:
    """Base class for federated learning aggregation algorithms"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.round_num = 0
        
    def aggregate(self, client_weights: List[Dict], client_samples: List[int]) -> Dict:
        """Aggregate client weights"""
        raise NotImplementedError
    
    def get_algorithm_name(self) -> str:
        """Get algorithm name"""
        raise NotImplementedError

class FedAvg(FederatedAggregator):
    """FedAvg: Federated Averaging Algorithm"""
    
    def __init__(self, device: torch.device):
        super().__init__(device)
        
    def aggregate(self, client_weights: List[Dict], client_samples: List[int]) -> Dict:
        """
        Aggregate client weights using weighted averaging
        
        Args:
            client_weights: List of client model weights
            client_samples: List of number of samples per client
            
        Returns:
            Aggregated global weights
        """
        if not client_weights:
            raise ValueError("No client weights provided")
            
        total_samples = sum(client_samples)
        
        # Initialize global weights with zeros
        global_weights = OrderedDict()
        for key in client_weights[0].keys():
            global_weights[key] = torch.zeros_like(client_weights[0][key]).to(self.device)
        
        # Weighted average
        for client_weight, num_samples in zip(client_weights, client_samples):
            weight_factor = num_samples / total_samples
            for key in global_weights.keys():
                global_weights[key] += weight_factor * client_weight[key].to(self.device)
        
        self.round_num += 1
        logger.info(f"FedAvg round {self.round_num}: Aggregated {len(client_weights)} clients")
        
        return global_weights
    
    def get_algorithm_name(self) -> str:
        return "FedAvg"

class FedProx(FederatedAggregator):
    """FedProx: Federated Optimization with Proximal Term"""
    
    def __init__(self, device: torch.device, mu: float = 0.01):
        super().__init__(device)
        self.mu = mu  # Proximal parameter
        self.global_weights_prev = None
        
    def aggregate(self, client_weights: List[Dict], client_samples: List[int]) -> Dict:
        """
        Aggregate client weights using FedProx algorithm
        
        Args:
            client_weights: List of client model weights
            client_samples: List of number of samples per client
            
        Returns:
            Aggregated global weights
        """
        if not client_weights:
            raise ValueError("No client weights provided")
            
        total_samples = sum(client_samples)
        
        # Initialize global weights with zeros
        global_weights = OrderedDict()
        for key in client_weights[0].keys():
            global_weights[key] = torch.zeros_like(client_weights[0][key]).to(self.device)
        
        # Weighted average (same as FedAvg)
        for client_weight, num_samples in zip(client_weights, client_samples):
            weight_factor = num_samples / total_samples
            for key in global_weights.keys():
                global_weights[key] += weight_factor * client_weight[key].to(self.device)
        
        # Store previous global weights for next round's proximal term
        self.global_weights_prev = copy.deepcopy(global_weights)
        
        self.round_num += 1
        logger.info(f"FedProx round {self.round_num}: Aggregated {len(client_weights)} clients with μ={self.mu}")
        
        return global_weights
        
    def get_proximal_loss(self, model_weights: Dict, global_weights: Dict) -> torch.Tensor:
        """Calculate proximal loss term for client training"""
        if global_weights is None:
            return torch.tensor(0.0).to(self.device)
            
        proximal_loss = 0.0
        for key in model_weights.keys():
            if key in global_weights:
                proximal_loss += torch.norm(model_weights[key] - global_weights[key]) ** 2
        
        return self.mu / 2 * proximal_loss
    
    def get_algorithm_name(self) -> str:
        return f"FedProx(μ={self.mu})"

class FedAdam(FederatedAggregator):
    """FedAdam: Adaptive Federated Optimization"""
    
    def __init__(self, device: torch.device, 
                 eta: float = 1e-3,  # Server learning rate
                 beta1: float = 0.9, 
                 beta2: float = 0.999, 
                 epsilon: float = 1e-8):
        super().__init__(device)
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Server-side momentum buffers
        self.m_t = None  # First moment
        self.v_t = None  # Second moment
        self.global_weights_prev = None
        
    def aggregate(self, client_weights: List[Dict], client_samples: List[int]) -> Dict:
        """
        Aggregate client weights using FedAdam algorithm
        
        Args:
            client_weights: List of client model weights  
            client_samples: List of number of samples per client
            
        Returns:
            Aggregated global weights
        """
        if not client_weights:
            raise ValueError("No client weights provided")
            
        total_samples = sum(client_samples)
        
        # Calculate weighted average (pseudo-gradient)
        delta_weights = OrderedDict()
        for key in client_weights[0].keys():
            delta_weights[key] = torch.zeros_like(client_weights[0][key]).to(self.device)
            
        for client_weight, num_samples in zip(client_weights, client_samples):
            weight_factor = num_samples / total_samples
            for key in delta_weights.keys():
                delta_weights[key] += weight_factor * client_weight[key].to(self.device)
        
        # Initialize momentum buffers if first round
        if self.m_t is None:
            self.m_t = OrderedDict()
            self.v_t = OrderedDict()
            self.global_weights_prev = OrderedDict()
            
            for key in delta_weights.keys():
                self.m_t[key] = torch.zeros_like(delta_weights[key]).to(self.device)
                self.v_t[key] = torch.zeros_like(delta_weights[key]).to(self.device)
                self.global_weights_prev[key] = torch.zeros_like(delta_weights[key]).to(self.device)
        
        # Calculate pseudo-gradient (difference from previous global weights)
        pseudo_gradient = OrderedDict()
        for key in delta_weights.keys():
            pseudo_gradient[key] = delta_weights[key] - self.global_weights_prev[key]
        
        # Update momentum buffers
        for key in pseudo_gradient.keys():
            self.m_t[key] = self.beta1 * self.m_t[key] + (1 - self.beta1) * pseudo_gradient[key]
            self.v_t[key] = self.beta2 * self.v_t[key] + (1 - self.beta2) * (pseudo_gradient[key] ** 2)
        
        # Bias correction
        self.round_num += 1
        m_hat = OrderedDict()
        v_hat = OrderedDict()
        
        for key in self.m_t.keys():
            m_hat[key] = self.m_t[key] / (1 - self.beta1 ** self.round_num)
            v_hat[key] = self.v_t[key] / (1 - self.beta2 ** self.round_num)
        
        # Update global weights using Adam-like update
        global_weights = OrderedDict()
        for key in delta_weights.keys():
            global_weights[key] = (self.global_weights_prev[key] + 
                                 self.eta * m_hat[key] / (torch.sqrt(v_hat[key]) + self.epsilon))
        
        # Store current weights for next round
        self.global_weights_prev = copy.deepcopy(global_weights)
        
        logger.info(f"FedAdam round {self.round_num}: Aggregated {len(client_weights)} clients with η={self.eta}")
        
        return global_weights
    
    def get_algorithm_name(self) -> str:
        return f"FedAdam(η={self.eta})"

def create_aggregator(algorithm: str, device: torch.device, **kwargs) -> FederatedAggregator:
    """
    Create federated aggregator based on algorithm name
    
    Args:
        algorithm: Algorithm name ('fedavg', 'fedprox', 'fedadam')
        device: Device to run on
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Initialized aggregator
    """
    algorithm = algorithm.lower()
    
    if algorithm == 'fedavg':
        return FedAvg(device)
    elif algorithm == 'fedprox':
        mu = kwargs.get('mu', 0.01)
        return FedProx(device, mu=mu)
    elif algorithm == 'fedadam':
        eta = kwargs.get('eta', 1e-3)
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-8)
        return FedAdam(device, eta=eta, beta1=beta1, beta2=beta2, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def calculate_model_size(model_weights: Dict) -> int:
    """Calculate model size in bytes"""
    total_bytes = 0
    for weight in model_weights.values():
        total_bytes += weight.numel() * weight.element_size()
    return total_bytes

def get_parameter_count(model_weights: Dict) -> Dict[str, int]:
    """Get parameter count statistics"""
    total_params = 0
    trainable_params = 0
    
    for name, param in model_weights.items():
        param_count = param.numel()
        total_params += param_count
        # Assume all parameters in the weights dict are trainable
        trainable_params += param_count
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }

class CommunicationTracker:
    """Track communication overhead in federated learning"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.round_communication = []
        
    def log_upload(self, model_weights: Dict):
        """Log client to server communication"""
        bytes_sent = calculate_model_size(model_weights)
        self.total_bytes_sent += bytes_sent
        return bytes_sent
        
    def log_download(self, model_weights: Dict):
        """Log server to client communication"""
        bytes_received = calculate_model_size(model_weights)
        self.total_bytes_received += bytes_received
        return bytes_received
        
    def log_round_end(self, round_num: int):
        """Log end of communication round"""
        round_total = self.total_bytes_sent + self.total_bytes_received
        self.round_communication.append({
            'round': round_num,
            'bytes_sent': self.total_bytes_sent,
            'bytes_received': self.total_bytes_received,
            'total_bytes': round_total
        })
        
    def get_communication_stats(self) -> Dict:
        """Get communication statistics"""
        return {
            'total_bytes_sent': self.total_bytes_sent,
            'total_bytes_received': self.total_bytes_received,
            'total_communication': self.total_bytes_sent + self.total_bytes_received,
            'total_mb': (self.total_bytes_sent + self.total_bytes_received) / (1024 * 1024),
            'round_history': self.round_communication
        }

if __name__ == "__main__":
    # Test aggregators
    device = torch.device('cpu')
    
    # Create dummy client weights
    client_weights = []
    client_samples = [100, 150, 120]
    
    for i in range(3):
        weights = OrderedDict()
        weights['layer1.weight'] = torch.randn(10, 5)
        weights['layer1.bias'] = torch.randn(10)
        weights['layer2.weight'] = torch.randn(2, 10)
        weights['layer2.bias'] = torch.randn(2)
        client_weights.append(weights)
    
    # Test all aggregators
    algorithms = ['fedavg', 'fedprox', 'fedadam']
    
    for alg in algorithms:
        print(f"\nTesting {alg.upper()}:")
        aggregator = create_aggregator(alg, device)
        
        for round_num in range(3):
            global_weights = aggregator.aggregate(client_weights, client_samples)
            print(f"  Round {round_num + 1}: {aggregator.get_algorithm_name()}")
            
        # Test communication tracking
        comm_tracker = CommunicationTracker()
        for weights in client_weights:
            comm_tracker.log_upload(weights)
            comm_tracker.log_download(global_weights)
        
        stats = comm_tracker.get_communication_stats()
        print(f"  Communication: {stats['total_mb']:.2f} MB")
        print(f"  Parameters: {get_parameter_count(global_weights)}")