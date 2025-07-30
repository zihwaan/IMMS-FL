import flwr as fl
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import logging
import csv
import os
from datetime import datetime

from fl_algorithms import create_aggregator, FederatedAggregator
from anomaly_transformer import AnomalyTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetectionStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for anomaly detection experiments"""
    
    def __init__(self,
                 model: AnomalyTransformer,
                 algorithm: str = 'fedavg',
                 min_fit_clients: int = 3,
                 min_evaluate_clients: int = 3,
                 min_available_clients: int = 3,
                 device: torch.device = torch.device('cpu'),
                 frozen_encoder: bool = False,
                 **kwargs):
        
        super().__init__()
        
        self.model = model.to(device)
        self.algorithm = algorithm.lower()
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.device = device
        self.frozen_encoder = frozen_encoder
        
        # Create aggregator
        self.aggregator = create_aggregator(algorithm, device, **kwargs)
        
        # Metrics tracking
        self.round_metrics = []
        self.results_dir = kwargs.get('results_dir', 'results')
        self.experiment_name = kwargs.get('experiment_name', 'experiment')
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize CSV file for logging
        self.csv_file = os.path.join(self.results_dir, f'{self.experiment_name}_history.csv')
        self._initialize_csv()
        
        logger.info(f"Strategy initialized: {algorithm} with {min_fit_clients} clients")
        
    def _initialize_csv(self):
        """Initialize CSV file for logging metrics"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'algorithm', 'frozen_encoder',
                'train_loss_avg', 'train_f1_avg', 'train_accuracy_avg',
                'val_loss_avg', 'val_f1_avg', 'val_accuracy_avg',
                'num_clients', 'timestamp'
            ])
    
    def initialize_parameters(self, client_manager) -> Optional[fl.common.Parameters]:
        """Initialize global model parameters"""
        if self.frozen_encoder:
            # Only return detection head parameters
            params = [val.cpu().numpy() for val in self.model.detection_head.state_dict().values()]
        else:
            # Return all parameters
            params = [val.cpu().numpy() for val in self.model.state_dict().values()]
        
        return fl.common.ndarrays_to_parameters(params)
    
    def configure_fit(self, 
                     server_round: int, 
                     parameters: fl.common.Parameters,
                     client_manager) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training"""
        
        # Sample clients
        sample_size = max(self.min_fit_clients, int(len(client_manager.all()) * 1.0))
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_fit_clients)
        
        # Create fit instructions
        config = {
            'epochs': 1,
            'round': server_round,
            'algorithm': self.algorithm,
            'frozen_encoder': self.frozen_encoder
        }
        
        # Apply distribution shift at round 50 for client 2 (as per requirements)
        if server_round == 50:
            config['apply_shift'] = True
            config['shift_intensity'] = 2.0
            logger.info(f"Round {server_round}: Applying distribution shift to client 2")
        
        fit_ins = fl.common.FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self,
                     server_round: int,
                     results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                     failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]) -> Tuple[Optional[fl.common.Parameters], Dict]:
        """Aggregate training results"""
        
        if not results:
            return None, {}
        
        # Extract client weights and sample counts
        client_weights = []
        client_samples = []
        client_metrics = []
        
        for client, fit_res in results:
            # Convert parameters to tensors
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            
            if self.frozen_encoder:
                # Create state dict for detection head only
                param_dict = OrderedDict()
                param_names = list(self.model.detection_head.state_dict().keys())
                for name, param in zip(param_names, params):
                    param_dict[name] = torch.tensor(param)
            else:
                # Create state dict for full model
                param_dict = OrderedDict()
                param_names = list(self.model.state_dict().keys())
                for name, param in zip(param_names, params):
                    param_dict[name] = torch.tensor(param)
            
            client_weights.append(param_dict)
            client_samples.append(fit_res.num_examples)
            client_metrics.append(fit_res.metrics)
        
        # Aggregate weights using the specified algorithm
        try:
            aggregated_weights = self.aggregator.aggregate(client_weights, client_samples)
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return None, {}
        
        # Convert back to parameters
        aggregated_params = [val.cpu().numpy() for val in aggregated_weights.values()]
        parameters = fl.common.ndarrays_to_parameters(aggregated_params)
        
        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(client_metrics, client_samples)
        
        # Store round metrics
        round_data = {
            'round': server_round,
            'algorithm': self.algorithm,
            'frozen_encoder': self.frozen_encoder,
            'num_clients': len(results),
            **avg_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.round_metrics.append(round_data)
        
        # Log to CSV
        self._log_to_csv(round_data)
        
        logger.info(f"Round {server_round}: "
                   f"Loss={avg_metrics.get('train_loss_avg', 0):.4f}, "
                   f"F1={avg_metrics.get('train_f1_avg', 0):.4f}, "
                   f"Acc={avg_metrics.get('train_accuracy_avg', 0):.4f}")
        
        return parameters, avg_metrics
    
    def configure_evaluate(self,
                          server_round: int,
                          parameters: fl.common.Parameters,
                          client_manager) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        """Configure the next round of evaluation"""
        
        # Sample clients for evaluation
        sample_size = max(self.min_evaluate_clients, int(len(client_manager.all()) * 1.0))
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_evaluate_clients)
        
        config = {'round': server_round}
        evaluate_ins = fl.common.EvaluateIns(parameters, config)
        
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(self,
                          server_round: int,
                          results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
                          failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict]:
        """Aggregate evaluation results"""
        
        if not results:
            return None, {}
        
        # Extract metrics
        losses = [res.loss for _, res in results]
        sample_counts = [res.num_examples for _, res in results]
        client_metrics = [res.metrics for _, res in results]
        
        # Calculate weighted average loss
        total_samples = sum(sample_counts)
        avg_loss = sum(loss * samples for loss, samples in zip(losses, sample_counts)) / total_samples
        
        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(client_metrics, sample_counts)
        
        # Update round metrics with evaluation results
        if self.round_metrics and self.round_metrics[-1]['round'] == server_round:
            self.round_metrics[-1].update({
                'val_loss_avg': avg_loss,
                **{k: v for k, v in avg_metrics.items() if k.startswith('val_')}
            })
            
            # Update CSV
            self._log_to_csv(self.round_metrics[-1])
        
        logger.info(f"Round {server_round} Evaluation: "
                   f"Loss={avg_loss:.4f}, "
                   f"F1={avg_metrics.get('val_f1_avg', 0):.4f}, "
                   f"Acc={avg_metrics.get('val_accuracy_avg', 0):.4f}")
        
        return avg_loss, avg_metrics
    
    def _calculate_average_metrics(self, client_metrics: List[Dict], sample_counts: List[int]) -> Dict:
        """Calculate weighted average of client metrics"""
        if not client_metrics:
            return {}
        
        total_samples = sum(sample_counts)
        avg_metrics = {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in client_metrics:
            all_keys.update(metrics.keys())
        
        # Calculate weighted averages
        for key in all_keys:
            if key in ['client_id', 'round', 'timestamp', 'confusion_matrix']:
                continue
                
            values = []
            weights = []
            
            for metrics, samples in zip(client_metrics, sample_counts):
                if key in metrics and isinstance(metrics[key], (int, float)):
                    values.append(metrics[key])
                    weights.append(samples)
            
            if values:
                weighted_avg = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                avg_metrics[f"{key}_avg"] = weighted_avg
        
        return avg_metrics
    
    def _log_to_csv(self, round_data: Dict):
        """Log round data to CSV file"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_data.get('round', ''),
                round_data.get('algorithm', ''),
                round_data.get('frozen_encoder', ''),
                round_data.get('train_loss_avg', ''),
                round_data.get('train_f1_avg', ''),
                round_data.get('train_accuracy_avg', ''),
                round_data.get('val_loss_avg', ''),
                round_data.get('val_f1_avg', ''),
                round_data.get('val_accuracy_avg', ''),
                round_data.get('num_clients', ''),
                round_data.get('timestamp', '')
            ])
    
    def evaluate(self, server_round: int, parameters: fl.common.Parameters) -> Optional[Tuple[float, Dict]]:
        """Server-side evaluation (optional)"""
        return None
    
    def get_experiment_results(self) -> Dict:
        """Get complete experiment results"""
        return {
            'algorithm': self.algorithm,
            'frozen_encoder': self.frozen_encoder,
            'total_rounds': len(self.round_metrics),
            'round_metrics': self.round_metrics,
            'csv_file': self.csv_file,
            'final_metrics': self.round_metrics[-1] if self.round_metrics else {}
        }

def create_strategy(model: AnomalyTransformer,
                   algorithm: str,
                   config: Dict) -> AnomalyDetectionStrategy:
    """Create federated learning strategy"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return AnomalyDetectionStrategy(
        model=model,
        algorithm=algorithm,
        device=device,
        min_fit_clients=config.get('min_fit_clients', 3),
        min_evaluate_clients=config.get('min_evaluate_clients', 3),
        min_available_clients=config.get('min_available_clients', 3),
        frozen_encoder=config.get('frozen_encoder', False),
        results_dir=config.get('results_dir', 'results'),
        experiment_name=config.get('experiment_name', 'experiment'),
        # Algorithm-specific parameters
        mu=config.get('mu', 0.01),  # FedProx
        eta=config.get('eta', 1e-3),  # FedAdam
        beta1=config.get('beta1', 0.9),  # FedAdam
        beta2=config.get('beta2', 0.999),  # FedAdam
        epsilon=config.get('epsilon', 1e-8)  # FedAdam
    )

def start_server(strategy: AnomalyDetectionStrategy,
                config: Dict) -> Dict:
    """Start Flower server"""
    
    server_config = fl.server.ServerConfig(
        num_rounds=config.get('num_rounds', 100)
    )
    
    # Start server
    history = fl.server.start_server(
        server_address=config.get('server_address', '0.0.0.0:8080'),
        config=server_config,
        strategy=strategy
    )
    
    # Get final results
    results = strategy.get_experiment_results()
    results['history'] = history
    
    logger.info(f"Experiment completed: {results['total_rounds']} rounds")
    
    return results

if __name__ == "__main__":
    # Test strategy creation
    from anomaly_transformer import create_model
    
    # Create model
    input_dim = 123  # WADI features
    model = create_model(input_dim)
    
    # Create strategy
    config = {
        'min_fit_clients': 3,
        'frozen_encoder': False,
        'experiment_name': 'test_experiment'
    }
    
    strategy = create_strategy(model, 'fedavg', config)
    print(f"Strategy created: {strategy.algorithm}")
    print(f"CSV file: {strategy.csv_file}")