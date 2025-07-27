# -*- coding: utf-8 -*-
import os
import logging
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import flwr as fl
from flwr.common import Metrics, Parameters
from flwr.server.strategy import FedAvg, FedProx, FedAdam
import pickle
import csv

from data_loader import prepare_federated_data
from experiment_utils import ModelAdapter, wasserstein_distance, apply_distribution_shift

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WADIClient(fl.client.NumPyClient):
    """WADI federated learning client"""
    
    def __init__(self, cid: str, model_adapter: ModelAdapter, train_loader, test_loader, 
                 freeze_encoder: bool = False, shift_config: Optional[Dict] = None):
        self.cid = cid
        self.model_adapter = model_adapter
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.freeze_encoder = freeze_encoder
        self.shift_config = shift_config
        self.round_num = 0
        
        # For distribution shift tracking
        self.original_data = None
        self.current_wasserstein = 0.0
        
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters"""
        if self.freeze_encoder:
            return self.model_adapter.get_classifier_parameters()
        else:
            return self.model_adapter.get_all_parameters()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters"""
        if self.freeze_encoder:
            self.model_adapter.set_classifier_parameters(parameters)
        else:
            self.model_adapter.set_all_parameters(parameters)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Perform local training"""
        self.round_num = config.get('round', 0)
        
        # Distribution shift simulation (apply to Edge 2 at round 50)
        if self.round_num == 50 and self.cid == "2" and self.shift_config:
            self._apply_distribution_shift()
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Local training
        loss = self.model_adapter.train_one_epoch(self.train_loader)
        
        # Return updated parameters
        updated_params = self.get_parameters(config)
        num_examples = len(self.train_loader.dataset)
        
        # Metrics
        metrics = {
            "loss": loss,
            "wasserstein_distance": self.current_wasserstein
        }
        
        return updated_params, num_examples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Perform local evaluation"""
        self.set_parameters(parameters)
        
        # Evaluation
        loss, f1, acc, conf_matrix = self.model_adapter.evaluate(self.test_loader)
        
        num_examples = len(self.test_loader.dataset)
        
        metrics = {
            "f1_score": f1,
            "accuracy": acc,
            "confusion_matrix": conf_matrix.tolist()
        }
        
        return float(loss), num_examples, metrics
    
    def _apply_distribution_shift(self):
        """Apply distribution shift"""
        if self.original_data is None:
            # Store original data
            self.original_data = []
            for batch_data, _ in self.train_loader:
                self.original_data.append(batch_data.numpy())
            self.original_data = np.vstack(self.original_data)
        
        # Apply distribution shift
        shifted_data = apply_distribution_shift(
            self.original_data, 
            shift_type=self.shift_config['type'],
            intensity=self.shift_config['intensity']
        )
        
        # Calculate Wasserstein distance
        self.current_wasserstein = wasserstein_distance(
            self.original_data.flatten(), 
            shifted_data.flatten()
        )
        
        logger.info(f"Client {self.cid}: Applied distribution shift, "
                   f"Wasserstein distance: {self.current_wasserstein:.4f}")
        
        if self.current_wasserstein > 0.2:
            logger.warning(f"Client {self.cid}: High distribution drift detected! "
                          f"Reallocation needed.")

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate weighted average"""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    # F1 score average
    f1_scores = [m["f1_score"] for _, m in metrics]
    weighted_f1 = sum(f1 * num for (num, _), f1 in zip(metrics, f1_scores)) / total_examples
    
    # Accuracy average
    accuracies = [m["accuracy"] for _, m in metrics]
    weighted_acc = sum(acc * num for (num, _), acc in zip(metrics, accuracies)) / total_examples
    
    return {
        "f1_score": weighted_f1,
        "accuracy": weighted_acc,
        "total_examples": total_examples
    }

class ExperimentLogger:
    """Experiment results logging class"""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.results = []
        
        # Write CSV header
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'f1_score', 'accuracy', 'loss', 
                'wasserstein_distance', 'total_examples'
            ])
    
    def log_round(self, round_num: int, metrics: Dict):
        """Log round results"""
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                metrics.get('f1_score', 0),
                metrics.get('accuracy', 0),
                metrics.get('loss', 0),
                metrics.get('wasserstein_distance', 0),
                metrics.get('total_examples', 0)
            ])

def client_fn(cid: str, federated_data: Dict, model_type: str, freeze_encoder: bool) -> WADIClient:
    """Client factory function"""
    client_id = int(cid)
    client_data = federated_data[client_id]
    
    # Create model adapter
    model_adapter = ModelAdapter(
        model_type=model_type,
        feature_dim=client_data['feature_dim'],
        freeze_encoder=freeze_encoder
    )
    
    # Distribution shift config (apply to Edge 2 only)
    shift_config = None
    if client_id == 2:
        shift_config = {
            'type': 'scale',
            'intensity': 2.0
        }
    
    return WADIClient(
        cid=cid,
        model_adapter=model_adapter,
        train_loader=client_data['train_loader'],
        test_loader=client_data['test_loader'],
        freeze_encoder=freeze_encoder,
        shift_config=shift_config
    )

def get_strategy(strategy_name: str, num_clients: int = 3) -> fl.server.strategy.Strategy:
    """Return federated learning strategy"""
    
    min_fit_clients = num_clients
    min_evaluate_clients = num_clients
    min_available_clients = num_clients
    
    strategy_config = {
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "fit_metrics_aggregation_fn": weighted_average,
    }
    
    if strategy_name == "FedAvg":
        return FedAvg(**strategy_config)
    elif strategy_name == "FedProx":
        return FedProx(proximal_mu=0.1, **strategy_config)
    elif strategy_name == "FedAdam":
        return FedAdam(eta=0.001, **strategy_config)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def run_federated_experiment(
    model_type: str,
    strategy_name: str,
    freeze_encoder: bool = False,
    num_rounds: int = 100,
    csv_path: str = 'WADI_FINAL_DATASET.csv'
) -> str:
    """Run federated learning experiment"""
    
    # Generate experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{model_type}_{strategy_name}"
    if freeze_encoder:
        experiment_id += "_frozen"
    experiment_id += f"_{timestamp}"
    
    # Setup log files
    log_dir = "experiment_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{experiment_id}.log")
    
    logger.info(f"Starting experiment: {experiment_id}")
    
    # Prepare data
    federated_data = prepare_federated_data(
        csv_path=csv_path,
        model_type=model_type,
        window_size=100,
        batch_size=32
    )
    
    # Initialize experiment logger
    exp_logger = ExperimentLogger(log_path.replace('.log', '_metrics.csv'))
    
    # Setup strategy
    strategy = get_strategy(strategy_name)
    
    # Client function setup
    def client_factory(cid: str) -> WADIClient:
        return client_fn(cid, federated_data, model_type, freeze_encoder)
    
    # Run federated learning simulation
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_factory,
            num_clients=3,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1}
        )
        
        # Save results
        results_path = log_path.replace('.log', '_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(history, f)
        
        # Log final metrics
        if history.metrics_distributed:
            final_metrics = history.metrics_distributed.get('f1_score', [])
            if final_metrics:
                final_f1 = final_metrics[-1][1]
                logger.info(f"Experiment {experiment_id} completed. Final F1: {final_f1:.4f}")
        
        logger.info(f"Results saved to: {results_path}")
        return experiment_id
        
    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Run federated learning experiments on WADI dataset')
    parser.add_argument('--model', type=str, choices=['duogat', 'anomaly_transformer'], 
                       default='duogat', help='Model type to use')
    parser.add_argument('--strategy', type=str, choices=['FedAvg', 'FedProx', 'FedAdam'], 
                       default='FedAvg', help='Federated learning strategy')
    parser.add_argument('--freeze_encoder', action='store_true', 
                       help='Freeze encoder and only train classifier')
    parser.add_argument('--num_rounds', type=int, default=100, 
                       help='Number of federated learning rounds')
    parser.add_argument('--csv_path', type=str, default='WADI_FINAL_DATASET.csv',
                       help='Path to WADI dataset CSV file')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment_id = run_federated_experiment(
        model_type=args.model,
        strategy_name=args.strategy,
        freeze_encoder=args.freeze_encoder,
        num_rounds=args.num_rounds,
        csv_path=args.csv_path
    )
    
    print(f"Experiment completed: {experiment_id}")

if __name__ == "__main__":
    main()