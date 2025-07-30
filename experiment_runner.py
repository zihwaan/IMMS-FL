import torch
import multiprocessing as mp
import subprocess
import time
import os
import json
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime
import argparse

from data_loader import WADIDataLoader
from anomaly_transformer import create_model, pretrain_encoder
from fl_server import create_strategy, start_server
from fl_client import create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Experiment runner for federated anomaly detection"""
    
    def __init__(self, data_path: str = "WADI_FINAL_DATASET.csv"):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load and prepare data
        self.data_loader = WADIDataLoader(data_path)
        self.X, self.y = self.data_loader.load_and_preprocess()
        self.data_splits = self.data_loader.create_federated_split(self.X, self.y)
        self.dataloaders = self.data_loader.create_dataloaders(self.data_splits)
        
        # Print data statistics
        stats = self.data_loader.get_data_stats(self.data_splits)
        logger.info("Data Statistics:")
        self._print_data_stats(stats)
        
    def _print_data_stats(self, stats: Dict):
        """Print data statistics"""
        for split_name, split_stats in stats.items():
            print(f"\n{split_name.upper()}:")
            if split_name == 'edges':
                for edge_name, edge_stats in split_stats.items():
                    print(f"  {edge_name}:")
                    for key, value in edge_stats.items():
                        print(f"    {key}: {value}")
            else:
                for key, value in split_stats.items():
                    print(f"  {key}: {value}")
    
    def run_experiment(self, 
                      algorithm: str,
                      frozen_encoder: bool = False,
                      num_rounds: int = 100,
                      num_clients: int = 3,
                      pretrain_epochs: int = 10) -> Dict:
        """Run a single federated learning experiment"""
        
        experiment_name = f"{algorithm}_{'frozen' if frozen_encoder else 'full'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting experiment: {experiment_name}")
        
        # Create model
        input_dim = self.X.shape[1]
        model_config = {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'd_ff': 1024,
            'max_seq_len': 100,
            'dropout': 0.1
        }
        
        model = create_model(input_dim, model_config, frozen_encoder=False)
        
        # Pretrain encoder if using frozen encoder strategy
        if frozen_encoder:
            logger.info("Pretraining encoder on normal data...")
            pretrain_dataloader = self.dataloaders['pretrain']
            model = pretrain_encoder(model, pretrain_dataloader, self.device, 
                                   epochs=pretrain_epochs, lr=1e-4)
            model.freeze_encoder()
            logger.info("Encoder frozen for federated training")
        
        # Server configuration
        server_config = {
            'min_fit_clients': num_clients,
            'min_evaluate_clients': num_clients,
            'min_available_clients': num_clients,
            'frozen_encoder': frozen_encoder,
            'results_dir': 'results',
            'experiment_name': experiment_name,
            'num_rounds': num_rounds,
            'server_address': '0.0.0.0:8080'
        }
        
        # Add algorithm-specific parameters
        if algorithm.lower() == 'fedprox':
            server_config['mu'] = 0.01
        elif algorithm.lower() == 'fedadam':
            server_config['eta'] = 1e-3
            server_config['beta1'] = 0.9
            server_config['beta2'] = 0.999
            server_config['epsilon'] = 1e-8
        
        # Create strategy
        strategy = create_strategy(model, algorithm, server_config)
        
        # Start server in background
        server_process = mp.Process(
            target=self._run_server,
            args=(strategy, server_config)
        )
        server_process.start()
        
        # Wait for server to start
        time.sleep(5)
        
        try:
            # Start clients
            client_processes = []
            for client_id in range(num_clients):
                client_process = mp.Process(
                    target=self._run_client,
                    args=(client_id, model, algorithm, frozen_encoder, server_config['server_address'])
                )
                client_processes.append(client_process)
                client_process.start()
            
            # Wait for all clients to finish
            for process in client_processes:
                process.join()
            
            # Wait for server to finish
            server_process.join()
            
            # Load results
            results = self._load_experiment_results(experiment_name)
            
            logger.info(f"Experiment {experiment_name} completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            # Clean up processes
            server_process.terminate()
            for process in client_processes:
                process.terminate()
            raise e
    
    def _run_server(self, strategy, config):
        """Run server process"""
        try:
            import flwr as fl
            
            server_config = fl.server.ServerConfig(num_rounds=config['num_rounds'])
            fl.server.start_server(
                server_address=config['server_address'],
                config=server_config,
                strategy=strategy
            )
        except Exception as e:
            logger.error(f"Server error: {e}")
    
    def _run_client(self, client_id: int, model, algorithm: str, frozen_encoder: bool, server_address: str):
        """Run client process"""
        try:
            import flwr as fl
            
            # Get client data
            edge_key = f'edge_{client_id}'
            trainloader = self.dataloaders['edges'][edge_key]
            valloader = self.dataloaders['val']
            
            # Create client
            client_config = {
                'frozen_encoder': frozen_encoder,
                'algorithm': algorithm
            }
            
            client = create_client(client_id, model, trainloader, valloader, 
                                 self.device, client_config)
            
            # Start client
            fl.client.start_numpy_client(server_address=server_address, client=client)
            
        except Exception as e:
            logger.error(f"Client {client_id} error: {e}")
    
    def _load_experiment_results(self, experiment_name: str) -> Dict:
        """Load experiment results from CSV"""
        csv_file = os.path.join('results', f'{experiment_name}_history.csv')
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            return {
                'experiment_name': experiment_name,
                'total_rounds': len(df),
                'final_metrics': df.iloc[-1].to_dict() if len(df) > 0 else {},
                'history': df.to_dict('records'),
                'csv_file': csv_file
            }
        else:
            logger.warning(f"Results file not found: {csv_file}")
            return {'experiment_name': experiment_name, 'error': 'Results file not found'}
    
    def run_all_experiments(self, 
                           algorithms: List[str] = ['fedavg', 'fedprox', 'fedadam'],
                           encoder_strategies: List[bool] = [False, True],
                           num_rounds: int = 100) -> Dict:
        """Run all experiment combinations"""
        
        all_results = {}
        total_experiments = len(algorithms) * len(encoder_strategies)
        experiment_count = 0
        
        logger.info(f"Starting {total_experiments} experiments...")
        
        for algorithm in algorithms:
            for frozen_encoder in encoder_strategies:
                experiment_count += 1
                logger.info(f"Experiment {experiment_count}/{total_experiments}: "
                           f"{algorithm} with {'frozen' if frozen_encoder else 'full'} encoder")
                
                try:
                    results = self.run_experiment(
                        algorithm=algorithm,
                        frozen_encoder=frozen_encoder,
                        num_rounds=num_rounds
                    )
                    
                    experiment_key = f"{algorithm}_{'frozen' if frozen_encoder else 'full'}"
                    all_results[experiment_key] = results
                    
                    # Save intermediate results
                    self._save_all_results(all_results)
                    
                    # Wait between experiments
                    time.sleep(10)
                    
                except Exception as e:
                    logger.error(f"Experiment {experiment_count} failed: {e}")
                    continue
        
        logger.info(f"All experiments completed. Results saved to results/")
        return all_results
    
    def _save_all_results(self, results: Dict):
        """Save all experiment results to JSON"""
        results_file = os.path.join('results', f'all_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        # Convert any numpy/torch objects to serializable format
        serializable_results = {}
        for key, value in results.items():
            serializable_results[key] = self._make_serializable(value)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, torch.device)):
            return str(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def analyze_results(self, results: Dict) -> Dict:
        """Analyze experiment results"""
        analysis = {
            'summary': {},
            'algorithm_comparison': {},
            'encoder_comparison': {},
            'communication_analysis': {}
        }
        
        # Extract final metrics for each experiment
        for exp_name, exp_results in results.items():
            if 'final_metrics' in exp_results:
                final_metrics = exp_results['final_metrics']
                analysis['summary'][exp_name] = {
                    'final_f1': final_metrics.get('val_f1_avg', 0),
                    'final_accuracy': final_metrics.get('val_accuracy_avg', 0),
                    'final_loss': final_metrics.get('val_loss_avg', 0),
                    'total_rounds': exp_results.get('total_rounds', 0)
                }
        
        # Compare algorithms
        for encoder_type in ['full', 'frozen']:
            algorithm_results = {}
            for algorithm in ['fedavg', 'fedprox', 'fedadam']:
                exp_key = f"{algorithm}_{encoder_type}"
                if exp_key in analysis['summary']:
                    algorithm_results[algorithm] = analysis['summary'][exp_key]
            analysis['algorithm_comparison'][encoder_type] = algorithm_results
        
        # Compare encoder strategies
        for algorithm in ['fedavg', 'fedprox', 'fedadam']:
            encoder_results = {}
            for encoder_type in ['full', 'frozen']:
                exp_key = f"{algorithm}_{encoder_type}"
                if exp_key in analysis['summary']:
                    encoder_results[encoder_type] = analysis['summary'][exp_key]
            analysis['encoder_comparison'][algorithm] = encoder_results
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description='Run federated anomaly detection experiments')
    parser.add_argument('--data_path', type=str, default='WADI_FINAL_DATASET.csv',
                       help='Path to WADI dataset')
    parser.add_argument('--algorithm', type=str, choices=['fedavg', 'fedprox', 'fedadam', 'all'],
                       default='all', help='Aggregation algorithm to run')
    parser.add_argument('--encoder', type=str, choices=['full', 'frozen', 'both'],
                       default='both', help='Encoder strategy')
    parser.add_argument('--rounds', type=int, default=100,
                       help='Number of federated learning rounds')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze existing results')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Load and analyze existing results
        results_dir = 'results'
        json_files = [f for f in os.listdir(results_dir) if f.startswith('all_experiments_') and f.endswith('.json')]
        
        if json_files:
            latest_file = sorted(json_files)[-1]
            with open(os.path.join(results_dir, latest_file), 'r') as f:
                results = json.load(f)
            
            runner = ExperimentRunner(args.data_path)
            analysis = runner.analyze_results(results)
            
            print("\n=== EXPERIMENT ANALYSIS ===")
            print(json.dumps(analysis, indent=2))
        else:
            print("No experiment results found for analysis")
        
        return
    
    # Create experiment runner
    runner = ExperimentRunner(args.data_path)
    
    # Determine which experiments to run
    algorithms = ['fedavg', 'fedprox', 'fedadam'] if args.algorithm == 'all' else [args.algorithm]
    encoder_strategies = [False, True] if args.encoder == 'both' else [args.encoder == 'frozen']
    
    if len(algorithms) == 1 and len(encoder_strategies) == 1:
        # Run single experiment
        results = runner.run_experiment(
            algorithm=algorithms[0],
            frozen_encoder=encoder_strategies[0],
            num_rounds=args.rounds
        )
        print(f"\nExperiment completed: {results['experiment_name']}")
    else:
        # Run multiple experiments
        results = runner.run_all_experiments(
            algorithms=algorithms,
            encoder_strategies=encoder_strategies,
            num_rounds=args.rounds
        )
        
        # Analyze results
        analysis = runner.analyze_results(results)
        print("\n=== EXPERIMENT ANALYSIS ===")
        print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Required for multiprocessing on Windows
    main()