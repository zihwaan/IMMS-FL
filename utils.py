import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import logging
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributionMonitor:
    """Monitor and detect distribution shifts in federated learning"""
    
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
        self.baseline_distributions = {}
        self.shift_history = []
        
    def set_baseline(self, client_id: int, data: np.ndarray):
        """Set baseline distribution for a client"""
        if data.ndim > 1:
            data = data.flatten()
        self.baseline_distributions[client_id] = data
        logger.info(f"Baseline distribution set for client {client_id}")
    
    def detect_shift(self, client_id: int, current_data: np.ndarray, round_num: int) -> Dict:
        """Detect distribution shift for a client"""
        if client_id not in self.baseline_distributions:
            logger.warning(f"No baseline set for client {client_id}")
            return {'shift_detected': False, 'distance': 0.0}
        
        if current_data.ndim > 1:
            current_data = current_data.flatten()
        
        baseline = self.baseline_distributions[client_id]
        
        # Calculate Wasserstein distance
        try:
            wd = wasserstein_distance(baseline, current_data)
        except Exception as e:
            logger.error(f"Error calculating Wasserstein distance: {e}")
            return {'shift_detected': False, 'distance': 0.0, 'error': str(e)}
        
        shift_detected = wd > self.threshold
        
        # Log shift detection
        shift_info = {
            'client_id': client_id,
            'round': round_num,
            'wasserstein_distance': wd,
            'threshold': self.threshold,
            'shift_detected': shift_detected,
            'timestamp': datetime.now().isoformat()
        }
        
        self.shift_history.append(shift_info)
        
        if shift_detected:
            logger.warning(f"Distribution shift detected for client {client_id} "
                          f"at round {round_num}: WD = {wd:.4f} > {self.threshold}")
        
        return shift_info
    
    def get_shift_history(self) -> List[Dict]:
        """Get history of all detected shifts"""
        return self.shift_history
    
    def trigger_rebalancing(self, client_id: int, round_num: int) -> Dict:
        """Trigger data rebalancing (placeholder for actual implementation)"""
        rebalancing_info = {
            'client_id': client_id,
            'round': round_num,
            'action': 'rebalancing_triggered',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Rebalancing triggered for client {client_id} at round {round_num}")
        
        return rebalancing_info

class DistributionShiftSimulator:
    """Simulate various types of distribution shifts"""
    
    @staticmethod
    def apply_scale_shift(data: np.ndarray, intensity: float = 2.0) -> np.ndarray:
        """Apply scaling shift to data"""
        return data * intensity
    
    @staticmethod
    def apply_offset_shift(data: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Apply offset shift to data"""
        return data + intensity
    
    @staticmethod
    def apply_noise_shift(data: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply noise-based shift to data"""
        noise = np.random.normal(0, intensity * np.std(data), data.shape)
        return data + noise
    
    @staticmethod
    def apply_gradual_shift(data: np.ndarray, progress: float, max_intensity: float = 2.0) -> np.ndarray:
        """Apply gradual shift based on progress (0 to 1)"""
        current_intensity = progress * max_intensity
        return DistributionShiftSimulator.apply_scale_shift(data, 1 + current_intensity)
    
    @staticmethod
    def simulate_attack_pattern_change(data: np.ndarray, attack_indices: np.ndarray, 
                                     intensity: float = 2.0) -> np.ndarray:
        """Simulate change in attack patterns"""
        shifted_data = data.copy()
        shifted_data[attack_indices] *= intensity
        return shifted_data

class MetricsCalculator:
    """Calculate and track various metrics for federated learning"""
    
    @staticmethod
    def calculate_f1_recovery_speed(f1_scores: List[float], shift_round: int) -> Dict:
        """Calculate F1 recovery speed after distribution shift"""
        if len(f1_scores) <= shift_round:
            return {'recovery_speed': 0, 'rounds_to_recover': float('inf')}
        
        pre_shift_f1 = np.mean(f1_scores[:shift_round])
        post_shift_scores = f1_scores[shift_round:]
        
        # Find when F1 recovers to 95% of pre-shift level
        recovery_threshold = 0.95 * pre_shift_f1
        recovery_round = None
        
        for i, score in enumerate(post_shift_scores):
            if score >= recovery_threshold:
                recovery_round = i + 1  # Rounds after shift
                break
        
        if recovery_round is None:
            recovery_round = len(post_shift_scores)
        
        recovery_speed = 1 / recovery_round if recovery_round > 0 else 0
        
        return {
            'pre_shift_f1': pre_shift_f1,
            'recovery_threshold': recovery_threshold,
            'rounds_to_recover': recovery_round,
            'recovery_speed': recovery_speed
        }
    
    @staticmethod
    def calculate_communication_efficiency(total_params: int, frozen_params: int) -> Dict:
        """Calculate communication efficiency metrics"""
        trainable_params = total_params - frozen_params
        reduction_ratio = frozen_params / total_params if total_params > 0 else 0
        
        return {
            'total_parameters': total_params,
            'frozen_parameters': frozen_params,
            'trainable_parameters': trainable_params,
            'communication_reduction': reduction_ratio,
            'efficiency_gain': reduction_ratio * 100  # Percentage
        }
    
    @staticmethod
    def calculate_aggregation_stability(loss_history: List[float], window_size: int = 10) -> Dict:
        """Calculate stability metrics for aggregation algorithms"""
        if len(loss_history) < window_size:
            return {'stability_score': 0, 'variance': float('inf')}
        
        # Calculate moving variance
        variances = []
        for i in range(window_size, len(loss_history)):
            window = loss_history[i-window_size:i]
            variances.append(np.var(window))
        
        avg_variance = np.mean(variances) if variances else float('inf')
        stability_score = 1 / (1 + avg_variance)  # Higher score = more stable
        
        return {
            'average_variance': avg_variance,
            'stability_score': stability_score,
            'convergence_rate': -np.gradient(loss_history[-window_size:]).mean()
        }

class ResultsAnalyzer:
    """Analyze and visualize federated learning experiment results"""
    
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def compare_algorithms(self, experiment_results: Dict) -> Dict:
        """Compare different aggregation algorithms"""
        comparison = {}
        
        for exp_name, results in experiment_results.items():
            if 'history' not in results:
                continue
            
            df = pd.DataFrame(results['history'])
            
            algorithm = exp_name.split('_')[0]
            encoder_type = 'frozen' if 'frozen' in exp_name else 'full'
            
            if algorithm not in comparison:
                comparison[algorithm] = {}
            
            comparison[algorithm][encoder_type] = {
                'final_f1': df['val_f1_avg'].iloc[-1] if 'val_f1_avg' in df else 0,
                'final_accuracy': df['val_accuracy_avg'].iloc[-1] if 'val_accuracy_avg' in df else 0,
                'final_loss': df['val_loss_avg'].iloc[-1] if 'val_loss_avg' in df else 0,
                'convergence_rounds': self._calculate_convergence_rounds(df),
                'stability_score': MetricsCalculator.calculate_aggregation_stability(
                    df['train_loss_avg'].tolist() if 'train_loss_avg' in df else []
                )['stability_score']
            }
        
        return comparison
    
    def _calculate_convergence_rounds(self, df: pd.DataFrame) -> int:
        """Calculate number of rounds to convergence"""
        if 'val_f1_avg' not in df or len(df) < 10:
            return len(df)
        
        # Define convergence as when F1 stops improving significantly
        f1_scores = df['val_f1_avg'].tolist()
        window_size = 5
        improvement_threshold = 0.001
        
        for i in range(window_size, len(f1_scores)):
            recent_window = f1_scores[i-window_size:i]
            if max(recent_window) - min(recent_window) < improvement_threshold:
                return i
        
        return len(f1_scores)
    
    def generate_plots(self, experiment_results: Dict):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Federated Learning Experiment Results', fontsize=16)
        
        # Plot 1: F1 Score Comparison
        ax1 = axes[0, 0]
        self._plot_metric_comparison(experiment_results, 'val_f1_avg', 'F1 Score', ax1)
        
        # Plot 2: Accuracy Comparison
        ax2 = axes[0, 1]
        self._plot_metric_comparison(experiment_results, 'val_accuracy_avg', 'Accuracy', ax2)
        
        # Plot 3: Loss Comparison
        ax3 = axes[1, 0]
        self._plot_metric_comparison(experiment_results, 'val_loss_avg', 'Validation Loss', ax3)
        
        # Plot 4: Algorithm Comparison Bar Chart
        ax4 = axes[1, 1]
        self._plot_algorithm_comparison(experiment_results, ax4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'experiment_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {self.results_dir}/experiment_results.png")
    
    def _plot_metric_comparison(self, experiment_results: Dict, metric: str, 
                              title: str, ax):
        """Plot metric comparison across experiments"""
        for exp_name, results in experiment_results.items():
            if 'history' not in results:
                continue
            
            df = pd.DataFrame(results['history'])
            if metric in df:
                ax.plot(df['round'], df[metric], label=exp_name, marker='o', markersize=3)
        
        ax.set_xlabel('Round')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Over Rounds')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_algorithm_comparison(self, experiment_results: Dict, ax):
        """Plot final performance comparison"""
        algorithms = ['fedavg', 'fedprox', 'fedadam']
        encoder_types = ['full', 'frozen']
        
        data = {}
        for algorithm in algorithms:
            data[algorithm] = {}
            for encoder in encoder_types:
                exp_name = f"{algorithm}_{encoder}"
                if exp_name in experiment_results and 'history' in experiment_results[exp_name]:
                    df = pd.DataFrame(experiment_results[exp_name]['history'])
                    data[algorithm][encoder] = df['val_f1_avg'].iloc[-1] if 'val_f1_avg' in df else 0
                else:
                    data[algorithm][encoder] = 0
        
        # Create bar chart
        x = np.arange(len(algorithms))
        width = 0.35
        
        full_scores = [data[alg]['full'] for alg in algorithms]
        frozen_scores = [data[alg]['frozen'] for alg in algorithms]
        
        ax.bar(x - width/2, full_scores, width, label='Full Model', alpha=0.8)
        ax.bar(x + width/2, frozen_scores, width, label='Frozen Encoder', alpha=0.8)
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Final F1 Score')
        ax.set_title('Final Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([alg.upper() for alg in algorithms])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def generate_summary_report(self, experiment_results: Dict, analysis: Dict) -> str:
        """Generate text summary report"""
        report = []
        report.append("# Federated Anomaly Detection Experiment Results\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary statistics
        report.append("## Summary Statistics\n")
        for exp_name, metrics in analysis.get('summary', {}).items():
            report.append(f"### {exp_name.upper()}")
            report.append(f"- Final F1 Score: {metrics.get('final_f1', 0):.4f}")
            report.append(f"- Final Accuracy: {metrics.get('final_accuracy', 0):.4f}")
            report.append(f"- Final Loss: {metrics.get('final_loss', 0):.4f}")
            report.append(f"- Total Rounds: {metrics.get('total_rounds', 0)}\n")
        
        # Algorithm comparison
        report.append("## Algorithm Comparison\n")
        for encoder_type, algorithms in analysis.get('algorithm_comparison', {}).items():
            report.append(f"### {encoder_type.upper()} Encoder Strategy")
            sorted_algorithms = sorted(algorithms.items(), 
                                     key=lambda x: x[1].get('final_f1', 0), 
                                     reverse=True)
            for i, (alg, metrics) in enumerate(sorted_algorithms, 1):
                report.append(f"{i}. **{alg.upper()}**: F1={metrics.get('final_f1', 0):.4f}, "
                            f"Acc={metrics.get('final_accuracy', 0):.4f}")
            report.append("")
        
        # Encoder strategy comparison
        report.append("## Encoder Strategy Comparison\n")
        for algorithm, encoders in analysis.get('encoder_comparison', {}).items():
            report.append(f"### {algorithm.upper()}")
            for encoder_type, metrics in encoders.items():
                report.append(f"- {encoder_type.title()}: F1={metrics.get('final_f1', 0):.4f}")
            
            # Calculate communication savings
            if 'full' in encoders and 'frozen' in encoders:
                full_f1 = encoders['full'].get('final_f1', 0)
                frozen_f1 = encoders['frozen'].get('final_f1', 0)
                performance_diff = ((frozen_f1 - full_f1) / full_f1 * 100) if full_f1 > 0 else 0
                report.append(f"- Performance difference: {performance_diff:+.2f}%")
                report.append("- Estimated communication savings: ~50%")
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = os.path.join(self.results_dir, 'experiment_report.md')
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Summary report saved to {report_file}")
        return report_text

def load_and_analyze_results(results_dir: str = 'results') -> Dict:
    """Load and analyze all experiment results"""
    import glob
    import json
    
    # Find all result JSON files
    json_files = glob.glob(os.path.join(results_dir, 'all_experiments_*.json'))
    
    if not json_files:
        logger.warning("No experiment result files found")
        return {}
    
    # Load the most recent results
    latest_file = sorted(json_files)[-1]
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Analyze results
    analyzer = ResultsAnalyzer(results_dir)
    analysis = analyzer.compare_algorithms(results)
    
    # Generate visualizations and report
    analyzer.generate_plots(results)
    analyzer.generate_summary_report(results, {'algorithm_comparison': analysis})
    
    return results

if __name__ == "__main__":
    # Test distribution monitoring
    monitor = DistributionMonitor(threshold=0.2)
    
    # Simulate data
    baseline_data = np.random.normal(0, 1, 1000)
    shifted_data = baseline_data * 2 + np.random.normal(0, 0.1, 1000)
    
    monitor.set_baseline(0, baseline_data)
    shift_info = monitor.detect_shift(0, shifted_data, 50)
    
    print(f"Shift detection result: {shift_info}")
    
    # Test metrics calculator
    f1_scores = [0.8, 0.82, 0.81, 0.6, 0.65, 0.75, 0.82, 0.84]  # Simulated F1 scores
    recovery_metrics = MetricsCalculator.calculate_f1_recovery_speed(f1_scores, 3)
    print(f"F1 recovery metrics: {recovery_metrics}")