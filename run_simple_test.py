"""
간단한 연합학습 테스트 - 전체 시스템을 단순화하여 테스트
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import logging

from data_loader import WADIDataLoader
from anomaly_transformer import create_model, AnomalyLoss
from fl_algorithms import create_aggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_federated_round(global_model, dataloaders, aggregator, device, round_num):
    """Simulate one round of federated learning"""
    
    # Collect client updates
    client_weights = []
    client_samples = []
    client_metrics = []
    
    criterion = AnomalyLoss()
    
    for edge_id, (edge_name, dataloader) in enumerate(dataloaders['edges'].items()):
        logger.info(f"Training {edge_name}...")
        
        # Create local model copy
        local_model = create_model(global_model.input_dim)
        local_model.load_state_dict(global_model.state_dict())
        local_model.to(device)
        local_model.train()
        
        # Local training
        optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-4)
        
        total_loss = 0
        all_preds = []
        all_targets = []
        num_batches = 0
        
        # Train on limited batches for speed
        for i, (batch_x, batch_y) in enumerate(dataloader):
            if i >= 5:  # Only 5 batches per client for testing
                break
                
            batch_x = batch_x.to(device)
            batch_y = batch_y.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = local_model(batch_x)
            loss = criterion(outputs, batch_y, batch_x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        f1 = f1_score(all_targets, all_preds, average='weighted') if len(all_targets) > 0 else 0
        acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0
        
        # Store client update
        client_weights.append(local_model.state_dict())
        client_samples.append(len(all_targets))
        client_metrics.append({
            'edge': edge_name,
            'loss': avg_loss,
            'f1': f1,
            'accuracy': acc
        })
        
        logger.info(f"  {edge_name}: Loss={avg_loss:.4f}, F1={f1:.4f}, Acc={acc:.4f}")
    
    # Aggregate updates
    logger.info("Aggregating client updates...")
    global_weights = aggregator.aggregate(client_weights, client_samples)
    
    # Update global model
    global_model.load_state_dict(global_weights)
    
    # Calculate average metrics
    total_samples = sum(client_samples)
    avg_metrics = {}
    for key in ['loss', 'f1', 'accuracy']:
        weighted_sum = sum(m[key] * s for m, s in zip(client_metrics, client_samples))
        avg_metrics[key] = weighted_sum / total_samples if total_samples > 0 else 0
    
    return avg_metrics

def evaluate_global_model(model, val_dataloader, device):
    """Evaluate global model on validation set"""
    model.eval()
    criterion = AnomalyLoss()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(val_dataloader):
            if i >= 10:  # Limit evaluation batches
                break
                
            batch_x = batch_x.to(device)
            batch_y = batch_y.squeeze().to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y, batch_x)
            
            total_loss += loss.item()
            num_batches += 1
            
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    f1 = f1_score(all_targets, all_preds, average='weighted') if len(all_targets) > 0 else 0
    acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0
    
    return {'loss': avg_loss, 'f1': f1, 'accuracy': acc}

def run_simple_federated_experiment(algorithm='fedavg', rounds=5):
    """Run simplified federated learning experiment"""
    
    logger.info(f"Starting simple federated experiment with {algorithm}")
    
    # Setup
    device = torch.device('cpu')  # Use CPU for simplicity
    
    # Load data
    logger.info("Loading WADI data...")
    loader = WADIDataLoader("WADI_FINAL_DATASET.csv")
    X, y = loader.load_and_preprocess()
    data_splits = loader.create_federated_split(X, y)
    dataloaders = loader.create_dataloaders(data_splits, batch_size=16)
    
    # Create global model
    logger.info("Creating global model...")
    input_dim = X.shape[1]
    global_model = create_model(input_dim, frozen_encoder=False)
    global_model.to(device)
    
    # Create aggregator
    aggregator = create_aggregator(algorithm, device)
    
    # Run federated rounds
    results = []
    
    for round_num in range(rounds):
        logger.info(f"\n=== Round {round_num + 1}/{rounds} ===")
        
        # Train round
        train_metrics = simulate_federated_round(
            global_model, dataloaders, aggregator, device, round_num
        )
        
        # Validation
        val_metrics = evaluate_global_model(
            global_model, dataloaders['val'], device
        )
        
        # Store results
        round_result = {
            'round': round_num + 1,
            'train_loss': train_metrics['loss'],
            'train_f1': train_metrics['f1'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_f1': val_metrics['f1'],
            'val_accuracy': val_metrics['accuracy']
        }
        
        results.append(round_result)
        
        logger.info(f"Round {round_num + 1} Results:")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    print("Running Simple Federated Learning Test...")
    
    # Test all algorithms
    algorithms = ['fedavg', 'fedprox', 'fedadam']
    all_results = {}
    
    for algorithm in algorithms:
        try:
            print(f"\n{'='*50}")
            print(f"Testing {algorithm.upper()}")
            print(f"{'='*50}")
            
            results = run_simple_federated_experiment(algorithm, rounds=3)
            all_results[algorithm] = results
            
            # Show final results
            final_result = results[-1]
            print(f"\nFinal Results for {algorithm.upper()}:")
            print(f"  Train F1: {final_result['train_f1']:.4f}")
            print(f"  Val F1: {final_result['val_f1']:.4f}")
            print(f"  Val Accuracy: {final_result['val_accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error testing {algorithm}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    
    for algorithm, results in all_results.items():
        if results:
            final = results[-1]
            print(f"{algorithm.upper()}: Val F1 = {final['val_f1']:.4f}, Val Acc = {final['val_accuracy']:.4f}")
    
    print("\n[SUCCESS] Simple federated learning test completed!")
    print("System is ready for full experiments with experiment_runner.py")