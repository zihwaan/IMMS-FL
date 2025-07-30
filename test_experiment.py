"""
간단한 테스트 실험 - 시스템이 제대로 작동하는지 확인
"""
import torch
import numpy as np
from data_loader import WADIDataLoader
from anomaly_transformer import create_model, pretrain_encoder
from fl_algorithms import create_aggregator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic functionality without full federated learning"""
    
    print("=== Testing Basic Functionality ===")
    
    # 1. Test data loading
    print("\n1. Testing data loading...")
    try:
        loader = WADIDataLoader("WADI_FINAL_DATASET.csv")
        X, y = loader.load_and_preprocess()
        data_splits = loader.create_federated_split(X, y)
        print(f"[OK] Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"[OK] Data split into {len(data_splits['edges'])} edges")
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        return False
    
    # 2. Test model creation
    print("\n2. Testing model creation...")
    try:
        device = torch.device('cpu')  # Use CPU for testing
        input_dim = X.shape[1]
        
        # Test full model
        model_full = create_model(input_dim, frozen_encoder=False)
        print(f"[OK] Full model created: {sum(p.numel() for p in model_full.parameters())} parameters")
        
        # Test frozen model
        model_frozen = create_model(input_dim, frozen_encoder=True)
        trainable_params = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
        print(f"[OK] Frozen model created: {trainable_params} trainable parameters")
        
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        return False
    
    # 3. Test model forward pass
    print("\n3. Testing model forward pass...")
    try:
        batch_size = 4
        seq_len = 60
        x_test = torch.randn(batch_size, seq_len, input_dim)
        
        with torch.no_grad():
            outputs = model_full(x_test)
            print(f"[OK] Forward pass successful: output shape {outputs['logits'].shape}")
            
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return False
    
    # 4. Test aggregation algorithms
    print("\n4. Testing aggregation algorithms...")
    try:
        # Create dummy client weights
        client_weights = []
        for i in range(3):
            client_weights.append(model_full.state_dict())
        client_samples = [100, 120, 80]
        
        for algorithm in ['fedavg', 'fedprox', 'fedadam']:
            aggregator = create_aggregator(algorithm, device)
            global_weights = aggregator.aggregate(client_weights, client_samples)
            print(f"[OK] {algorithm.upper()} aggregation successful")
            
    except Exception as e:
        print(f"[FAIL] Aggregation failed: {e}")
        return False
    
    # 5. Test data loaders
    print("\n5. Testing data loaders...")
    try:
        dataloaders = loader.create_dataloaders(data_splits, batch_size=32)
        
        # Test each dataloader
        for split_name, dataloader_dict in dataloaders.items():
            if split_name == 'edges':
                for edge_name, dl in dataloader_dict.items():
                    batch_x, batch_y = next(iter(dl))
                    print(f"[OK] {edge_name} dataloader: {batch_x.shape}")
            else:
                batch_x, batch_y = next(iter(dataloader_dict))
                print(f"[OK] {split_name} dataloader: {batch_x.shape}")
                
    except Exception as e:
        print(f"[FAIL] DataLoader test failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    return True

def test_mini_training():
    """Test mini training loop"""
    
    print("\n=== Testing Mini Training Loop ===")
    
    try:
        # Setup
        device = torch.device('cpu')
        loader = WADIDataLoader("WADI_FINAL_DATASET.csv")
        X, y = loader.load_and_preprocess()
        data_splits = loader.create_federated_split(X, y)
        dataloaders = loader.create_dataloaders(data_splits, batch_size=16)
        
        # Create model
        input_dim = X.shape[1]
        model = create_model(input_dim, frozen_encoder=False)
        model.to(device)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Mini training on one edge
        edge_dataloader = dataloaders['edges']['edge_0']
        model.train()
        
        print("Running mini training for 5 batches...")
        for i, (batch_x, batch_y) in enumerate(edge_dataloader):
            if i >= 5:  # Only 5 batches for testing
                break
                
            batch_x = batch_x.to(device)
            batch_y = batch_y.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs['logits'], batch_y)
            loss.backward()
            optimizer.step()
            
            print(f"  Batch {i+1}: Loss = {loss.item():.4f}")
        
        print("[OK] Mini training completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Mini training failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting System Test...")
    
    # Test basic functionality
    basic_test_passed = test_basic_functionality()
    
    if basic_test_passed:
        # Test mini training
        training_test_passed = test_mini_training()
        
        if training_test_passed:
            print("\n[SUCCESS] All tests passed! System is ready for full experiments.")
            print("\nTo run full experiments:")
            print("python experiment_runner.py --algorithm fedavg --encoder full --rounds 10")
        else:
            print("\n[WARNING] Basic tests passed but training test failed. Check model implementation.")
    else:
        print("\n[ERROR] Basic tests failed. Please fix the issues before running experiments.")