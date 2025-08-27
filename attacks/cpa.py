import sys
import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ConfigManager
from datasets import dataset_registry
from distinguisher import DataProcessor, model_registry


def load_config_from_experiment(experiment_path):
    experiment_path = Path(experiment_path)
    yaml_files = list(experiment_path.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No yaml file found in {experiment_path}")
    yaml_path = yaml_files[0]
    print(f"Loading config from: {yaml_path}")
    config_manager = ConfigManager(str(yaml_path))
    return config_manager, yaml_path


def load_metadata_from_data_path(data_path):
    data_path = Path(data_path)
    metadata_path = data_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {metadata_path}")
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def load_model_from_checkpoint(config_manager, checkpoint_path, device):
    model_config = config_manager.get_model_config()
    model_type = model_config['type']
    if 'params' not in model_config:
        raise ValueError(f"Configuration file missing model parameters")
    params = model_config['params'].copy()
    print(f"Creating model: {model_type}")
    model = model_registry.get_model(model_type, **params)
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def generate_cpa_dataset(cipher_type, key, nr, n, diff):
    try:
        dataset_class = dataset_registry.get_dataset_class(cipher_type)
        dataset = dataset_class(n=n, nr=nr, key_mode='input_fixed', key=key, diff=diff)
        
        if hasattr(dataset, 'generate_plaintext_triples'):
            # Step 1: Generate plaintext triples: A, A+diff, B
            p1_list, p1_prime_list, p2_list = dataset.generate_plaintext_triples(n, diff)
            
            # Step 2: Encrypt A and B to get C(A) and C(B)
            c1_list, c2_list = dataset.encrypt_plaintext_pairs(p1_list, p2_list, nr)
            
            # Step 3: For inference, let all C come from A
            c_list = c1_list  # All C come from A
            
            # Step 4: Apply differential to B to get B', then encrypt A' and B'
            p2_prime_list = []
            for p2 in p2_list:
                p2_prime_l = p2[0] ^ diff[0]
                p2_prime_r = p2[1] ^ diff[1]
                p2_prime_list.append((p2_prime_l, p2_prime_r))
            
            c1_prime_list, c2_prime_list = dataset.encrypt_plaintext_pairs(p1_prime_list, p2_prime_list, nr)
            
            # Step 5: Convert to CPA attack format: [E(A'), C] and [E(B'), C]
            c1_prime_l = np.array([ct[0] for ct in c1_prime_list])
            c1_prime_r = np.array([ct[1] for ct in c1_prime_list])
            c2_prime_l = np.array([ct[0] for ct in c2_prime_list])
            c2_prime_r = np.array([ct[1] for ct in c2_prime_list])
            c_l = np.array([ct[0] for ct in c_list])
            c_r = np.array([ct[1] for ct in c_list])
            
            # Convert to bit format: [E(A'), C] and [E(B'), C]
            A_prime_C = dataset.cipher.to_bits([c1_prime_l, c1_prime_r, c_l, c_r])
            B_prime_C = dataset.cipher.to_bits([c2_prime_l, c2_prime_r, c_l, c_r])
            
        else:
            raise ValueError(f"Dataset {cipher_type} does not support plaintext triples generation")
            
    except Exception as e:
        raise ValueError(f"Cannot create dataset or generate CPA data: {e}")
    
    print(f"Data shape: {A_prime_C.shape}")
    
    return A_prime_C, B_prime_C


def evaluate_cpa_attack(model, A_prime_C, B_prime_C, config_manager, device, batch_size=5000):
    # Get processor configuration
    processor_config = config_manager.get('processor', {})
    processor = DataProcessor(processor_config)
    
    # Batch evaluation
    model.eval()
    confidence_scores = []
    correct_predictions = 0
    total_predictions = len(A_prime_C)
    
    with torch.no_grad():
        for i in tqdm(range(0, total_predictions, batch_size), desc="Evaluating CPA attack"):
            end_idx = min(i + batch_size, total_predictions)

            # Get current batch data
            A_prime_C_batch = A_prime_C[i:end_idx]
            B_prime_C_batch = B_prime_C[i:end_idx]
            
            # Convert to tensor
            A_prime_C_tensor = torch.tensor(A_prime_C_batch, dtype=torch.float32, device=device)
            B_prime_C_tensor = torch.tensor(B_prime_C_batch, dtype=torch.float32, device=device)
            
            # Process data using processor
            A_prime_C_processed = processor.process(A_prime_C_tensor)
            B_prime_C_processed = processor.process(B_prime_C_tensor)
            
            # Model prediction
            logits_A = model(A_prime_C_processed)
            logits_B = model(B_prime_C_processed)
            
            # Convert to probabilities
            probs_A = torch.sigmoid(logits_A).cpu().numpy().flatten()
            probs_B = torch.sigmoid(logits_B).cpu().numpy().flatten()
            
            # Calculate confidence and predictions
            for j in range(end_idx - i):
                prob_A = probs_A[j]
                prob_B = probs_B[j]
                
                # confidence = abs(prob_A - prob_B)
                confidence = abs(prob_A - prob_B)
                confidence_scores.append(confidence)
                
                # Prediction: if prob_A > prob_B, predict C comes from A; otherwise predict C comes from B
                # Since all C actually come from A, prob_A > prob_B is correct prediction
                if prob_A > prob_B:
                    correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    
    print(f"Results: {correct_predictions}/{total_predictions} correct, accuracy={accuracy:.4f}, confidence={np.mean(confidence_scores):.4f}±{np.std(confidence_scores):.4f}")
    
    return accuracy, confidence_scores


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        args: parsed arguments
    """
    parser = argparse.ArgumentParser(description='CPA Attack Accuracy Testing')
    parser.add_argument('--exp', type=str, required=True, help='Experiment folder path containing yaml config and checkpoint files')
    parser.add_argument('--nr', type=int, default=None, help='Number of encryption rounds, default from metadata nr')
    parser.add_argument('--ckpt', type=int, default=100, help='Checkpoint number, default 100 (checkpoint_100.pth)')
    parser.add_argument('--n', type=int, default=1000000, help='Number of test samples, default 10^6')
    parser.add_argument('--bs', type=int, default=5000, help='Batch size, default 5000')
    parser.add_argument('--device', type=str, default=None, help='Device, default auto-detect')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()
    np.random.seed(42)

    print("=" * 60)
    print("CPA Attack Accuracy Testing")
    print("=" * 60)
    print(f"Experiment path: {args.exp}")
    print(f"Test sample count: {args.n}")
    
    # Load configuration
    try:
        config_manager, yaml_path = load_config_from_experiment(args.exp)
    except Exception as e:
        print(f"Configuration loading failed: {e}")
        return
    
    # Get data path
    data_path = config_manager.get('data.path')
    print(f"Data path: {data_path}")
    
    # Load metadata
    try:
        metadata = load_metadata_from_data_path(data_path)
    except Exception as e:
        print(f"Metadata loading failed: {e}")
        return
    
    # Extract key information
    nr_meta = metadata['nr']
    key_meta = metadata['key']
    cipher_type_meta = metadata['cipher_type']
    diff_meta = metadata['diff']
    
    # Determine rounds to use
    nr = args.nr if args.nr is not None else nr_meta
    
    # Check if checkpoint file exists
    experiment_path = Path(args.exp)
    checkpoint_path = experiment_path / f"checkpoint_{args.ckpt}.pth"
    
    if not checkpoint_path.exists():
        print(f"Checkpoint file does not exist: {checkpoint_path}")
        print("Available checkpoint files:")
        for checkpoint_file in experiment_path.glob("checkpoint_*.pth"):
            print(f"  - {checkpoint_file.name}")
        return
    
    # Set device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Batch size: {args.bs}")
    
    # Load model
    try:
        model = load_model_from_checkpoint(config_manager, checkpoint_path, device)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    print("Starting CPA attack data generation...")
    
    # Generate CPA attack dataset
    try:
        A_prime_C, B_prime_C = generate_cpa_dataset(
            cipher_type=cipher_type_meta,
            key=key_meta,
            nr=nr,
            n=args.n,
            diff=diff_meta
        )
    except Exception as e:
        print(f"CPA attack data generation failed: {e}")
        return
    
    print("Starting CPA attack evaluation...")
    
    # Execute CPA attack evaluation
    try:
        accuracy, confidence_scores = evaluate_cpa_attack(
            model=model,
            A_prime_C=A_prime_C,
            B_prime_C=B_prime_C,
            config_manager=config_manager,
            device=device,
            batch_size=args.bs
        )
    except Exception as e:
        print(f"CPA attack evaluation failed: {e}")
        return
    
    print("=" * 60)
    print("CPA Attack Completed")
    print("=" * 60)
    
    return {
        'config_manager': config_manager,
        'metadata': metadata,
        'model': model,
        'nr': nr,
        'key': key_meta,
        'cipher_type': cipher_type_meta,
        'device': device,
        'n': args.n,
        'batch_size': args.bs,
        'accuracy': accuracy,
        'confidence_scores': confidence_scores,
    }


if __name__ == "__main__":
    result = main()
