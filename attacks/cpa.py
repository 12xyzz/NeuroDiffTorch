import sys
import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ConfigManager
from datasets import dataset_registry
from distinguisher import DataProcessor, model_registry


class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


def log_print(message, tee=None, indent=0):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    indent_str = " " * indent
    log_entry = f"[{timestamp}] {indent_str}{message}\n"
    
    if tee:
        tee.write(log_entry)
    else:
        print(log_entry, end='')


def load_config_from_experiment(experiment_path, tee=None):
    experiment_path = Path(experiment_path)
    yaml_files = list(experiment_path.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No yaml file found in {experiment_path}")
    yaml_path = yaml_files[0]
    log_print(f"Loading config from: {yaml_path}", tee)
    config_manager = ConfigManager(str(yaml_path))
    return config_manager


def load_metadata_from_data_path(data_path, tee=None):
    data_path = Path(data_path)
    metadata_path = data_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {metadata_path}")
    log_print(f"Metadata path: {metadata_path}", tee, indent=2)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def load_training_plaintexts(data_path, tee=None):
    data_path = Path(data_path)
    pt_path = data_path / "pt.npy"
    
    log_print(f"Training plaintexts path: {pt_path}", tee, indent=2)
    pt_array = np.load(pt_path)
    pt_set = set()
    if isinstance(pt_array, np.ndarray) and pt_array.ndim == 2:
        for pt_row in pt_array:
            pt_set.add(tuple(pt_row))
    else:
        for pt_item in pt_array:
            if isinstance(pt_item, (list, np.ndarray)):
                pt_set.add(tuple(pt_item))
            else:
                pt_set.add(pt_item)
    
    log_print(f"Loaded {len(pt_set)} training plaintexts", tee, indent=2)
    return pt_set


def load_model_from_checkpoint(config_manager, checkpoint_path, device, tee=None):
    model_config = config_manager.get_model_config()
    model_type = model_config['type']
    if 'params' not in model_config:
        raise ValueError(f"Configuration file missing model parameters")
    params = model_config['params'].copy()
    log_print(f"Creating model: {model_type}", tee)
    model = model_registry.get_model(model_type, **params)
    log_print(f"Loading checkpoint from: {checkpoint_path}", tee, indent=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def generate_cpa_dataset(cipher_type, key, nr, n, diff, data_path, tee=None):
    try:
        training_plaintexts = load_training_plaintexts(data_path, tee)
        
        dataset_class = dataset_registry.get_dataset_class(cipher_type)
        dataset = dataset_class(n=n, nr=nr, key_mode='input_fixed', key=key, diff=diff)
        
        if hasattr(dataset, 'generate_plaintext_triples'):
            # Step 1: Generate plaintext triples: A, A', B
            p1_list = []
            p1_prime_list = []
            p2_list = []
            max_attempts = n * 100
            attempts = 0
            while len(p1_list) < n and attempts < max_attempts:
                attempts += 1
                batch_size = min(1000, n - len(p1_list))
                p1_batch, p1_prime_batch, p2_batch = dataset.generate_plaintext_triples(batch_size, diff)
                if isinstance(p1_batch, np.ndarray):
                    p1_batch_tuples = [tuple(p1_batch[i]) for i in range(len(p1_batch))]
                    p1_prime_batch_tuples = [tuple(p1_prime_batch[i]) for i in range(len(p1_prime_batch))]
                    p2_batch_tuples = [tuple(p2_batch[i]) for i in range(len(p2_batch))]
                else:
                    p1_batch_tuples = p1_batch
                    p1_prime_batch_tuples = p1_prime_batch
                    p2_batch_tuples = p2_batch
                
                # Filter out pairs in training set
                valid_indices = []
                for i in range(len(p1_batch_tuples)):
                    p1 = p1_batch_tuples[i]
                    p2 = p2_batch_tuples[i]
                    if p1 not in training_plaintexts and p2 not in training_plaintexts:
                        valid_indices.append(i)
                for idx in valid_indices:
                    p1_list.append(p1_batch_tuples[idx])
                    p1_prime_list.append(p1_prime_batch_tuples[idx])
                    p2_list.append(p2_batch_tuples[idx])
            
            if len(p1_list) < n:
                raise ValueError(f"Could not generate {n} valid CPA pairs after {attempts} attempts. "
                               f"Only generated {len(p1_list)} pairs. "
                               f"This may indicate too many training plaintexts or insufficient randomness.")
            p1_list = p1_list[:n]
            p1_prime_list = p1_prime_list[:n]
            p2_list = p2_list[:n]
            log_print(f"Generated {n} CPA pairs, all values are not in training set", tee)
            
            # Step 2: Encrypt A,B to get C(A), C(B)
            c1_list, c2_list = dataset.encrypt_plaintext_pairs(p1_list, p2_list, nr)
            
            # Step 3: Apply diff to B to get B', then encrypt A', B'
            block_size = len(p2_list[0])
            sample0 = p2_list[0][0]
            try:
                plaintext_dtype = np.asarray(sample0).dtype
            except (TypeError, ValueError):
                plaintext_dtype = np.uint8
            
            if isinstance(diff, (list, tuple)):
                diff_arr = np.array(diff[:block_size], dtype=plaintext_dtype)
            else:
                diff_arr = np.unpackbits(
                    np.frombuffer(np.array([diff], dtype=np.uint64).tobytes(), dtype=np.uint8).reshape(1, 8),
                    axis=1
                )[0]

            p2_prime_list = []
            for p2 in p2_list:
                p2_prime_list.append(tuple(int(np.asarray(p2[i], dtype=plaintext_dtype) ^ diff_arr[i]) for i in range(block_size)))

            c1_prime_list, c2_prime_list = dataset.encrypt_plaintext_pairs(p1_prime_list, p2_prime_list, nr)
            
            c1_prime_arrays = [np.array([ct[i] for ct in c1_prime_list]) for i in range(block_size)]
            c2_prime_arrays = [np.array([ct[i] for ct in c2_prime_list]) for i in range(block_size)]
            c_arrays = [np.array([ct[i] for ct in c1_list]) for i in range(block_size)]
            A_prime_C = dataset.cipher.to_bits(c_arrays + c1_prime_arrays)
            B_prime_C = dataset.cipher.to_bits(c_arrays + c2_prime_arrays)
            
            if isinstance(A_prime_C, list):
                A_prime_C = np.column_stack(A_prime_C)
            if isinstance(B_prime_C, list):
                B_prime_C = np.column_stack(B_prime_C)
            
        else:
            raise ValueError(f"Dataset {cipher_type} does not support plaintext triples generation")
            
    except Exception as e:
        raise ValueError(f"Cannot create dataset or generate CPA data: {e}")
    
    return A_prime_C, B_prime_C


def evaluate_cpa(model, A_prime_C, B_prime_C, config_manager, device, batch_size=5000, tee=None):
    processor_config = config_manager.get('processor', {})
    processor = DataProcessor(processor_config)
    model.eval()

    correct_predictions = 0
    total_predictions = len(A_prime_C)
    all_probs_A = []
    all_probs_B = []

    with torch.no_grad():
        for i in range(0, total_predictions, batch_size):
            end_idx = min(i + batch_size, total_predictions)

            A_prime_C_batch = A_prime_C[i:end_idx]
            B_prime_C_batch = B_prime_C[i:end_idx]

            A_prime_C_tensor = torch.tensor(A_prime_C_batch, dtype=torch.float32, device=device)
            B_prime_C_tensor = torch.tensor(B_prime_C_batch, dtype=torch.float32, device=device)

            A_prime_C_processed = processor.process(A_prime_C_tensor)
            B_prime_C_processed = processor.process(B_prime_C_tensor)

            logits_A = model(A_prime_C_processed)
            logits_B = model(B_prime_C_processed)

            probs_A = torch.sigmoid(logits_A).cpu().numpy().flatten()
            probs_B = torch.sigmoid(logits_B).cpu().numpy().flatten()
            all_probs_A.append(probs_A)
            all_probs_B.append(probs_B)

            for j in range(end_idx - i):
                if probs_A[j] > probs_B[j]:
                    correct_predictions += 1

    probs_A_arr = np.concatenate(all_probs_A)
    probs_B_arr = np.concatenate(all_probs_B)
    per_sample_ties = np.sum(np.isclose(probs_A_arr, probs_B_arr, rtol=1e-5, atol=1e-5))
    all_tie = per_sample_ties == total_predictions

    if all_tie:
        log_print(f"Results: success rate = NaN ({per_sample_ties}/{total_predictions} ties, no discrimination)", tee)
    else:
        accuracy = correct_predictions / total_predictions
        log_print(f"Results: success rate = {accuracy:.4f} ({correct_predictions}/{total_predictions} correct)", tee)
    if per_sample_ties > 0:
        log_print(f"  ties: {per_sample_ties}/{total_predictions}", tee, indent=2)
    log_print(f"  prob_A: mean={probs_A_arr.mean():.4f}, std={probs_A_arr.std():.4f}, min={probs_A_arr.min():.4f}, max={probs_A_arr.max():.4f}", tee, indent=2)
    log_print(f"  prob_B: mean={probs_B_arr.mean():.4f}, std={probs_B_arr.std():.4f}, min={probs_B_arr.min():.4f}, max={probs_B_arr.max():.4f}", tee, indent=2)

    return np.nan if all_tie else (correct_predictions / total_predictions)


def parse_arguments():
    parser = argparse.ArgumentParser(description='CPA evaluation')
    parser.add_argument('--exp', type=str, required=True, help='Experiment folder path containing yaml config and checkpoint files')
    parser.add_argument('--nr', type=int, default=None, help='Number of encryption rounds, default from metadata nr')
    parser.add_argument('--ckpt', type=int, default=40, help='Checkpoint number, default checkpoint_40.pth')
    parser.add_argument('--n', type=int, default=1000000, help='Number of test samples, default 10^6')
    parser.add_argument('--bs', type=int, default=10000, help='Batch size, default 10000')
    parser.add_argument('--device', type=str, default=None, help='Device, default auto-detect')
    return parser.parse_args()


def main():
    args = parse_arguments()
    experiment_path = Path(args.exp)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = experiment_path / f"cpa_results_{timestamp_str}.log"
    tee = Tee(log_path)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        random_seed = np.random.randint(0, 2**31)
        np.random.seed(random_seed)
        
        log_print("=" * 60, tee)
        log_print(f"Starting CPA evaluation", tee)
        log_print("=" * 60, tee)
        log_print(f"Experiment path: {args.exp}", tee)
        log_print(f"Test sample count: {args.n}, Random seed: {random_seed}", tee)
        try:
            config_manager = load_config_from_experiment(args.exp, tee)
        except Exception as e:
            log_print(f"Configuration loading failed: {e}", tee, indent=2)
            return
        data_path = config_manager.get('data.path')
        log_print(f"Data path: {data_path}", tee, indent=2)
        try:
            metadata = load_metadata_from_data_path(data_path, tee)
        except Exception as e:
            log_print(f"Metadata loading failed: {e}", tee, indent=2)
            return
        nr_meta = metadata['nr']
        key_meta = metadata['key']
        cipher_type_meta = metadata['cipher_type']
        diff_meta = metadata['diff']
        
        log_print(f"Negative Samples: {metadata['neg']}", tee, indent=2)
        nr = args.nr if args.nr is not None else nr_meta
        checkpoint_path = experiment_path / f"checkpoint_{args.ckpt}.pth"
        
        if not checkpoint_path.exists():
            log_print(f"Checkpoint file does not exist: {checkpoint_path}", tee, indent=2)
            log_print("Available checkpoint files:", tee, indent=2)
            for checkpoint_file in experiment_path.glob("checkpoint_*.pth"):
                log_print(f"  - {checkpoint_file.name}", tee, indent=2)
            return
        if args.device is not None:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_print(f"Device: {device}, Batch size: {args.bs}", tee, indent=2)
        try:
            model = load_model_from_checkpoint(config_manager, checkpoint_path, device, tee)
        except Exception as e:
            log_print(f"Model loading failed: {e}", tee, indent=2)
            return
        total_params = sum(p.numel() for p in model.parameters())
        log_print(f"Model parameters: {total_params:,}", tee, indent=2)
        log_print("Starting CPA data generation...", tee)
        try:
            A_prime_C, B_prime_C = generate_cpa_dataset(
                cipher_type=cipher_type_meta,
                key=key_meta,
                nr=nr,
                n=args.n,
                diff=diff_meta,
                data_path=data_path,
                tee=tee
            )
        except Exception as e:
            log_print(f"CPA data generation failed: {e}", tee, indent=2)
            return
        try:
            accuracy = evaluate_cpa(
                model=model,
                A_prime_C=A_prime_C,
                B_prime_C=B_prime_C,
                config_manager=config_manager,
                device=device,
                batch_size=args.bs,
                tee=tee
            )
        except Exception as e:
            log_print(f"CPA evaluation failed: {e}", tee, indent=2)
            return
        log_print("=" * 60, tee)
        log_print(f"CPA results saved to: {log_path}", tee)
    
    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
