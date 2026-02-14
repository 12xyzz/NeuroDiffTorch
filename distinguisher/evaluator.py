import numpy as np
import torch
from torch.cuda.amp import autocast
from typing import Dict, Any, List
from .metrics import metric_registry

class Evaluator:
    def __init__(self, metric_type: str = None):
        self.metric_type = metric_type
        self._setup_metrics()
    
    def _setup_metrics(self):
        if self.metric_type is None:
            self.metrics = None
            return
            
        try:
            self.metrics = metric_registry.create_metric(self.metric_type)
        except Exception as e:
            available_metrics = metric_registry.list_metrics()
            raise ValueError(f"Unsupported metric type: {self.metric_type}. Available metrics: {available_metrics}")
    
    def evaluate_model(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                      device: torch.device, epoch: int = None, use_amp: bool = False) -> Dict[str, Any]:
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, Y_batch in data_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                
                if use_amp and device.type == 'cuda':
                    with autocast():
                        logits = model(X_batch)
                else:
                    logits = model(X_batch)
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(Y_batch.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        if self.metrics is not None:
            metrics = self.metrics.compute(all_labels, all_preds)
        else:
            accuracy = np.mean((all_preds.flatten() == all_labels.flatten()).astype(float))
            metrics = {'accuracy': accuracy}
        
        return metrics
    
    def format_evaluation_results(self, metrics: Dict[str, Any], epoch: int = None) -> str:
        if self.metrics is not None:
            return self.metrics.format_output(metrics, epoch)
        else:
            return f"Accuracy: {metrics.get('accuracy', 0.0):.4f}"
    
    def get_accuracy(self, metrics: Dict[str, Any]) -> float:
        return metrics.get('accuracy', 0.0)
    
    def log_evaluation(self, metrics: Dict[str, Any], logger, epoch: int = None):
        accuracy = self.get_accuracy(metrics)
        if epoch is not None:
            logger.log(f"Epoch {epoch} - Eval Acc: {accuracy:.4f}")
        
        if self.metrics is not None:
            logger.log(self.format_evaluation_results(metrics, epoch)) 