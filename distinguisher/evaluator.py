import numpy as np
import torch
from typing import Dict, Any, List
from .metrics import metric_registry

class Evaluator:
    """Evaluator for managing model evaluation process"""
    
    def __init__(self, metric_type: str = None):
        """
        Initialize evaluator
        
        Args:
            metric_type: Evaluation metric type, if None then detailed metrics won't be calculated
        """
        self.metric_type = metric_type
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup evaluation metrics"""
        if self.metric_type is None:
            self.metrics = None
            return
            
        try:
            self.metrics = metric_registry.create_metric(self.metric_type)
        except Exception as e:
            available_metrics = metric_registry.list_metrics()
            raise ValueError(f"Unsupported metric type: {self.metric_type}. Available metrics: {available_metrics}")
    
    def evaluate_model(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                      device: torch.device, epoch: int = None, processor = None) -> Dict[str, Any]:
        """
        Evaluate model
        
        Args:
            model: Model to evaluate
            data_loader: Data loader
            device: Device
            epoch: Current epoch (optional)
            processor: Data processor (optional)
            
        Returns:
            Evaluation results dictionary
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, Y_batch in data_loader:
                # Use processor to process data (if provided)
                if processor is not None:
                    X_batch = processor.process(X_batch)
                
                logits = model(X_batch)
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(Y_batch.cpu().numpy())
        
        # Combine results from all batches
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # If detailed metrics are set, calculate them
        if self.metrics is not None:
            metrics = self.metrics.compute(all_labels, all_preds)
        else:
            # Only calculate accuracy
            accuracy = np.mean((all_preds.flatten() == all_labels.flatten()).astype(float))
            metrics = {'accuracy': accuracy}
        
        return metrics
    
    def format_evaluation_results(self, metrics: Dict[str, Any], epoch: int = None) -> str:
        """Format evaluation results"""
        if self.metrics is not None:
            return self.metrics.format_output(metrics, epoch)
        else:
            return f"Accuracy: {metrics.get('accuracy', 0.0):.4f}"
    
    def get_accuracy(self, metrics: Dict[str, Any]) -> float:
        """Get accuracy"""
        return metrics.get('accuracy', 0.0)
    
    def log_evaluation(self, metrics: Dict[str, Any], logger, epoch: int = None):
        """Log evaluation results"""
        accuracy = self.get_accuracy(metrics)
        if epoch is not None:
            logger.log(f"Epoch {epoch} - Eval Acc: {accuracy:.4f}")
        
        # If detailed metrics are set, log them
        if self.metrics is not None:
            logger.log(self.format_evaluation_results(metrics, epoch)) 