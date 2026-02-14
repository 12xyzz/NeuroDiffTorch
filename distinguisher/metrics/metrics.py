import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any

class ConfusionMatrix:
    def __init__(self):
        pass
    
    def _convert_to_labels(self, y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if y_pred.shape[1] == 1:
            y_pred_proba = y_pred.flatten()
        else:
            y_pred_proba = y_pred[:, 1]
        
        return (y_pred_proba >= threshold).astype(int)
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        y_pred_labels = self._convert_to_labels(y_pred)
        
        cm = confusion_matrix(y_true, y_pred_labels)
        accuracy = accuracy_score(y_true, y_pred_labels)
        precision = precision_score(y_true, y_pred_labels, zero_division=0)
        recall = recall_score(y_true, y_pred_labels, zero_division=0)
        f1 = f1_score(y_true, y_pred_labels, zero_division=0)
        
        return {
            # 'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            # 'tn': cm[0, 0],  # true negative
            # 'fp': cm[0, 1],  # false positive
            # 'fn': cm[1, 0],  # false negative
            # 'tp': cm[1, 1],  # true positive
        }
    
    def format_output(self, metrics: Dict[str, Any], epoch: int = None) -> str:
        log_lines = []
        if epoch is not None:
            log_lines.append(f"Epoch {epoch} - Metrics:")
        
        log_lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
        log_lines.append(f"Precision: {metrics['precision']:.4f}")
        log_lines.append(f"Recall: {metrics['recall']:.4f}")
        log_lines.append(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return "\n".join(log_lines)