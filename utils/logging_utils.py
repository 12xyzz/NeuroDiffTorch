import os
import time
from datetime import datetime, timedelta
from typing import Optional

class Logger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        self.start_time = None
        self.epoch_times = []
        self.total_epochs = None
    
    def log(self, message: str, print_to_console: bool = True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        
        if print_to_console:
            print(log_entry)
    
    def log_epoch(self, epoch: int, total_epochs: int, train_loss: float = None,
                  learning_rate: Optional[float] = None, train_accuracy: Optional[float] = None,
                  eval_accuracy: Optional[float] = None, time_info: Optional[dict] = None,
                  warmup_info: str = ""):
        message = f"Epoch {epoch}/{total_epochs}"
        if train_loss is not None:
            message += f" - Train Loss: {train_loss:.4f}"
        if learning_rate is not None:
            message += f", LR: {learning_rate:.6f}"
        if train_accuracy is not None:
            message += f", Train Acc: {train_accuracy:.4f}"
        if eval_accuracy is not None:
            message += f", Eval Acc: {eval_accuracy:.4f}"
        
        if warmup_info:
            message += warmup_info
        
        if time_info:
            avg_epoch_time = time_info['avg_epoch_time']
            remaining_time = time_info['estimated_remaining_time']
            completion_time = time_info['estimated_completion_time']
            
            avg_epoch_str = str(timedelta(seconds=int(avg_epoch_time)))
            remaining_str = str(timedelta(seconds=int(remaining_time)))
            completion_str = datetime.fromtimestamp(completion_time).strftime("%H:%M:%S")
            
            message += f" | Avg Epoch: {avg_epoch_str}, ETA: {remaining_str}"
        
        self.log(message)
    
    def log_epoch_warmup(self, epoch: int, total_epochs: int, train_loss: float = None,
                        learning_rate: Optional[float] = None, warmup_step: int = 0, 
                        total_warmup_steps: int = 0):
        message = f"Epoch {epoch}/{total_epochs}"
        if train_loss is not None:
            message += f" - Train Loss: {train_loss:.4f}"
        if learning_rate is not None:
            message += f", LR: {learning_rate:.6f}"

        message += f" [Warmup {warmup_step}/{total_warmup_steps}]"
        
        self.log(message)
    
    def log_checkpoint(self, checkpoint_path: str):
        self.log(f"Checkpoint saved: {checkpoint_path}")
    
    def start_training_timer(self, total_epochs: int):
        self.start_time = time.time()
        self.total_epochs = total_epochs
        self.epoch_times = []
        self.log(f"Started training tracking, total epochs: {total_epochs}")
    
    def record_epoch_time(self, epoch: int):
        if self.start_time is None:
            return
        
        current_time = time.time()
        epoch_time = current_time - self.start_time
        self.epoch_times.append(epoch_time)
        
        avg_epoch_time = epoch_time / len(self.epoch_times)
        remaining_epochs = self.total_epochs - epoch
        estimated_remaining_time = remaining_epochs * avg_epoch_time
        estimated_completion_time = current_time + estimated_remaining_time
        
        return {
            'epoch_time': epoch_time,
            'avg_epoch_time': avg_epoch_time,
            'remaining_epochs': remaining_epochs,
            'estimated_remaining_time': estimated_remaining_time,
            'estimated_completion_time': estimated_completion_time
        }
    
    def log_experiment_start(self, experiment_name: str):
        self.log("=" * 50)
        self.log(f"Starting experiment: {experiment_name}")
        self.log("=" * 50)
    
    def log_experiment_end(self, experiment_name: str):
        self.log("=" * 50)
        self.log(f"Experiment completed: {experiment_name}")
        self.log("=" * 50) 