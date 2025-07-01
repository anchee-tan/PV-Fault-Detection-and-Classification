import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime
from PIL import ImageOps
import torch.nn.functional as F
import psutil
import pandas as pd
import gc
from pynvml import *
from torchsummary import summary
from sklearn.model_selection import train_test_split
import torch.utils.data as data

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------
# Benchmarking Utilities
# ----------------------------

def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """Calculate FLOPs for the model with better operator support"""
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    
    model.eval()
    device = next(model.parameters()).device
    inputs = torch.randn(input_size).to(device)
    
    # Calculate FLOPs
    flops = FlopCountAnalysis(model, inputs)
    
    # Print detailed table (optional)
    print("\nFLOPs Breakdown:")
    print(flop_count_table(flops))
    
    return flops.total()

def benchmark_model(model, test_loader, device, num_runs=100):
    """Enhanced benchmarking with more metrics"""
    model.eval()
    times = []
    flops = None
    
    try:
        # Calculate FLOPs first
        flops = calculate_flops(model)
    except Exception as e:
        print(f"\nFLOPs calculation warning: {str(e)}")
        flops = None
    
    # Create dummy input with same shape as actual data
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        # Benchmark
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    # Memory usage
    if device.type == 'cuda':
        torch.cuda.synchronize()
        mem_stats = {
            'allocated_mb': torch.cuda.memory_allocated(device) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(device) / 1024**2,
            'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024**2
        }
    else:
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_stats = {
            'rss_mb': mem_info.rss / 1024**2,
            'vms_mb': mem_info.vms / 1024**2
        }
    
    return {
        'avg_inference_time': np.mean(times),
        'fps': 1/np.mean(times),
        'std_dev': np.std(times),
        'flops': flops,
        'flops_g': flops / 1e9 if flops else None,
        **mem_stats
    }

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def count_parameters(model):
    """Count total number of trainable parameters in millions"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def get_memory_usage(device):
    """Get current memory usage in MB"""
    if device.type == 'cuda':
        torch.cuda.synchronize()
        return {
            'allocated_memory_mb': torch.cuda.memory_allocated(device) / 1024**2,
            'reserved_memory_mb': torch.cuda.memory_reserved(device) / 1024**2
        }
    else:
        process = psutil.Process()
        return {'ram_usage_mb': process.memory_info().rss / 1024**2}

def get_detailed_memory_usage(device):
    """Get detailed memory usage statistics"""
    if device.type == 'cuda':
        torch.cuda.synchronize()
        return {
            'allocated_memory_mb': torch.cuda.memory_allocated(device) / 1024**2,
            'reserved_memory_mb': torch.cuda.memory_reserved(device) / 1024**2,
            'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024**2,
            'max_reserved_mb': torch.cuda.max_memory_reserved(device) / 1024**2
        }
    else:
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'ram_usage_mb': mem_info.rss / 1024**2,
            'virtual_memory_mb': mem_info.vms / 1024**2
        }

# ----------------------------
# Performance Logger Class
# ----------------------------

class PerformanceLogger:
    def __init__(self, experiment_name="PV_Fault_Detection"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the base directory if it doesn't exist
        base_log_dir = "C:/Users/tanan/Downloads/anchee_fyp2/performance_logs"
        os.makedirs(base_log_dir, exist_ok=True)
        
        # Create the experiment-specific directory
        self.log_dir = os.path.join(base_log_dir, f"{experiment_name}_{self.timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize data structures
        self.epoch_metrics = []
        self.system_metrics = []
        self.final_metrics = {}
        self.confusion_matrix = None
        self.classification_report = None
        
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Log metrics for each epoch"""
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'learning_rate': float(lr),
            'timestamp': datetime.now().isoformat()
        }
        self.epoch_metrics.append(epoch_data)
        
    def log_system(self, cpu, ram, gpu=None, gpu_mem=None):
        """Log system metrics"""
        sys_data = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': float(cpu),
            'ram_usage': float(ram),
            'gpu_usage': float(gpu) if gpu is not None else None,
            'gpu_memory': float(gpu_mem) if gpu_mem is not None else None
        }
        self.system_metrics.append(sys_data)
        
    def log_final_metrics(self, metrics_dict):
        """Log final metrics"""
        self.final_metrics = metrics_dict
        
    def log_classification_report(self, report_dict):
        """Log classification report"""
        self.classification_report = report_dict
        
    def log_confusion_matrix(self, cm, class_names):
        """Log confusion matrix"""
        self.confusion_matrix = {
            'matrix': cm.tolist(),
            'class_names': class_names
        }
    
    def save_all(self):
        """Save all logged data to files"""
        try:
            # Ensure directory exists
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Save epoch metrics
            epoch_path = os.path.join(self.log_dir, "epoch_metrics.json")
            with open(epoch_path, 'w') as f:
                json.dump(self.epoch_metrics, f, indent=4)
                
            # Save system metrics
            system_path = os.path.join(self.log_dir, "system_metrics.json")
            with open(system_path, 'w') as f:
                json.dump(self.system_metrics, f, indent=4)
                
            # Save final metrics
            final_path = os.path.join(self.log_dir, "final_metrics.json")
            with open(final_path, 'w') as f:
                json.dump(self.final_metrics, f, indent=4)
                
            # Save classification report
            if self.classification_report:
                report_json_path = os.path.join(self.log_dir, "classification_report.json")
                with open(report_json_path, 'w') as f:
                    json.dump(self.classification_report, f, indent=4)
                
                report_txt_path = os.path.join(self.log_dir, "classification_report.txt")
                with open(report_txt_path, 'w') as f:
                    f.write(classification_report_to_text(self.classification_report))
                    
            # Save confusion matrix
            if self.confusion_matrix:
                cm_path = os.path.join(self.log_dir, "confusion_matrix.json")
                with open(cm_path, 'w') as f:
                    json.dump(self.confusion_matrix, f, indent=4)
                
                # Save confusion matrix plot
                plt.figure(figsize=(12, 10))
                sns.heatmap(self.confusion_matrix['matrix'], annot=True, fmt='d', cmap='Blues',
                            xticklabels=self.confusion_matrix['class_names'],
                            yticklabels=self.confusion_matrix['class_names'])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                plt.tight_layout()
                cm_plot_path = os.path.join(self.log_dir, "confusion_matrix.png")
                plt.savefig(cm_plot_path)
                plt.close()
                
            # Save training curves
            if len(self.epoch_metrics) > 0:
                self._save_training_curves()
                
            print(f"\nAll performance metrics saved to: {self.log_dir}")
            
        except Exception as e:
            print(f"\nError saving performance metrics: {str(e)}")
        
    def _save_training_curves(self):
        """Save training and validation curves"""
        epochs = [m['epoch'] for m in self.epoch_metrics]
        train_loss = [m['train_loss'] for m in self.epoch_metrics]
        val_loss = [m['val_loss'] for m in self.epoch_metrics]
        train_acc = [m['train_accuracy'] for m in self.epoch_metrics]
        val_acc = [m['val_accuracy'] for m in self.epoch_metrics]
        
        # Loss curve
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.log_dir}/loss_curve.png")
        plt.close()
        
        # Accuracy curve
        plt.figure(figsize=(12, 5))
        plt.plot(epochs, train_acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.log_dir}/accuracy_curve.png")
        plt.close()
        
        # Combined curve
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(epochs, train_loss, label='Training Loss')
        ax1.plot(epochs, val_loss, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, train_acc, label='Training Accuracy')
        ax2.plot(epochs, val_acc, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/training_curves.png")
        plt.close()

def classification_report_to_text(report_dict):
    """Convert classification report dictionary to text format"""
    text = "\nClassification Report:\n"
    text += f"{'':<15}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}\n"
    
    # Add metrics for each class
    for class_name in report_dict:
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        metrics = report_dict[class_name]
        text += f"{class_name:<15}{metrics['precision']:>10.2f}{metrics['recall']:>10.2f}{metrics['f1-score']:>10.2f}{metrics['support']:>10}\n"
    
    # Add overall accuracy
    if 'accuracy' in report_dict:
        text += f"\n{'accuracy':<15}{'':30}{report_dict['accuracy']:>10.2f}\n"
    
    # Add macro and weighted averages
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report_dict:
            metrics = report_dict[avg_type]
            text += f"{avg_type:<15}{metrics['precision']:>10.2f}{metrics['recall']:>10.2f}{metrics['f1-score']:>10.2f}{metrics['support']:>10}\n"
    
    return text

# ----------------------------
# System Monitoring Class
# ----------------------------

class SystemMonitor:
    def __init__(self):
        self.cpu_usages = []
        self.ram_usages = []
        self.gpu_usages = []
        self.gpu_memories = []
        self.timestamps = []
        self.gpu_available = False
        
        # Initialize GPU monitoring if available
        try:
            nvmlInit()
            self.gpu_available = True
            self.gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(nvmlDeviceGetCount())]
        except:
            self.gpu_available = False
        
    def record(self):
        """Record current system metrics"""
        # Record timestamp
        self.timestamps.append(time.time())
        
        # CPU and RAM usage
        self.cpu_usages.append(psutil.cpu_percent())
        self.ram_usages.append(psutil.virtual_memory().percent)
        
        # GPU usage if available
        if self.gpu_available:
            try:
                gpu_usage = nvmlDeviceGetUtilizationRates(self.gpu_handles[0]).gpu
                gpu_mem = nvmlDeviceGetMemoryInfo(self.gpu_handles[0]).used / nvmlDeviceGetMemoryInfo(self.gpu_handles[0]).total * 100
                self.gpu_usages.append(gpu_usage)
                self.gpu_memories.append(gpu_mem)
            except:
                self.gpu_usages.append(0)
                self.gpu_memories.append(0)
        else:
            self.gpu_usages.append(0)
            self.gpu_memories.append(0)
    
    def get_summary(self):
        """Get summary statistics of system metrics"""
        summary = {
            'cpu': {
                'mean': float(np.mean(self.cpu_usages)),
                'max': float(np.max(self.cpu_usages)),
                'min': float(np.min(self.cpu_usages)),
                'std': float(np.std(self.cpu_usages))
            },
            'ram': {
                'mean': float(np.mean(self.ram_usages)),
                'max': float(np.max(self.ram_usages)),
                'min': float(np.min(self.ram_usages)),
                'std': float(np.std(self.ram_usages))
            }
        }
        
        if any(usage > 0 for usage in self.gpu_usages):
            summary['gpu'] = {
                'usage_mean': float(np.mean(self.gpu_usages)),
                'usage_max': float(np.max(self.gpu_usages)),
                'usage_min': float(np.min(self.gpu_usages)),
                'usage_std': float(np.std(self.gpu_usages)),
                'memory_mean': float(np.mean(self.gpu_memories)),
                'memory_max': float(np.max(self.gpu_memories)),
                'memory_min': float(np.min(self.gpu_memories)),
                'memory_std': float(np.std(self.gpu_memories))
            }
        
        return summary
    
    def save_metrics_plot(self, save_path):
        """Save system metrics visualization"""
        plt.figure(figsize=(15, 10))
        
        # Calculate relative timestamps
        if len(self.timestamps) > 0:
            relative_times = [t - self.timestamps[0] for t in self.timestamps]
        else:
            relative_times = []
        
        # CPU Usage
        plt.subplot(2, 2, 1)
        plt.plot(relative_times, self.cpu_usages, label='CPU Usage')
        plt.title('CPU Usage During Training')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        plt.grid(True)
        
        # RAM Usage
        plt.subplot(2, 2, 2)
        plt.plot(relative_times, self.ram_usages, label='RAM Usage')
        plt.title('RAM Usage During Training')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        plt.grid(True)
        
        # GPU Usage (if available)
        if any(usage > 0 for usage in self.gpu_usages):
            plt.subplot(2, 2, 3)
            plt.plot(relative_times, self.gpu_usages, label='GPU Usage')
            plt.title('GPU Usage During Training')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Percentage (%)')
            plt.ylim(0, 100)
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.plot(relative_times, self.gpu_memories, label='GPU Memory')
            plt.title('GPU Memory Usage During Training')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Percentage (%)')
            plt.ylim(0, 100)
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# ----------------------------
# Data Preparation Functions
# ----------------------------

def preparing_data():
    """Prepare and load the dataset with correct train/validation/test split (80/10/10)"""
    # Define data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Set the full data directory
    data_dir = "C:/Users/tanan/Downloads/anchee_fyp2/PVF_10/PVF_10_Processed"

    # Create dataset from the full directory with no transform
    print("\nLoading full dataset...")
    full_dataset = PVFaultDataset(data_dir, transform=None)

    # First split into train and temp (80/20)
    train_indices, temp_indices = train_test_split(
        list(range(len(full_dataset))), 
        test_size=0.2,
        stratify=[full_dataset.labels[i] for i in range(len(full_dataset))],
        random_state=42
    )
    
    # Further split temp into validation and test (10/10 of the original dataset)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        stratify=[full_dataset.labels[i] for i in temp_indices],
        random_state=42
    )

    # Create datasets with transforms
    train_dataset = PVFaultDataset(data_dir, transform=data_transforms['train'])
    train_dataset.img_paths = [full_dataset.img_paths[i] for i in train_indices]
    train_dataset.labels = [full_dataset.labels[i] for i in train_indices]

    val_dataset = PVFaultDataset(data_dir, transform=data_transforms['val_test'])
    val_dataset.img_paths = [full_dataset.img_paths[i] for i in val_indices]
    val_dataset.labels = [full_dataset.labels[i] for i in val_indices]

    test_dataset = PVFaultDataset(data_dir, transform=data_transforms['val_test'])
    test_dataset.img_paths = [full_dataset.img_paths[i] for i in test_indices]
    test_dataset.labels = [full_dataset.labels[i] for i in test_indices]

    # Print dataset sizes
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True)

    # Get class information
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class names: {class_names}")

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, class_names, num_classes

# Dataset class
class PVFaultDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.img_paths = []
        self.labels = []
        
        # Load all image paths and their labels
        for class_name in self.classes:
            class_dir = os.path.join(img_dir, class_name)
            class_count = 0
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    self.img_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
                    class_count += 1
            print(f"Class {class_name}: {class_count} images")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transform here consistently
        if self.transform:
            image = self.transform(image)
        else:
            # Ensure tensor conversion even without transform
            image = transforms.ToTensor()(image)
            
        return image, label

# ----------------------------
# Visualization Functions
# ----------------------------

def visualize_samples(dataset, num_samples=5):
    """Visualize sample images from the dataset"""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Get class names from the original dataset if it's a TransformDataset
    if hasattr(dataset, 'subset') and hasattr(dataset.subset.dataset, 'classes'):
        classes = dataset.subset.dataset.classes
    elif hasattr(dataset, 'classes'):
        classes = dataset.classes
    else:
        # Fallback if classes can't be found
        classes = [f"Class {i}" for i in range(10)]  # Generic class names
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {classes[label]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_results(train_losses, train_accuracies, val_losses, val_accuracies, all_preds, all_labels, class_names):
    """Visualize training results and metrics"""
    # Figure 1: Training and Validation Loss
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Figure 2: Training and Validation Accuracy
    plt.figure(figsize=(12, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Figure 3: Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def visualize_computational_metrics(monitor, total_time, train_size, test_size):
    """Visualize computational efficiency metrics"""
    plt.figure(figsize=(15, 10))
    
    # Calculate relative timestamps
    if len(monitor.timestamps) > 0:
        relative_times = [t - monitor.timestamps[0] for t in monitor.timestamps]
    else:
        relative_times = []
    
    # CPU Usage
    plt.subplot(2, 2, 1)
    plt.plot(relative_times, monitor.cpu_usages, label='CPU Usage')
    plt.title('CPU Usage During Training')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    
    # RAM Usage
    plt.subplot(2, 2, 2)
    plt.plot(relative_times, monitor.ram_usages, label='RAM Usage')
    plt.title('RAM Usage During Training')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    
    # GPU Usage (if available)
    if any(usage > 0 for usage in monitor.gpu_usages):
        plt.subplot(2, 2, 3)
        plt.plot(relative_times, monitor.gpu_usages, label='GPU Usage')
        plt.title('GPU Usage During Training')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(relative_times, monitor.gpu_memories, label='GPU Memory')
        plt.title('GPU Memory Usage During Training')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nComputational Efficiency Metrics:")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Training Set Size: {train_size} samples")
    print(f"Test Set Size: {test_size} samples")
    print(f"Average CPU Usage: {np.mean(monitor.cpu_usages):.2f}%")
    print(f"Average RAM Usage: {np.mean(monitor.ram_usages):.2f}%")
    
    if any(usage > 0 for usage in monitor.gpu_usages):
        print(f"Average GPU Usage: {np.mean(monitor.gpu_usages):.2f}%")
        print(f"Average GPU Memory Usage: {np.mean(monitor.gpu_memories):.2f}%")

def visualize_class_distribution(train_dataset, val_dataset, test_dataset, class_names):
    """Visualize class distribution across train, val, and test sets in a single bar chart"""
    # Get counts for each dataset
    train_counts = np.zeros(len(class_names))
    val_counts = np.zeros(len(class_names))
    test_counts = np.zeros(len(class_names))
    
    # Count samples in train set
    for _, label in train_dataset:
        train_counts[label] += 1
    
    # Count samples in validation set
    for _, label in val_dataset:
        val_counts[label] += 1
    
    # Count samples in test set
    for _, label in test_dataset:
        test_counts[label] += 1
    
    # Create figure
    plt.figure(figsize=(15, 8))
    bar_width = 0.25
    index = np.arange(len(class_names))
    
    # Plot bars for each dataset
    plt.bar(index, train_counts, bar_width, label='Train')
    plt.bar(index + bar_width, val_counts, bar_width, label='Validation')
    plt.bar(index + 2*bar_width, test_counts, bar_width, label='Test')
    
    # Customize plot
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution Across Datasets')
    plt.xticks(index + bar_width, class_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Add exact numbers on top of bars
    for i in range(len(class_names)):
        plt.text(i, train_counts[i] + 2, str(int(train_counts[i])), ha='center')
        plt.text(i + bar_width, val_counts[i] + 2, str(int(val_counts[i])), ha='center')
        plt.text(i + 2*bar_width, test_counts[i] + 2, str(int(test_counts[i])), ha='center')
    
    plt.show()

# ----------------------------
# Model Functions
# ----------------------------

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        # Enhanced initial block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Enhanced residual blocks with skip connections
        self.res_block1 = self._make_residual_block(32, 64)
        self.res_block2 = self._make_residual_block(64, 128)
        self.res_block3 = self._make_residual_block(128, 256)
        
        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Enhanced classifier with more capacity
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Enhanced attention
        attention = self.attention(x)
        x = x * attention
        
        return self.classifier(x)
        
def create_customcnn_model(num_classes):
    """Create a custom CNN model from scratch"""
    print("\nCreating custom CNN model with architecture:")
    print("-------------------------------------------")
    print("Input: 3x224x224 RGB image")
    print("5 convolutional blocks with increasing filters (32, 64, 128, 256, 512)")
    print("Each block contains: Conv2d -> BatchNorm -> ReLU -> MaxPool")
    print("Classifier head with 3 fully connected layers (1024, 512, num_classes)")
    print("Dropout layers for regularization")
    
    model = CustomCNN(num_classes)
    
    # Print model summary
    print("\nModel Summary:")
    print("-------------")
    print(model)
    
    return model

def report_model_params(model, model_name="CustomCNN"):
    """Analyzes and reports model's trainable vs total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n{model_name} Parameter Analysis:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params:.2%})")
    
    return {
        "model": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_percentage": trainable_params/total_params
    }

def initialize_model(num_classes, device):
    """Initialize the model and move it to the specified device"""
    model_name = CustomCNN
    print(f"\nInitializing {model_name}...")
    
    model = create_customcnn_model(num_classes)
    
    # Report parameter statistics
    param_stats = report_model_params(model, model_name)
    
    model = model.to(device)
    return model, param_stats

# ----------------------------
# Training Utilities
# ----------------------------

def setup_optimizer(model, optimizer_type='Adagrad'):
    """Enhanced setup with default hyperparameters"""
    
    # Optimizer setup with all default hyperparameters explicitly shown
    optimizers = {
        'Adam': optim.Adam(
            model.parameters(), 
            lr=0.001, 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=0, 
            amsgrad=False,
            foreach=None, 
            maximize=False, 
            capturable=False, 
            differentiable=False, 
            fused=None
        ),
        'SGD': optim.SGD(
            model.parameters(), 
            lr=0.01,  # Note: SGD typically needs higher LR
            momentum=0, 
            dampening=0, 
            weight_decay=0, 
            nesterov=False,
            maximize=False, 
            foreach=None, 
            differentiable=False, 
            fused=None
        ),
        'AdamW': optim.AdamW(
            model.parameters(), 
            lr=0.001, 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=0.01,  # AdamW default includes weight decay
            amsgrad=False,
            maximize=False, 
            foreach=None, 
            capturable=False, 
            differentiable=False, 
            fused=None
        ),
        'RMSprop': optim.RMSprop(
            model.parameters(), 
            lr=0.01, 
            alpha=0.99, 
            eps=1e-08, 
            weight_decay=0, 
            momentum=0, 
            centered=False, 
            capturable=False, 
            foreach=None, 
            maximize=False, 
            differentiable=False
        ),
        'Adagrad': optim.Adagrad(
            model.parameters(), 
            lr=0.01,  # Adagrad default is higher
            lr_decay=0, 
            weight_decay=0, 
            initial_accumulator_value=0, 
            eps=1e-10,
            foreach=None,
            maximize=False, 
            differentiable=False, 
            fused=None
        ),
        'Nadam': optim.NAdam(
            model.parameters(), 
            lr=0.002,  # NAdam default is slightly higher
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=0, 
            momentum_decay=0.004,
            foreach=None, 
            maximize=False, 
            capturable=False, 
            differentiable=False
        ),
    }
    
    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Available: {list(optimizers.keys())}")
    
    optimizer = optimizers[optimizer_type]
    criterion = nn.CrossEntropyLoss()

    return optimizer, criterion

def evaluate_model(model, test_loader, criterion=None, device='cuda'):
    """Evaluate the model on test data"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Create criterion if not provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    return test_loss, test_acc, all_preds, all_labels
    
# ----------------------------
# Main Training Function
# ----------------------------

def training_model(train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, num_classes, device, class_names, optimizer_type='Adagrad'):
    """Main training pipeline with comprehensive performance tracking"""
    # Initialize performance logger and system monitor
    logger = PerformanceLogger(experiment_name=f"CustomCNN_{optimizer_type}")
    monitor = SystemMonitor()
    
    # Initialize model
    print(f"\nTraining with {optimizer_type} optimizer")
    print("\nModel Summary")
    model = create_customcnn_model(num_classes).to(device)
    try:
        summary(model, input_size=(3, 224, 224))
    except:
        print("Could not generate model summary")

    # Track total training time
    total_start_time = time.time()
    
    # Setup optimizer, scheduler and criterion (without class weights)
    optimizer, criterion = setup_optimizer(model, optimizer_type)
    
    # Training parameters
    num_epochs = 50
    
    # Training loop
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    def log_system_metrics():
        """Helper function to record and log system metrics"""
        try:
            # Get CPU and RAM usage
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            
            # Get GPU metrics if available
            gpu = None
            gpu_mem = None
            if torch.cuda.is_available() and monitor.gpu_available:
                try:
                    gpu = nvmlDeviceGetUtilizationRates(monitor.gpu_handles[0]).gpu
                    gpu_mem = nvmlDeviceGetMemoryInfo(monitor.gpu_handles[0]).used / nvmlDeviceGetMemoryInfo(monitor.gpu_handles[0]).total * 100
                except Exception as e:
                    print(f"GPU metric error: {str(e)}")
            
            # Record and log the metrics
            monitor.record()
            logger.log_system(cpu, ram, gpu, gpu_mem)
                
        except Exception as e:
            print(f"Error logging system metrics: {str(e)}")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Record initial system metrics
        log_system_metrics()
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (inputs, labels) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
                
                # Record system stats every 10 batches
                if batch_idx % 10 == 0:
                    log_system_metrics()
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tepoch.set_postfix(loss=loss.item(), accuracy=correct/total)
        
        # Record final metrics for the epoch
        log_system_metrics()
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")
        
        # Validation
        val_loss, val_acc, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        # Log epoch metrics
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, epoch_loss, epoch_acc, val_loss, val_acc, current_lr)
    
    total_training_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_training_time:.2f} seconds")
    
    # Save the final model
    final_model_path = f"C:/Users/tanan/Downloads/anchee_fyp2/customcnn_pv_fault_final.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, final_model_path)
    print(f"Final model saved at epoch {num_epochs}")
    
    # Final evaluation on test set
    test_loss, test_acc, test_preds, test_labels = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}")
    
    # Generate classification report and confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    report_dict = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
    
    logger.log_confusion_matrix(cm, class_names)
    logger.log_classification_report(report_dict)
    
    # Initialize final metrics
    final_metrics = {
        'training_metrics': {
            'total_training_time_s': total_training_time,
            'final_epoch': num_epochs,
            'final_val_loss': val_loss,
            'final_val_accuracy_percent': val_acc * 100,
            'final_train_loss': train_losses[-1],
            'final_train_accuracy_percent': train_accuracies[-1] * 100,
            'test_loss': test_loss,
            'test_accuracy_percent': test_acc * 100,
        },
        'system_metrics': monitor.get_summary(),
        'dataset_metrics': {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'num_classes': num_classes
        },
        'training_parameters': {
            'batch_size': train_loader.batch_size,
            'num_epochs': num_epochs,
            'optimizer': str(optimizer.__class__.__name__),
            'criterion': str(criterion.__class__.__name__),
            'scheduler': 'None'        
            }
    }
    
    # Add model metrics if benchmarking succeeds
    try:
        benchmark_results = benchmark_model(model, test_loader, device)
        model_size_mb = get_model_size(model)
        model_params_m = count_parameters(model)
        
        final_metrics['model_metrics'] = {
            'model_size_mb': model_size_mb,
            'total_parameters_m': model_params_m,
            'inference_time_ms': benchmark_results['avg_inference_time'] * 1000,
            'inference_fps': benchmark_results['fps'],
            'inference_std_dev_ms': benchmark_results['std_dev'] * 1000,
            **get_detailed_memory_usage(device)
        }
        
        if benchmark_results['flops'] is not None:
            final_metrics['model_metrics']['flops'] = benchmark_results['flops']
            final_metrics['model_metrics']['gigaflops'] = benchmark_results['flops_g']
    except Exception as e:
        print(f"Benchmarking failed: {str(e)}")
        # Still include basic model metrics
        final_metrics['model_metrics'] = {
            'model_size_mb': get_model_size(model),
            'total_parameters_m': count_parameters(model),
            'benchmark_error': str(e)
        }
    
    # Log final metrics
    logger.log_final_metrics(final_metrics)
    
    # Save all data
    logger.save_all()
    monitor.save_metrics_plot(f"{logger.log_dir}/system_metrics.png")
    
    print("Training complete. Final model saved and evaluated.")
    
    return model, train_losses, train_accuracies, val_losses, val_accuracies, test_preds, test_labels, monitor

# ----------------------------
# Prediction Function
# ----------------------------

def predict_single_image(model, image_path, transform, class_names, device='cuda'):
    """Make prediction on a single image"""
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        top_p, top_class = torch.topk(probabilities, 3)
    
    # Display image with prediction
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"Prediction: {class_names[top_class[0]]}\nProbability: {top_p[0]:.4f}")
    plt.axis('off')
    plt.show()
    
    # Print top 3 predictions
    print("Top 3 predictions:")
    for i in range(3):
        print(f"{class_names[top_class[i]]}: {top_p[i]:.4f}")
    
    return class_names[top_class[0]], top_p[0].item()

# ----------------------------
# Main Execution Flow
# ----------------------------

def main_flow():
    """Main execution flow of the program"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print system info
    print("\nSystem Information:")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("No GPU available, using CPU")
    
    # 1. Prepare data with 80/10/10 split
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, class_names, num_classes = preparing_data()
    
    # 2. Visualize samples
    print("\nVisualizing training samples:")
    visualize_samples(train_dataset)
    
    print("\nVisualizing class distribution across datasets:")
    visualize_class_distribution(train_dataset, val_dataset, test_dataset, class_names)
    
    # 3. Train model with enhanced performance tracking
    print("\nStarting model training with comprehensive performance tracking...")
    start_time = time.time()
    
    # Fixed function call - match the function signature
    model, train_losses, train_accuracies, val_losses, val_accuracies, test_preds, test_labels, monitor = training_model(
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, num_classes, device, class_names, optimizer_type='Adagrad'
    )
    total_time = time.time() - start_time
    
    # 4. Visualize results using the test set predictions
    print("\nVisualizing results from the test set...")
    visualize_results(train_losses, train_accuracies, val_losses, val_accuracies, test_preds, test_labels, class_names)
    
    # 5. Visualize computational metrics
    print("\nVisualizing computational metrics...")
    visualize_computational_metrics(monitor, total_time, len(train_dataset), len(test_dataset))
    
    # 6. Save final model with timing information
    print("\nSaving final model with performance metrics...")
    model_dir = "C:/Users/tanan/Downloads/anchee_fyp2"
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = os.path.join(model_dir, "customcnn_pv_fault_final.pth")
    
    # Get test accuracy from the training results
    test_loss, test_acc, _, _ = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_time': total_time,
        'best_val_loss': min(val_losses),
        'best_val_acc': max(val_accuracies),
        'test_loss': test_loss,
        'test_acc': test_acc,
        'cpu_usage': np.mean(monitor.cpu_usages),
        'ram_usage': np.mean(monitor.ram_usages),
        'gpu_usage': np.mean(monitor.gpu_usages) if any(usage > 0 for usage in monitor.gpu_usages) else 0,
        'gpu_memory': np.mean(monitor.gpu_memories) if any(mem > 0 for mem in monitor.gpu_memories) else 0
    }, final_model_path)
    
    print(f"Final model saved with metrics: {final_model_path}")
    print(f"Test accuracy: {test_acc:.4f}")

    print(f"Saved model size: {os.path.getsize(final_model_path) / (1024**2):.2f} MB")

    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main_flow()