import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_training_curves(train_losses, train_accuracies, test_accuracies, save_path='training_curves.png'):
    """Plot training curves"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def get_device():
    """Get the best available device (CUDA or CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    """Print a summary of the model architecture and parameters"""
    print("Model Architecture:")
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Calculate model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_all_mb:.2f} MB") 