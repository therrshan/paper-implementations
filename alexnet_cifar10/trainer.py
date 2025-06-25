import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_file='training_log.json'):
        self.log_file = log_file
        self.metrics = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'learning_rate': [],
            'batch_metrics': []
        }
        self.start_time = datetime.now()
    
    def log_epoch(self, epoch, train_loss, train_acc, test_loss, test_acc, learning_rate):
        """Log metrics for each epoch"""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(float(train_loss))
        self.metrics['train_acc'].append(float(train_acc))
        self.metrics['test_loss'].append(float(test_loss))
        self.metrics['test_acc'].append(float(test_acc))
        self.metrics['learning_rate'].append(float(learning_rate))
        
        # Save to file after each epoch
        self.save_metrics()
    
    def log_batch(self, epoch, batch_idx, total_batches, loss, accuracy):
        """Log metrics for each batch"""
        batch_metric = {
            'epoch': epoch,
            'batch': batch_idx,
            'total_batches': total_batches,
            'loss': float(loss),
            'accuracy': float(accuracy),
            'timestamp': datetime.now().isoformat()
        }
        self.metrics['batch_metrics'].append(batch_metric)
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_metrics(self):
        """Load metrics from JSON file"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.metrics = json.load(f)
            return True
        return False
    
    def get_training_summary(self):
        """Get summary statistics of training"""
        if not self.metrics['epochs']:
            return {}
        
        return {
            'total_epochs': len(self.metrics['epochs']),
            'best_train_acc': max(self.metrics['train_acc']),
            'best_test_acc': max(self.metrics['test_acc']),
            'final_train_loss': self.metrics['train_loss'][-1],
            'final_test_loss': self.metrics['test_loss'][-1],
            'final_train_acc': self.metrics['train_acc'][-1],
            'final_test_acc': self.metrics['test_acc'][-1],
            'training_time': str(datetime.now() - self.start_time)
        }

def train_epoch(model, trainloader, criterion, optimizer, device, logger=None, epoch=0):
    """Train for one epoch with comprehensive logging"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Log batch metrics if logger is provided
        if logger:
            batch_acc = 100. * correct / total
            logger.log_batch(epoch, batch_idx, len(trainloader), loss.item(), batch_acc)
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(trainloader)}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, testloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(testloader)
    accuracy = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename, logger=None):
    """Save model checkpoint with additional metadata"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add training summary if logger is available
    if logger:
        checkpoint['training_summary'] = logger.get_training_summary()
    
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    return epoch, loss, accuracy 