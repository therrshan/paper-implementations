import torch
import torch.nn as nn
import torch.optim as optim

# Import our modules
from model import AlexNet
from data import get_data_loaders
from trainer import train_epoch, evaluate, save_checkpoint, TrainingLogger
from utils import plot_training_curves, get_device, print_model_summary
from config import *

def main():
    # Set device
    device = get_device()
    print(f'Using device: {device}')
    
    # Initialize training logger
    logger = TrainingLogger(log_file='training_log.json')
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = get_data_loaders(BATCH_SIZE, NUM_WORKERS)
    
    # Initialize model
    model = AlexNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Print model summary
    print_model_summary(model)
    
    # Training loop
    print("Starting training...")
    print(f"Training for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("-" * 50)
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device, 
            logger=logger, epoch=epoch+1
        )
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        # Log epoch metrics
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch+1, train_loss, train_acc, test_loss, test_acc, current_lr)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch+1, train_loss, test_acc, 
                          CHECKPOINT_PATH, logger=logger)
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Save the final model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved as {MODEL_SAVE_PATH}")
    
    # Print training summary
    summary = logger.get_training_summary()
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total epochs: {summary['total_epochs']}")
    print(f"Best training accuracy: {summary['best_train_acc']:.2f}%")
    print(f"Best test accuracy: {summary['best_test_acc']:.2f}%")
    print(f"Final training accuracy: {summary['final_train_acc']:.2f}%")
    print(f"Final test accuracy: {summary['final_test_acc']:.2f}%")
    print(f"Training time: {summary['training_time']}")
    print("="*50)
    
    # Plot training curves using saved metrics
    if logger.metrics['epochs']:
        plot_training_curves(
            logger.metrics['train_loss'], 
            logger.metrics['train_acc'], 
            logger.metrics['test_acc'], 
            TRAINING_CURVES_PATH
        )
        print(f"Training curves saved as {TRAINING_CURVES_PATH}")
    
    print("\nTraining completed! You can now generate a detailed report using:")
    print("python generate_report.py")

if __name__ == "__main__":
    main() 