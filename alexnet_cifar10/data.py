import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=128, num_workers=2):
    """Load CIFAR-10 dataset with data augmentation"""
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers)
    
    return trainloader, testloader

def get_cifar10_classes():
    """Return CIFAR-10 class names"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'] 