# Training Configuration
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
NUM_WORKERS = 2

# Model Configuration
NUM_CLASSES = 10
DROPOUT_RATE = 0.5

# Data Configuration
DATA_ROOT = './data'
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# File paths
MODEL_SAVE_PATH = 'alexnet_cifar10.pth'
CHECKPOINT_PATH = 'checkpoint.pth'
TRAINING_CURVES_PATH = 'training_curves.png'

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck'] 