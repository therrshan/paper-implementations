# AlexNet Implementation on CIFAR-10

This is a modular implementation of AlexNet for image classification on the CIFAR-10 dataset using PyTorch.

## 📋 Executive Summary

This project implements AlexNet, a groundbreaking deep convolutional neural network that won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. The implementation is adapted for CIFAR-10 and demonstrates the core architectural innovations that revolutionized computer vision.

**Key Metrics:**
- **Model Size**: ~60M parameters (~240MB)
- **Dataset**: CIFAR-10 (32x32 color images)
- **Classes**: 10 categories
- **Expected Performance**: ~70-80% test accuracy

## 📊 Training Results

After training, you can view detailed training curves and metrics in the generated HTML report.

**Performance Metrics:**
- **Total Epochs**: 50
- **Best Training Accuracy**: ~85%
- **Best Test Accuracy**: ~75%
- **Training Time**: ~30-60 minutes (CPU) / ~5-15 minutes (GPU)


## 🚀 Quick Start

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate 
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

To train the AlexNet model:

```bash
python main.py
```

This will:
- Download CIFAR-10 dataset automatically
- Train AlexNet for 50 epochs
- Save the model as `alexnet_cifar10.pth`
- Generate training curves plot as `training_curves.png`
- Save checkpoints every 10 epochs
- Create `training_log.json` with detailed metrics

### Generate Report

To create a detailed HTML report:

```bash
python generate_report.py
```

This generates `alexnet_report.html` with comprehensive analysis and visualizations.

### Inference

To test the trained model:

```bash
python inference.py
```

## 🔗 References

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25.
2. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images.
3. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
