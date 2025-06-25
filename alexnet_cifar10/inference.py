import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from model import AlexNet
from utils import get_device
from config import *

def load_model(model_path):
    """Load a trained AlexNet model"""
    device = get_device()
    model = AlexNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def preprocess_image(image_path):
    """Preprocess a single image for inference"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict(model, image_tensor, device):
    """Make prediction on a single image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0]

def main():
    # Example usage
    model_path = MODEL_SAVE_PATH
    
    try:
        model, device = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        print(f"Model is on device: {device}")
        
        # Example: You can add your own image path here
        # image_path = "path/to/your/image.jpg"
        # image_tensor = preprocess_image(image_path)
        # predicted_class, confidence, probabilities = predict(model, image_tensor, device)
        # 
        # print(f"Predicted class: {CIFAR10_CLASSES[predicted_class]}")
        # print(f"Confidence: {confidence:.4f}")
        # print("All class probabilities:")
        # for i, (class_name, prob) in enumerate(zip(CIFAR10_CLASSES, probabilities)):
        #     print(f"  {class_name}: {prob:.4f}")
        
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main() 