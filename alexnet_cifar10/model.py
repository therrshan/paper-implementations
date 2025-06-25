import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Layer 1: Conv1 -> ReLU -> MaxPool
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),  # 32x32x3 -> 32x32x96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),      # 32x32x96 -> 16x16x96
            
            # Layer 2: Conv2 -> ReLU -> MaxPool
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1), # 16x16x96 -> 16x16x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # 16x16x256 -> 8x8x256
            
            # Layer 3: Conv3 -> ReLU
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), # 8x8x256 -> 8x8x384
            nn.ReLU(inplace=True),
            
            # Layer 4: Conv4 -> ReLU
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1), # 8x8x384 -> 8x8x384
            nn.ReLU(inplace=True),
            
            # Layer 5: Conv5 -> ReLU -> MaxPool
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), # 8x8x384 -> 8x8x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),        # 8x8x256 -> 4x4x256
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) 