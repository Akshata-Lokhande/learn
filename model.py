import torch
import torch as nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1 channel (grayscale MNIST), 28x28 image
        # Output: 10 classes (digits 0-9)

        # Convolutional Layer 1: 1 input channel, 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Max Pooling 1: 2x2 window
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max Pooling 2: 2x2 window
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate input features for the first fully connected layer
        # After two 2x2 max pools, image size becomes (28/2)/2 = 7x7
        # And we have 64 channels, so 64 * 7 * 7 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # Fully Connected Layer 1
        self.dropout = nn.Dropout(0.5) # Dropout for regularization
        self.fc2 = nn.Linear(128, 10) # Fully Connected Layer 2 (output 10 classes)

    def forward(self, x):
        # x is (batch_size, 1, 28, 28)
        x = self.pool1(F.relu(self.conv1(x))) # -> (batch_size, 32, 14, 14)
        x = self.pool2(F.relu(self.conv2(x))) # -> (batch_size, 64, 7, 7)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7) # -> (batch_size, 3136)

        x = F.relu(self.fc1(x))    # -> (batch_size, 128)
        x = self.dropout(x)        # Apply dropout
        x = self.fc2(x)            # -> (batch_size, 10)

        # Apply log_softmax for multi-class classification
        return F.log_softmax(x, dim=1)