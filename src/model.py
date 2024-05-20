import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Max pooling layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        # Max pooling layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layer 3
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        # Convolutional layer 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer 1
        self.fc1 = nn.Linear(2048, 512)

        # Fully connected layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        assert x.shape[1:] == torch.Size(
            [3, 32, 32]
        ), f"Input shape should be (3, 32, 32), recieved {tuple(x.shape[1:])}"

        # Apply first convolutional layer, ReLU activation and max pooling
        x = self.conv1(x)  # 32x32x32
        x = F.relu(x)
        x = self.pool1(x)  # 32x16x16

        # Apply second convolutional layer, ReLU activation and max pooling
        x = self.conv2(x)  # 64x16x16
        x = F.relu(x)
        x = self.pool2(x)  # 64x8x8

        # Apply third convolutional layer, ReLU activation and max pooling
        x = self.conv3(x)  # 128x8x8
        x = F.relu(x)
        x = self.pool3(x)  # 128x4x4

        # Flatten the tensor before passing to fully connected layers
        x = x.view(-1, 2048)

        # Apply first fully connected layer and ReLU activation
        x = self.fc1(x)  # 512
        # x = F.relu(x)

        # Apply second fully connected layer (output layer)
        x = self.fc2(x)  # 10

        return x


class ImageClassifierWithDropout(ImageClassifier):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        assert x.shape[1:] == torch.Size(
            [3, 32, 32]
        ), f"Input shape should be (3, 32, 32), recieved {tuple(x.shape[1:])}"

        # Apply first convolutional layer, ReLU activation and max pooling
        x = self.conv1(x)  # 32x32x32
        x = F.relu(x)
        x = self.pool1(x)  # 32x16x16

        # Apply second convolutional layer, ReLU activation and max pooling
        x = self.conv2(x)  # 64x16x16
        x = F.relu(x)
        x = self.pool2(x)  # 64x8x8

        # Apply third convolutional layer, ReLU activation and max pooling
        x = self.conv3(x)  # 128x8x8
        x = F.relu(x)
        x = self.pool3(x)  # 128x4x4

        # Flatten the tensor before passing to fully connected layers
        x = x.view(-1, 2048)

        # Apply first fully connected layer and ReLU activation
        x = self.fc1(x)  # 512
        x = F.relu(x)
        x = self.dropout1(x)

        # Apply second fully connected layer (output layer)
        x = self.fc2(x)  # 10
        x = self.dropout2(x)

        return x
