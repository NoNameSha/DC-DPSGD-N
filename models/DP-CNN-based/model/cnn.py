import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2),
                                    nn.Tanh(),
                                    nn.MaxPool2d(kernel_size=2, stride=1))

        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
                                    nn.Tanh(),
                                    nn.MaxPool2d(kernel_size=2, stride=1))

        self.fc = nn.Sequential(nn.Linear(4 * 4 * 32, 32),
                                nn.Tanh(),
                                nn.Linear(32, output_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LeNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LeNet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5),
                                    nn.Tanh(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5),
                                    nn.Tanh(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64*5*5, 384),
                                nn.Tanh(),
                                nn.Linear(384, 192),
                                nn.Tanh(),
                                nn.Linear(192, output_dim),
                                )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
