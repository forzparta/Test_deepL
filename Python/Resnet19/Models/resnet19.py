import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.init as init

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        init.constant_(m.bias, 0)

# ResNet19_fc model
class Resnet19_fc(nn.Module):
    def __init__(self, num_classes):
        super(Resnet19_fc, self).__init__()
        self.epoch = 0
        self.resnet_base = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet_base.fc.in_features
        self.resnet_base.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs // 2, num_classes)
        )

    def forward(self, x):
        return self.resnet_base(x)

# ResNet19_conv model
class Resnet19_conv(nn.Module):
    def __init__(self, num_classes):
        super(Resnet19_conv, self).__init__()
        self.epoch = 0
        self.resnet_base = resnet18(weights=ResNet18_Weights.DEFAULT)
        additional_conv_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.resnet_base.layer4.add_module("additional_conv_layer", additional_conv_layer)
        num_ftrs = self.resnet_base.fc.in_features
        self.resnet_base.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet_base(x)

class Basic_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Basic_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu_out(out)
        return out

class Resnet19_snn(nn.Module):
    def __init__(self, num_classes):
        super(Resnet19_snn, self).__init__()
        self.epoch = 0

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.block1 = nn.Sequential(
            Basic_block(128, 128, stride=1),
            Basic_block(128, 128, stride=1),
            Basic_block(128, 128, stride=1)
        )

        self.block2 = nn.Sequential(
            Basic_block(128, 256, stride=2),
            Basic_block(256, 256, stride=1),
            Basic_block(256, 256, stride=1),
        )

        self.block3 = nn.Sequential(
            Basic_block(256, 512, stride=2),
            Basic_block(512, 512, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        
        return x


def select_model(model_type, num_classes, device):
    """
    Args:
    model_type (str): Type of the model ('fc', 'conv', or 'snn').
    num_classes (int): Number of classes for the final output layer.
    device: mps (mac), cuda, cpu.

    Returns:
    torch.nn.Module: The selected ResNet model.
    """
    if model_type == 'fc':
        model = Resnet19_fc(num_classes)
    elif model_type == 'conv':
        model = Resnet19_conv(num_classes)
    elif model_type == 'snn':
        model = Resnet19_snn(num_classes)
        model.apply(initialize_weights)
    else:
        raise ValueError("Model type not found")

    model.to(device)
    return model