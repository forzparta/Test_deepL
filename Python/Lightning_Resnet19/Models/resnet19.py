'''Module that contains all model present in experiment'''

from torch import nn
from torch.nn import init
from torchvision.models import resnet18, ResNet18_Weights


def initialize_weights(m: nn.Module) -> None:
    '''Function to initialize model parameters'''
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


class Resnet19Fc(nn.Module):
    '''Resnet 18 model with additional fc layer added at the end'''

    def __init__(self, num_classes: int) -> nn.Module:
        super().__init__()
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


class Resnet19Conv(nn.Module):
    '''Resnet 18 model with additional (conv-norm-ReLU) layer added at the end before fc'''

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.epoch = 0
        self.resnet_base = resnet18(weights=ResNet18_Weights.DEFAULT)
        additional_conv_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.resnet_base.layer4.add_module(
            "additional_conv_layer", additional_conv_layer)
        num_ftrs = self.resnet_base.fc.in_features
        self.resnet_base.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet_base(x)


class BasicBlock(nn.Module):
    '''Resnet 18 model layer from paper'''

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, padding=0),
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


class Resnet19Snn(nn.Module):
    '''Resnet 19 structure like https://arxiv.org/pdf/2011.05280.pdf but classic residual'''

    def __init__(self, num_classes: int):
        super(Resnet19Snn, self).__init__()
        self.epoch = 0

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.block1 = nn.Sequential(
            BasicBlock(128, 128, stride=1),
            BasicBlock(128, 128, stride=1),
            BasicBlock(128, 128, stride=1)
        )

        self.block2 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
            BasicBlock(256, 256, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.fc2(out)

        return out


class BasicBlockDropOut(nn.Module):
    '''Resnet 18 model layer from paper'''

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu_out(out)
        return out


class Resnet19SnnDropOut(nn.Module):
    '''Resnet 19 structure like https://arxiv.org/pdf/2011.05280.pdf but classic residual'''

    def __init__(self, num_classes: int):
        super().__init__()
        self.epoch = 0

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.block1 = nn.Sequential(
            BasicBlock(128, 128, stride=1),
            BasicBlockDropOut(128, 128, stride=1),
            BasicBlockDropOut(128, 128, stride=1)
        )

        self.block2 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlockDropOut(256, 256, stride=1),
            BasicBlockDropOut(256, 256, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlockDropOut(512, 512, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


def select_model(model_type: str, num_classes: int) -> nn.Module:
    """
    Args:
    model_type (str): Type of the model ('fc', 'conv', or 'snn').
    num_classes (int): Number of classes for the final output layer.
    device: mps (mac), cuda, cpu.

    Returns:
    torch.nn.Module: The selected ResNet model.
    """
    if model_type == 'fc':
        model = Resnet19Fc(num_classes)
    elif model_type == 'conv':
        model = Resnet19Conv(num_classes)
    elif model_type == 'snn':
        model = Resnet19Snn(num_classes)
        model.apply(initialize_weights)
    else:
        raise ValueError("Model type not found")

    return model
