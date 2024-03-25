# './utils/resnet_34.py'
import torch
import torch.nn as nn

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# conv3x3 정의
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# 34-layer ResNet에서는 Bottleneck 블록 사용X
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0.0):
        super(BasicBlock, self).__init__()

        # Define the layers
        self.conv1 = conv3x3(inplanes, planes, stride) # inplanes -> planes
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes * self.expansion) # planes -> planes
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.downsample = downsample
        self.stride = stride
        
    # forward pass
    # (3x3 conv 2개) -> 기본 블록
    def forward(self, x):
        identity = x # ResNet은 나중에 자신을 더해줌

        out = self.conv1(x) # inplanes -> planes
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out) # planes -> planes
        out = self.bn2(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # 자신 더해주기
        out = self.relu(out)
        out = self.dropout(out)

        return out

class Resnet(nn.Module):
    
    def __init__(self, block, layers, num_classes, dropout_rate=0.0):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.dropout_rate = dropout_rate
        
        # BasicBlock 반복 전까지의 레이어
        # Define the layers referring to the ResNet
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) # channels: 3(RGB) -> 64 / same padding / 해상도는 1/2
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Follow the ResNet architecture and fill in the blanks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # self.avgpool = nn.AvgPool2d(4)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes) # self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Define the forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        #x = self.maxpool(x)

        # Define the Layer 1 ~ 4
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
    
def resnet34(num_classes, dropout_rate=0.0):
    return Resnet(BasicBlock, [3, 4, 6, 3], num_classes, dropout_rate)