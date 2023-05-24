import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import torchvision

class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(L2CS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)

       # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        
        # gaze
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze




class ML2CS(nn.Module):
    def __init__(self):
        self.num_bins = 90
        # super(ML2CS, self).__init__()
        # # self.backbone = torchvision.models.mobilenet_v2().features
        # self.backbone = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        # # self.backbone = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')


        # # self.fc_yaw_gaze = nn.Linear(1280, self.num_bins)
        # # self.fc_pitch_gaze = nn.Linear(1280, self.num_bins)
        
        # self.fc_yaw_gaze = nn.Linear(1000, self.num_bins)
        # self.fc_pitch_gaze = nn.Linear(1000, self.num_bins)



    def forward(self, x):
        x = self.backbone(x)
        # # gaze
        # pre_yaw_gaze =  self.fc_yaw_gaze(x)
        # pre_pitch_gaze = self.fc_pitch_gaze(x)
        # return pre_yaw_gaze, pre_pitch_gaze

class ML2CS180(nn.Module):
    def __init__(self):
        self.num_bins = 180
        super(ML2CS180, self).__init__()
        # self.backbone = torchvision.models.mobilenet_v2().features
        # self.backbone = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        # mobilenet_v2 = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1', num_classes=self.num_bins)
        mobilenet_v2 = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.backbone = mobilenet_v2.features
        # Freeze weights
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        classifier_dict = mobilenet_v2.classifier.state_dict()
        classifier_dict["weight"] = classifier_dict["1.weight"]
        classifier_dict["bias"] = classifier_dict["1.bias"]
        # classifier_dict.remove("1.weight")
        # classifier_dict.remove("1.bias")

        self.fc_yaw_gaze = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(1280, self.num_bins)
        )

        self.fc_pitch_gaze = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(1280, self.num_bins)
        )

        try:
            self.fc_yaw_gaze.load_state_dict(classifier_dict)
            self.fc_pitch_gaze.load_state_dict(classifier_dict)
        except RuntimeError as e:
            print(f"IGNORING State dict errors")

        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        x = self.backbone(x)

        # straight from https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        # x = self.classifier(x)

        # gaze
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)

        yaw = self.softmax(pre_yaw_gaze)
        pitch = self.softmax(pre_pitch_gaze)

        return yaw, pitch

