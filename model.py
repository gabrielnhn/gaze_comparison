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


class VRI_GazeNet(nn.Module):
    
    def __init__(self, num_bins=180):
        self.num_bins = num_bins
        self.binwidth = int(360/self.num_bins)

        super(VRI_GazeNet, self).__init__()
        # mobilenet_v2 = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        mobilenet_v2 = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V2', dropout=0.3)
        self.backbone = mobilenet_v2.features
        
        classifier_dict = mobilenet_v2.classifier.state_dict()
        classifier_dict["weight"] = classifier_dict["1.weight"]
        classifier_dict["bias"] = classifier_dict["1.bias"]
        self.fc_yaw_gaze = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, self.num_bins)
        )

        self.fc_pitch_gaze = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, self.num_bins)
        )

        try:
            self.fc_yaw_gaze.load_state_dict(classifier_dict)
            self.fc_pitch_gaze.load_state_dict(classifier_dict)
        except RuntimeError as e:
            print(f"IGNORING State dict errors")

        self.softmax = nn.Softmax(dim=1)
        idx_tensor = [idx for idx in range(self.num_bins)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).cpu()
        

    def forward(self, x):
        x = self.backbone(x)
        # straight from https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        # gaze
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)

        yaw = self.softmax(pre_yaw_gaze)
        pitch = self.softmax(pre_pitch_gaze)

        return yaw, pitch


    def angles(self, images):
        y, p = self.forward(images)
        pitch_predicted_cpu = torch.sum(p * self.idx_tensor, 1).cpu().detach().numpy() * self.binwidth - 180
        yaw_predicted_cpu = torch.sum(y * self.idx_tensor, 1).cpu().detach().numpy() * self.binwidth - 180
        return list(zip(yaw_predicted_cpu, pitch_predicted_cpu))


class GazeLSTM(nn.Module):
    def __init__(self):
        super(GazeLSTM, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet18(pretrained=True)

        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim,bidirectional=True,num_layers=2,batch_first=True)

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(2*self.img_feature_dim, 3)


    def forward(self, input):

        base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))

        base_out = base_out.view(input.size(0),7,self.img_feature_dim)

        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:,3,:]
        output = self.last_layer(lstm_out).view(-1,3)


        angular_output = output[:,:2]
        angular_output[:,0:1] = math.pi*nn.Tanh()(angular_output[:,0:1])
        angular_output[:,1:2] = (math.pi/2)*nn.Tanh()(angular_output[:,1:2])

        var = math.pi*nn.Sigmoid()(output[:,2:3])
        var = var.view(-1,1).expand(var.size(0),2)

        return angular_output,var
class PinBallLoss(nn.Module):
    def __init__(self):
        super(PinBallLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1-self.q1

    def forward(self, output_o,target_o,var_o):
        q_10 = target_o-(output_o-var_o)
        q_90 = target_o-(output_o+var_o)

        loss_10 = torch.max(self.q1*q_10, (self.q1-1)*q_10)
        loss_90 = torch.max(self.q9*q_90, (self.q9-1)*q_90)


        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10+loss_90