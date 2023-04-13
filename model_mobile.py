import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import torchvision


class ML2CS(nn.Module):
    def __init__(self, num_bins):
        super(L2CS, self).__init__()
        self.backbone = torchvision.models.mobilenet_v2().features

        self.fc_yaw_gaze = nn.Linear(1280, num_bins)
        self.fc_pitch_gaze = nn.Linear(1280, num_bins)


    def forward(self, x):
        x = self.backbone(x)


        # gaze
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze

