import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from t_net_transformation import Transformation, pointnetloss_mertix


class PointNetSeg(nn.Module):
    def __init__(self, num_class: int = 10):
        super().__init__()

        self.transformation = Transformation(
            global_feat=True, feature_transformation=True, xyz_axis=3
        )
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, num_class, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, input):
        bs = input.size()[0]
        n_pts = input.size()[2]

        x, matrix_3x3, matrix_kxk = self.transformation(input)

        # point features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(bs, n_pts, self.k)

        return x, matrix_3x3, matrix_kxk


class Loss:
    def pointnetloss(self, outputs, labels, matrix_3x3, matrix_kxk, alpha=0.0001):
        criterion = torch.nn.NLLLoss()
        bs = outputs.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
        id128x128 = torch.eye(128, requires_grad=True).repeat(bs, 1, 1)
        if outputs.is_cuda:
            id3x3 = id3x3.cuda()
            id128x128 = id128x128.cuda()
        diff3x3 = id3x3 - torch.bmm(matrix_3x3, matrix_3x3.transpose(1, 2))
        diff128x128 = id128x128 - torch.bmm(matrix_kxk, matrix_kxk.transpose(1, 2))
        return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff128x128)) / float(bs)
