import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class TNet3D(nn.Module):
    """T-net is a special Neural Network that learns a transformation matrix
    that will rotate the input point cloud to a consistent orientation.

    This class is for TNet 3D (xyz)
    """

    def __init__(self, xyz_axis: int = 3):
        super().__init__()
        self.xyz_axis = xyz_axis

        self.conv1 = torch.nn.Conv1d(xyz_axis, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        # input == (bs,3,n_pts)
        bs = input.size()[0]
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(bs, 1)
        )
        if x.is_cuda:
            identity = identity.cuda()

        x = x + identity
        matrix = x.view(-1, 3, 3)
        return matrix


class TNetKD(nn.Module):
    """T-net is a special Neural Network that learns a transformation matrix
    that will rotate the input point cloud to a consistent orientation.

    This class is for TNet K-dims (such as 64)
    The process is the almost same as the TNet3D, but the dims are different
    """

    def __init__(self, k_dims: int):
        super().__init__()
        self.k_dims = k_dims

        self.conv1 = torch.nn.Conv1d(self.k_dims, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.k_dims * self.k_dims)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        # input == (bs,3,n_pts)
        bs = input.size()[0]
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = (
            Variable(torch.from_numpy(np.eye(self.k_dims).flatten().astype(np.float32)))
            .view(1, self.k_dims * self.k_dims)
            .repeat(bs, 1)
        )
        if x.is_cuda:
            identity = identity.cuda()

        x = x + identity
        matrix = x.view(-1, self.k_dims, self.k_dims)
        return matrix


class Transformation(nn.Module):
    def __init__(self, global_feat=True, feature_transformation=False, xyz_axis=3):
        super().__init__()
        self.t_net3d = TNet3D(xyz_axis)
        self.conv1 = torch.nn.Conv1d(xyz_axis, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transformation = feature_transformation

        if self.feature_transformation:
            self.t_net_kd = TNetKD(k_dims=64)

    def forward(self, input):
        # input == (bs,3,n_pts)
        B, D, N = input.size()
        matrix_3x3 = self.t_net3d(input)

        x = input.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, matrix_3x3)

        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            matrix_kxk = self.t_net_kd(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, matrix_kxk)
            x = x.transpose(2, 1)
        else:
            matrix_kxk = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:  # global features such as car
            return x, matrix_3x3, matrix_kxk

        else:  # local features such as tires
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), matrix_3x3, matrix_kxk
