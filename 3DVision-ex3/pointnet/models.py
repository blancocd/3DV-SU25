import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, dims):
        super(TNet, self).__init__()
        self.dims = dims
        self.conv1 = nn.Conv1d(dims, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dims * dims)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        '''
        x: tensor of size (B, N, d)
           , where B is batch size, d is the number of dimensions and N is the number of points per object
        output: tensor of size (B, d, d) transformation matrix
        '''
        B, N, D = x.size()
        x = x.transpose(2, 1)  # (B, D, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(self.dims, device=x.device).view(1, self.dims * self.dims).repeat(B, 1)
        x = x + identity
        x = x.view(-1, self.dims, self.dims)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, vanilla=False):
        """
        PointNet Encoder
        Args:
            vanilla (bool): If True, use vanilla PointNet without T-Nets.
                            If False, use T-Net for both input and per point features.
        """
        super(PointNetEncoder, self).__init__()
        self.vanilla = vanilla

        if not self.vanilla:
            self.input_transform = TNet(dims=3)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        if not self.vanilla:
            self.feature_transform = TNet(dims=64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        
        
    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
        output: 
            global_feat: global feature tensor of size (B, 1024)
            pp_feat: per point feature tensor of size (B, N, 64)
            rot1: transform matrix, tensor of size (B, 3, 3) if not vanilla
            rot2: transform matrix, tensor of size (B, 64, 64) if not vanilla
        You will need rot1, rot2 for the regularization loss in the training loop!
        '''
        assert points.size(2) == 3, "Input points should be of shape (B, N, 3)"
        B, N, _ = points.size()
        rot1 = rot2 = None

        x = points.transpose(2, 1)  # (B, 3, N)

        if not self.vanilla:
            rot1 = self.input_transform(points)  # (B, 3, 3)
            x = torch.bmm(rot1, x)  # (B, 3, N)

        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        pp_feat = x.transpose(2, 1)  # (B, N, 64)

        if not self.vanilla:
            rot2 = self.feature_transform(pp_feat)  # (B, 64, 64)
            x = torch.bmm(rot2, x)  # (B, 64, N)

        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = self.bn3(self.conv3(x))  # (B, 1024, N)
        global_feat = torch.max(x, 2)[0]  # (B, 1024)

        return global_feat, pp_feat, rot1, rot2

        
class cls_model(nn.Module):
    def __init__(self, vanilla=False, num_classes=3):
        super(cls_model, self).__init__()
        self.encoder = PointNetEncoder(vanilla=vanilla)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
        output: 
            logits: tensor of size (B, num_classes),
            rot1: transform matrix, tensor of size (B, 3, 3) if not vanilla
            rot2: transform matrix, tensor of size (B, 64, 64) if not vanilla
        '''
        global_feat, _, rot1, rot2 = self.encoder(points)
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits, rot1, rot2
        

class seg_model(nn.Module):
    def __init__(self, vanilla=False, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.encoder = PointNetEncoder(vanilla=vanilla)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, num_seg_classes, 1)
        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
        output: 
            logits: tensor of size (B, N, num_seg_classes)
            rot1: transform matrix, tensor of size (B, 3, 3) if not vanilla
            rot2: transform matrix, tensor of size (B, 64, 64) if not vanilla
        '''
        global_feat, pp_feat, rot1, rot2 = self.encoder(points)
        B, N, _ = points.shape

        global_expanded = global_feat.unsqueeze(2).repeat(1, 1, N)  # (B, 1024, N)
        x = torch.cat([pp_feat.transpose(2, 1), global_expanded], dim=1)  # (B, 1088, N)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        logits = self.conv4(x).transpose(2, 1)  # (B, N, num_seg_classes)

        return logits, rot1, rot2
    
    
class cor_model(torch.nn.Module):
    def __init__(self, num_points=4096):
        super(cor_model, self).__init__()
        self.encoder = PointNetEncoder(vanilla=False)
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_points * 3)
        )

        
    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
        output: 
            reconstruction: tensor of size (B, num_points, 3)
            r1: transform matrix, tensor of size (B, 3, 3)
            r2: transform matrix, tensor of size (B, 64, 64)
        '''
        global_feat, _, r1, r2 = self.encoder(points)  # global_feat: (B, 1024)
        out = self.mlp(global_feat)  # (B, num_points * 3)
        reconstruction = out.view(-1, self.num_points, 3)  # (B, num_points, 3)
        return reconstruction, r1, r2