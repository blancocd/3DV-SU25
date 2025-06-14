import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, dims):
        super(TNet, self).__init__()
        # TODO: Your code here
        
    def forward(self, x):
        '''
        x: tensor of size (B, N, d)
           , where B is batch size, d is the number of dimensions and N is the number of points per object
        output: tensor of size (B, d, d) transformation matrix
        '''
        # TODO: Your code here

class PointNetEncoder(nn.Module):
    def __init__(self, vanilla=False):
        """
        PointNet Encoder
        Args:
            vanilla (bool): If True, use vanilla PointNet without T-Nets.
                            If False, use T-Net for both input and per point features.
        """
        super(PointNetEncoder, self).__init__()
        # TODO: Your code here
        
        
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
        # TODO: Your code here

        
class cls_model(nn.Module):
    def __init__(self, vanilla=False, num_classes=3):
        super(cls_model, self).__init__()
        # TODO: Your code here
        
    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
        output: 
            logits: tensor of size (B, num_classes),
            rot1: transform matrix, tensor of size (B, 3, 3) if not vanilla
            rot2: transform matrix, tensor of size (B, 64, 64) if not vanilla
        '''
        # TODO: Your code here
        

class seg_model(nn.Module):
    def __init__(self, vanilla=False, num_seg_classes = 6):
        super(seg_model, self).__init__()
        # TODO: Your code here
        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
        output: 
            logits: tensor of size (B, N, num_seg_classes)
            rot1: transform matrix, tensor of size (B, 3, 3) if not vanilla
            rot2: transform matrix, tensor of size (B, 64, 64) if not vanilla
        '''
        # TODO: Your code here
    
    
class cor_model(torch.nn.Module):
    def __init__(self, num_points=4096):
        super(cor_model, self).__init__()
        # TODO: Your code here
        
    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
        output: 
            reconstruction: tensor of size (B, num_points, 3)
            r1: transform matrix, tensor of size (B, 3, 3)
            r2: transform matrix, tensor of size (B, 64, 64)
        '''
        # TODO: Your code here