'''
骨架提取模块
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 绘制光滑骨架
class smooth_skeleton(nn.Module):
    def __init__(self,
                 half_size=1,
                 iter=200,
                 epsilon=0.05):
        super(smooth_skeleton, self).__init__()
        
        self.ker = torch.ones((2*half_size+1, 2*half_size+1)).reshape((1, 1, 2*half_size+1, 2*half_size+1))
        self.iter = iter
        self.epsilon = epsilon
        self.size = self.conv.kernel_size[0] ** 2
    
    def smooth_trans(self, img):
        self.ker = self.ker.to(img.device)
        img = F.conv2d(F.pad(torch.exp(img/self.epsilon),
                             pad=(1,1,1,1), mode='replicate'), self.ker)
        return self.epsilon * (torch.log(img) - math.log(self.size))

    def erode(self, img): # 光滑腐蚀
        return -self.smooth_trans(-img)

    def dilate(self, img): # 光滑膨胀
        return self.smooth_trans(img)

    def open(self, img): # 开运算
        return torch.clamp(self.dilate(self.erode(img)), min=0, max=1)

    def close(self, img): # 闭运算
        return self.erode(self.dilate(img))

    def forward(self, img):
        skel = torch.relu(img - self.open(img))
        for j in range(self.iter):
            img = self.erode(img)
            delta = torch.relu(img - self.open(img))
            skel = skel + delta
            
        return skel

# 绘制骨架
class skeleton(nn.Module):
    def __init__(self,
                 half_size=1,
                 iter=10):
        super(skeleton, self).__init__()
        self.trans = nn.MaxPool2d(kernel_size=2*half_size+1,
                                  stride=1,
                                  padding=half_size) # 最大池化
        self.iter = iter

    def erode(self, img): # 最小池化腐蚀
        return -self.trans(-img)

    def dilate(self, img): # 最大池化膨胀
        return self.trans(img)

    def open(self, img): # 开运算
        return self.dilate(self.erode(img))

    def close(self, img): # 闭运算
        return self.erode(self.dilate(img))

    def forward(self, img):
        img0 = img.clone()
        skel = img0 - self.open(img0)
        for j in range(self.iter):
            img0 = self.erode(img0)
            delta = img0 - self.open(img0)
            skel = skel + delta
            
        return skel
    
class soft_cldice(nn.Module):
    def __init__(self,
                 device,
                 half_size=3,
                 iter=20,
                 epsilon=1e-6):
        """[function to compute dice loss]
        
        parms:
            device: 'cpu' or 'cuda'
            half_size: the half size of pool kernel
            iter: number of iterations of skeletonize
            epsilon: use for smoothing cl dice

        Args:
            y_true ([float32]): [ground truth image] (1, H, W)
            y_pred ([float32]): [predicted image] (1, H, W)

        Returns:
            [float32]: [loss value]
        """
        super(soft_cldice, self).__init__()
        self.epsilon = epsilon
        self.soft_skeletonize = skeleton(half_size, iter).to(device)

    def forward(self, y_pred, y_true):
        soft_sk = self.soft_skeletonize(torch.cat((y_pred, y_true), dim=0)) # 提取骨架
        skel_pred, skel_true = soft_sk[:y_pred.shape[0]], soft_sk[-y_true.shape[0]:]
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.epsilon)/(torch.sum(skel_pred)+self.epsilon)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.epsilon)/(torch.sum(skel_true)+self.epsilon)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice

