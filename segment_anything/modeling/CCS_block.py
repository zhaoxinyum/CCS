import torch
import torch.nn as nn
import torch.nn.functional as F

class CCS(nn.Module):
    def __init__(
            self,
            maxiter=10,
            entropy_epsilon=1
    ):
        super(CCS, self).__init__()
        self.maxiter = maxiter
        # Fixed paramater
        self.entropy_epsilon = entropy_epsilon
        self.tau = 1 * self.entropy_epsilon

        self.nabla = nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]],
                                                [[[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]]], requires_grad=False))

        self.div = nn.Parameter(torch.tensor([[[[0., -1., 0.],
                                                [0., 1., 0.],
                                                [0., 0., 0.]],
                                               [[0., 0., 0.],
                                                [-1., 1., 0.],
                                                [0., 0., 0.]]]], requires_grad=False))

    def forward(self, o, vector_field):
        # mask shape:(1,1,H,W),
        o = torch.squeeze(o, dim=1)
        # o shape:(1, H, W)

        u = torch.sigmoid(o / self.entropy_epsilon)

        # main iteration
        q = torch.zeros_like(o, device=o.device)
        for i in range(self.maxiter):
            # 1.star-shape
            u_nabla = F.conv2d(u.unsqueeze(1), weight=self.nabla, stride=1, padding=1)
            q = q - self.tau * (
                    u_nabla[:, 0, :, :] * vector_field[:, :, 1] + u_nabla[:, 1, :, :] * vector_field[:, :, 0])
            q[q < 0] = 0
            Tq = F.conv2d(torch.stack([vector_field[:, :, 1] * q, vector_field[:, :, 0] * q], dim=1), weight=self.div,
                          padding=1)
            # Tq:(1,1,H,W)
            # 2.sigmoid
            u = torch.sigmoid((o - Tq.squeeze(dim=1)) / self.entropy_epsilon)

        u1 = (o - Tq.squeeze(dim=1)) / self.entropy_epsilon
        return u1.squeeze(0)

