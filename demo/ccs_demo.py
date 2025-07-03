import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


def softmax(distance_matrix, epsilon):
    distance_matrix = -distance_matrix / epsilon

    max_vals = np.max(distance_matrix, axis=0, keepdims=True)
    exp_matrix = np.exp(distance_matrix - max_vals)  # shape: (n, h, w)

    # softmax：
    softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=0, keepdims=True)

    return softmax_matrix


def Smoothed_VF(points, h, w, epsilon=1):
    """
        points: ndarray with shape (n, 2)
        h: height of image
        w: width of image

        """
    # point coordinate matrix (h, w)
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # repeat matrix (n, h, w)
    distance_matrix = np.zeros((len(points), h, w))

    # compute distance matrix
    for i, (x, y) in enumerate(points):
        distance_matrix[i] = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)

    softmax_matrix = softmax(distance_matrix, epsilon=epsilon)
    # reshape  points to (n, 2, 1, 1)
    points = points[:, :, np.newaxis, np.newaxis]  # shape: (n, 2, 1, 1)

    # compute coordinates with weight matrix
    weighted_coordinates = softmax_matrix[:, np.newaxis, :, :] * points  # shape: (n, 2, h, w)

    result = np.sum(weighted_coordinates, axis=0)  # shape:(2,h,w)
    pixel_coords = np.stack([x_grid, y_grid], axis=0)

    #  (2, h, w)
    diff_matrix = result - pixel_coords

    # 将形状为 (2, h, w) 的差值矩阵变形为 (h, w, 2)
    matrix = np.transpose(diff_matrix, (1, 2, 0))  # 从 (2, h, w) 变形为 (h, w, 2)
    norm = np.linalg.norm(matrix, axis=2, keepdims=True)  # 结果为 (h, w, 1)

    # 防止除以0，避免数值不稳定
    norm[norm == 0] = 1

    # 对矩阵沿第 0 维度进行归一化
    normalized_matrix = matrix / norm

    return normalized_matrix


class CCS(nn.Module):
    def __init__(
            self,
            maxiter=5000,
            entropy_epsilon=0.5,
    ):
        super(CCS, self).__init__()
        self.maxiter = maxiter

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
        # mask shape:(B,1,H,W),
        o = torch.squeeze(o, dim=1)
        # o shape:(B, H, W)

        u = torch.sigmoid(o / self.entropy_epsilon)

        # main iteration
        q = torch.zeros_like(o, device=o.device)

        for i in range(self.maxiter):
            # 2.star-shape
            u_nabla = F.conv2d(u.unsqueeze(1), weight=self.nabla, stride=1, padding=1)
            q = q - self.tau * (
                    u_nabla[:, 0, :, :] * vector_field[:, :, 1] + u_nabla[:, 1, :, :] * vector_field[:, :, 0])
            q[q < 0] = 0
            Tq = F.conv2d(torch.stack([vector_field[:, :, 1] * q, vector_field[:, :, 0] * q], dim=1), weight=self.div,
                          padding=1)
            # Tq:(B,1,H,W)
            # 3.sigmoid
            u = torch.sigmoid((o -  Tq.squeeze(dim=1)) / self.entropy_epsilon)
            
        u1 = (o - Tq.squeeze(dim=1)) / self.entropy_epsilon
        return u1.squeeze(0)



if __name__ == '__main__':
    import cv2

    img = cv2.imread('./toy1.png', cv2.IMREAD_GRAYSCALE)
    img1 = np.where(img > 0, 10, -10)
    img_torch = torch.Tensor(img1).unsqueeze(0).unsqueeze(0)

    # 3. 计算 Softmax
    def softmax2(x, epislon):
        e_x = np.exp((x - np.max(x)) / epislon) 
        return e_x / e_x.sum()


    img_point = np.array([
        [42, 112],
        [126, 87],
        [126, 142]
    ])

 
    height, width = img.shape[:2]
    epsilon = 1

    vector_field0 = Smoothed_VF(img_point, height, width, epsilon)

    vector_field_torch0 = torch.Tensor(vector_field0)
    print(vector_field_torch0.shape)
    miter, entropy_e = 3000, 1
    msstd = CCS(maxiter=miter, entropy_epsilon=entropy_e)
    mask = msstd(img_torch, vector_field_torch0)
    
    mask_np = mask.detach().numpy()
    mask_uint8 = (mask_np > 0).astype(np.uint8) * 255
    img_color = np.stack([mask_uint8]*3, axis=-1)
    
    plt.imshow(img_color)
    plt.scatter([p[0] for p in img_point], [p[1] for p in img_point], c='red', s=20)
    plt.axis('off')
    plt.savefig(f'toy1_{epsilon}_T{miter}_E{entropy_e}.png', bbox_inches='tight', pad_inches=0)
    plt.show()

