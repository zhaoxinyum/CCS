import torch

def softmax(distance_matrix, epsilon):
    
    distance_matrix = -distance_matrix / epsilon
    max_vals = torch.max(distance_matrix, dim=0, keepdim=True).values
    exp_matrix = torch.exp(distance_matrix - max_vals)  # (n, h, w)

    softmax_matrix = exp_matrix / torch.sum(exp_matrix, dim=0, keepdim=True)

    return softmax_matrix


def Smoothed_VF(points, origin_size):
    """
    points: tensor with shape (n, 2)
    origin_size: (h, w)
    h: height of image
    w: width of image
    """
    h, w = origin_size

    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    y_grid = y_grid.to(points.device)
    x_grid = x_grid.to(points.device)


    distance_matrix = torch.sqrt((x_grid[None, :, :] - points[:, 0, None, None]) ** 2 +
                                 (y_grid[None, :, :] - points[:, 1, None, None]) ** 2)

    softmax_matrix = softmax(distance_matrix, epsilon=1)

    points = points[:, :, None, None]  # (n, 2, 1, 1)

    # 计算加权坐标
    weighted_coordinates = softmax_matrix[:, None, :, :] * points  # (n, 2, h, w)

    # 计算结果，形状为 (2, h, w)
    result = torch.sum(weighted_coordinates, dim=0)

    # 创建 pixel_coords 坐标矩阵
    pixel_coords = torch.stack([x_grid, y_grid], dim=0)  # 形状为 (2, h, w)

    # 计算差值矩阵，形状为 (2, h, w)
    diff_matrix = result - pixel_coords

    # 将 (2, h, w) 转换为 (h, w, 2)
    matrix = diff_matrix.permute(1, 2, 0)

    # 计算范数矩阵，并确保没有零值避免数值不稳定
    norm = torch.norm(matrix, dim=2, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)

    # 对矩阵进行归一化
    normalized_matrix = matrix / norm

    return normalized_matrix