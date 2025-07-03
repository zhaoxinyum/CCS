import os
import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm
from skeleton_block import skeleton

class OpenCenterPreprocessor:
    def __init__(self, mask_folder, output_folder, mask_points_folder, field_output_folder, kernel_size=5, iterations=1, num_samples=100):
        self.mask_folder = mask_folder
        self.output_folder = output_folder
        self.field_output_folder = field_output_folder
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.num_samples = num_samples

        self.mask_points_folder = mask_points_folder
        self.skel = skeleton()

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.mask_points_folder, exist_ok=True)
        os.makedirs(self.field_output_folder, exist_ok=True)

    
        
    def softmax_tensor(self, distance_matrix, epsilon=1):
        """
        distance_matrix: tensor of shape (n, h, w)
        epsilon: smoothing factor
        """
        distance_matrix = -distance_matrix / epsilon
    
        # 为了避免数值不稳定，使用max进行稳定softmax计算
        max_vals = torch.max(distance_matrix, dim=0, keepdim=True).values
        exp_matrix = torch.exp(distance_matrix - max_vals)  # shape: (n, h, w)

        # 计算softmax
        softmax_matrix = exp_matrix / torch.sum(exp_matrix, dim=0, keepdim=True)
        
        return softmax_matrix
    
    def save_sampled_point(self, sampled_points, output_file):
        np.savetxt(output_file, sampled_points, fmt='%d', delimiter=',', header='', comments='')
    
    def Smoothed_VF_tensor(self, points, h, w, epsilon=10):
        """
        points: tensor with shape (n, 2)
        h: height of image
        w: width of image
        epsilon: smoothing factor for softmax
        """
        # 点坐标矩阵 (h, w)
        points = points.to('cuda')
        y_grid, x_grid = torch.meshgrid(torch.arange(h).to('cuda'), torch.arange(w).to('cuda'))

        # 创建距离矩阵 (n, h, w)
        distance_matrix = torch.zeros((len(points), h, w)).to('cuda')

        # 计算距离矩阵
        for i, (x, y) in enumerate(points):
            distance_matrix[i] = torch.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)

        # 使用softmax计算权重矩阵
        softmax_matrix = self.softmax_tensor(distance_matrix, epsilon=epsilon)

        # 将points reshape为 (n, 2, 1, 1)
        points = points[:, :, None, None]  # shape: (n, 2, 1, 1)

        # 计算加权后的坐标
        weighted_coordinates = softmax_matrix[:, None, :, :] * points  # shape: (n, 2, h, w)

        # 对n个点进行求和，得到结果形状 (2, h, w)
        result = torch.sum(weighted_coordinates, dim=0)  # shape:(2, h, w)
        
        # 创建像素坐标矩阵 (2, h, w)
        pixel_coords = torch.stack([x_grid, y_grid], dim=0)  # shape: (2, h, w)

        # 计算差值矩阵 (2, h, w)
        diff_matrix = result - pixel_coords

        # 将 (2, h, w) 变形为 (h, w, 2)
        matrix = diff_matrix.permute(1, 2, 0)  # shape: (h, w, 2)

        # 计算每个箭头的模长，防止除以0
        norm = torch.linalg.norm(matrix, dim=2, keepdim=True)  # shape: (h, w, 1)
        norm[norm == 0] = 1  # 防止除以0

        # 对矩阵沿最后一维度进行归一化
        normalized_matrix = matrix / norm

        return normalized_matrix

    def process_mask(self, mask_file):
        mask_image = T.ToTensor()(Image.open(mask_file).convert('L')).to('cuda')
        h, w = mask_image.shape[-2], mask_image.shape[-1]

        with torch.no_grad():
            e = self.skel.erode(mask_image)
            S = torch.zeros_like(e)
            # S += e - skel.open(e)
            for i in range(2):
                e = self.skel.erode(e)
            S += e - self.skel.open(e)
            e = self.skel.erode(e)
            S += e - self.skel.open(e)
        non_zero_index = torch.nonzero(S.squeeze().cpu())
        
        
        if len(non_zero_index) > 0:
            sampled_points = torch.zeros_like(non_zero_index)
            sampled_points[:,0], sampled_points[:,1] = non_zero_index[:,1], non_zero_index[:,0]
            filename = os.path.splitext(os.path.basename(mask_file))[0]
            # save mask with points
            save_path = os.path.join(self.mask_points_folder, f'{filename}.png')
            plt.close()  # 关闭上一次的绘图
            plt.imshow(mask_image.squeeze().cpu(), cmap='gray')
            plt.scatter(sampled_points[:, 1], sampled_points[:, 0], s=10, c='red')
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            # plt.show()
            output_path = os.path.join(self.output_folder, f"{filename}.txt")
            self.save_sampled_point(sampled_points, output_path)
            with torch.no_grad():
                vector_field_torch = self.Smoothed_VF_tensor(sampled_points, h, w)
            field_save_path = os.path.join(self.field_output_folder, f'{filename}.pt')
            torch.save(vector_field_torch.cpu(), field_save_path)
        else:
            print("No centers found in the image")
    

    def process_all_masks(self):
        mask_file = os.listdir(self.mask_folder)
        loop = tqdm(enumerate(mask_file), total=len(mask_file))
        for _, filename in loop:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                name = os.path.splitext(os.path.basename(filename))[0]
                txtname = name+'.txt'
                # if txtname in os.listdir(self.output_folder):
                #     continue
                # print(f'process image{filename}')
                mask_path = os.path.join(self.mask_folder, filename)

                self.process_mask(mask_path)
        print('Finish Processing')


if __name__ == '__main__':
    
    Test_folder = './dataset/WHU/train'
    mask_folder = os.path.join(Test_folder, 'masks')
    output_folder = os.path.join(Test_folder, 'open_center_point')
    mask_point_folder = os.path.join(Test_folder, 'open_mask_point')
    field_output_folder = os.path.join(Test_folder, 'open_center_field')
    Processor = OpenCenterPreprocessor(mask_folder=mask_folder,
                                   output_folder=output_folder,
                                   mask_points_folder=mask_point_folder,
                                   field_output_folder=field_output_folder)
    Processor.process_all_masks()
    
    Test_folder = './dataset/WHU/val'
    mask_folder = os.path.join(Test_folder, 'masks')
    output_folder = os.path.join(Test_folder, 'open_center_point')
    mask_point_folder = os.path.join(Test_folder, 'open_mask_point')
    field_output_folder = os.path.join(Test_folder, 'open_center_field')
    Processor = OpenCenterPreprocessor(mask_folder=mask_folder,
                                   output_folder=output_folder,
                                   mask_points_folder=mask_point_folder,
                                   field_output_folder=field_output_folder)
    Processor.process_all_masks()
    
    Test_folder = './dataset/WHU/test'
    mask_folder = os.path.join(Test_folder, 'masks')
    output_folder = os.path.join(Test_folder, 'open_center_point')
    mask_point_folder = os.path.join(Test_folder, 'open_mask_point')
    field_output_folder = os.path.join(Test_folder, 'open_center_field')
    Processor = OpenCenterPreprocessor(mask_folder=mask_folder,
                                   output_folder=output_folder,
                                   mask_points_folder=mask_point_folder,
                                   field_output_folder=field_output_folder)
    Processor.process_all_masks()

    print('ok')