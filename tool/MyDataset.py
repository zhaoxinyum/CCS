import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple
from torchvision.transforms.functional import resize, to_pil_image
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, image_folder, target_folder, point_folder):
        self.origin_size = None
        self.image_scale_size = None
        self.image_folder = image_folder
        self.target_folder = target_folder
        self.point_folder = point_folder

        self.image_files = sorted(os.listdir(image_folder))
        self.target_files = sorted(os.listdir(target_folder))
        self.point_files = sorted(os.listdir(point_folder))
        self.pixel_mean = torch.Tensor([106.7138,  70.5478,  50.7495]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([74.6907, 52.2190, 37.5536]).view(-1, 1, 1)

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return [newh, neww]
    @staticmethod
    def softmax(distance_matrix, epsilon):
        distance_matrix = -distance_matrix / epsilon

        max_vals = np.max(distance_matrix, axis=0, keepdims=True)
        exp_matrix = np.exp(distance_matrix - max_vals)  # shape: (n, h, w)

        # softmax：
        softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=0, keepdims=True)

        return softmax_matrix

    def Smoothed_VF(self, points, h, w):
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

        softmax_matrix = self.softmax(distance_matrix, epsilon=1)
        # reshape  points to (n, 2, 1, 1)
        points = points[:, :, np.newaxis, np.newaxis]  # shape: (n, 2, 1, 1)

        # compute coordinates with weight matrix
        weighted_coordinates = softmax_matrix[:, np.newaxis, :, :] * points  # shape: (n, 2, h, w)

        result = np.sum(weighted_coordinates, axis=0)  # shape:(2,h,w)
        pixel_coords = np.stack([x_grid, y_grid], axis=0)  

        diff_matrix = result - pixel_coords

        matrix = np.transpose(diff_matrix, (1, 2, 0))  
        norm = np.linalg.norm(matrix, axis=2, keepdims=True)  
       
        norm[norm == 0] = 1
        normalized_matrix = matrix / norm

        return normalized_matrix
    def preprocess(self, image) -> torch.Tensor:
        """
        Expects a numpy with shape HxWxC
        """
        # scaling to 1024x1024
        img_size = 1024
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], img_size)

        image_scale = np.array(resize(to_pil_image(image), target_size))

        input_image_torch = torch.as_tensor(np.array(image_scale))
        # hxWxC -->  cxhxw
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        self.image_scale_size = tuple(input_image_torch.shape[-2:])
        self.origin_size = tuple(image.shape[0:2])
        # Normalize colors
        input_image_torch = (input_image_torch - self.pixel_mean) / self.pixel_std

        # Pad to Cx1024x1024

        h, w = input_image_torch.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(input_image_torch, (0, padw, 0, padh))
        return x

    @torch.no_grad()
    def preprocess_point(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], 1024
        )
        coords = coords.astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def __getitem__(self, idx):
        '''
        output:
        image:tensor(3,1024,1024)
        image_scale_size:tuple(2)
        origin_size:tuple(2)
        gt:ndarray(H,W)
        '''

        image_path = os.path.join(self.image_folder, self.image_files[idx])
        target_path = os.path.join(self.target_folder, self.target_files[idx])
        point_path = os.path.join(self.point_folder, self.point_files[idx])
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_torch = self.preprocess(image)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        target = np.where(target > 0, 1, 0)

        coordinates = []
        with open(point_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                x, y = map(float, line.strip().split(','))
                coordinates.append([x, y])
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
        # vector_field = self.Smoothed_VF(coordinates, self.origin_size[0], self.origin_size[1])

        output = {
            "image": image_torch,
            "image_scale_size": self.image_scale_size,
            "origin_size": self.origin_size,
            "gt": torch.FloatTensor(target),
            'point_coord': coordinates,
            'image_path':file_name,
        }
        return output


