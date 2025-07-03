import torch
import numpy as np
from skimage import segmentation as skimage_seg
import torch.nn.functional as F


def compute_metrics(ground_truth, segmentation_result):
    ground_truth = ground_truth.bool()
    segmentation_result = segmentation_result.bool()

    TP = (ground_truth & segmentation_result).sum().item()
    TN = (~ground_truth & ~segmentation_result).sum().item()
    FP = (~ground_truth & segmentation_result).sum().item()
    FN = (ground_truth & ~segmentation_result).sum().item()

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    return sensitivity, specificity, accuracy, precision



def get_dice(SR, GT):
    intersection = ((SR == 1) & (GT == 1)).sum().item()
    dice = 2 * intersection / (SR.sum().item() + GT.sum().item() + 1e-10)
    return dice


def calculate_dice_batch(predictions, targets, threshold=0.5, epsilon=1e-8):
    predictions = torch.sigmoid(predictions)
    binary_predictions = (predictions > threshold).float()

    batch_dice = []
    for i in range(predictions.size(0)):
        intersection = torch.sum(binary_predictions[i] * targets[i])
        union = torch.sum(binary_predictions[i]) + torch.sum(targets[i]) + epsilon
        dice = (2.0 * intersection + epsilon) / union
        batch_dice.append(dice.item())
    average_dice = torch.mean(torch.tensor(batch_dice))

    return average_dice.item()


def calculate_iou(prediction, target):
    intersection = torch.logical_and(prediction, target).sum().float()
    union = torch.logical_or(prediction, target).sum().float()
    iou = (intersection + 1e-6) / (union + 1e-6) 
    return iou.item()


# -----------------multi-point star-shaped loss-----------------
def sobel_kernels(device, dtype):
    # Define Sobel kernels for gradient calculation
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

    return sobel_x, sobel_y


def nabla_image(images):
    nabla = torch.tensor([[[[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]]],
                          [[[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]]], device=images.device)
    # Add channel dimension for convolution: (B, 1, H, W)
    images = images.unsqueeze(1)
    u_nabla = F.conv2d(images, weight=nabla, stride=1, padding=1)
    # Permute to match the output shape: (B, H, W, 2)
    gradient = u_nabla.permute(0, 2, 3, 1)

    return gradient



def compute_image_gradient_conv(images):
    # images: (B, H, W), batch of images
    B, H, W = images.shape

    # Add channel dimension for convolution: (B, 1, H, W)
    images = images.unsqueeze(1)

    # Get Sobel kernels
    # Get the device and dtype of the input images
    device = images.device
    dtype = images.dtype

    # Get Sobel kernels with the same device and dtype as images
    sobel_x, sobel_y = sobel_kernels(device, dtype)
    # Apply convolution with Sobel kernels to compute gradients
    grad_x = F.conv2d(images, sobel_x, padding=1)  # (B, 1, H, W)
    grad_y = F.conv2d(images, sobel_y, padding=1)  # (B, 1, H, W)

    # Stack gradients to form (B, H, W, 2)
    gradient = torch.cat((grad_x, grad_y), dim=1)  # (B, 2, H, W)

    # Permute to match the output shape: (B, H, W, 2)
    gradient = gradient.permute(0, 2, 3, 1)

    return gradient


def compute_cosine_similarity_loss_batch(gradient1, gradient2):
    # gradient1, gradient2: (B, H, W, 2) matrices
    B, H, W, _ = gradient1.shape
    gradient1_flat = gradient1.view(B, -1, 2)  # (B, H*W, 2)
    gradient2_flat = gradient2.view(B, -1, 2)  # (B, H*W, 2)

    # Compute cosine similarity along the last dimension for each batch
    cosine_similarity = F.cosine_similarity(gradient1_flat, gradient2_flat, dim=-1)  # (B, H*W)

    loss = torch.relu(-cosine_similarity)
    # Take the mean for each batch
    loss = loss.mean(dim=-1)  # (B,)

    return loss.mean()  # Mean over batch


def shape_loss_batch(images, known_gradients):
    # images: (B, H, W)
    # known_gradients: (B, H, W, 2)
    images = torch.sigmoid(images)
    # Compute image gradients for the batch using convolution
    image_gradients = nabla_image(images)
    loss = compute_cosine_similarity_loss_batch(image_gradients, known_gradients)

    return loss
