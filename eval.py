import argparse

import cv2
from PIL import Image
import os
from tqdm import tqdm
import random
join = os.path.join
import torch
import torch.nn as nn
import logging
from build_samccs import samccs_model_registry
from torch.utils.data import DataLoader
from tool.calculate_metric import *
from tool.MyDataset import MyDataset
from segment_anything.lora import LoRA_sam
from visual_field import Vis_Field
from tool.smooth_field_torch import Smoothed_VF

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the SAM model.")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument('--if_ccs', type=bool, default=False)
    parser.add_argument("--model_name", type=str, default='SAMccs', help='SAM or SAMccs or SAMsloss')
    parser.add_argument("--checkpoint_folder", type=str, default='work_dir/SAMccs/ISIC', help="parameterfault")
    parser.add_argument("--data_name", type=str, default="ISIC")
    parser.add_argument('--test_folder', type=str, default='test')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    return parser.parse_args()


# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(data, device):
    """Move tensors in a dictionary to device."""
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(element, device) for element in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def custom_collate_fn(batch):
    return batch 


def setup_logging(log_file='evaluation.log'):
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def main():
    seed = 0
    set_seed(seed)
    args = parse_args()
    result_folder = os.path.join('test_result', args.model_name, args.data_name)
    if args.model_name == 'SAMccs':
        args.if_ccs = True
    else:
        args.if_ccs = False
    os.makedirs(result_folder, exist_ok=True)

    logger = setup_logging(log_file=join(result_folder, 'evaluation.log'))
    decoder_checkpoint=join(args.checkpoint_folder, 'decoder_best_parameters')
    lora_checkpoint = join(args.checkpoint_folder, 'lora_best_parameters.pth')
    logger.info(f'Using device {args.device}')
    # -------------load SAM----------------------

    sam_parameter = 'parameterfault/sam_vit_b_01ec64.pth'

    sam = samccs_model_registry['vit_b'](checkpoint=sam_parameter, decoder_parameter=decoder_checkpoint, if_ccs=args.if_ccs)
    # Create SAM LoRA
    sam_lora = LoRA_sam(sam, 4)
    sam_lora.load_lora_parameters(lora_checkpoint)
    model = sam_lora.sam
    model.to(args.device)
    test_image_path = join('dataset', args.data_name, args.test_folder, 'image')
    test_mask_path = join('dataset', args.data_name, args.test_folder, 'mask')
    test_point_path = join('dataset', args.data_name, args.test_folder, 'star_center_point')
    test_dataset = MyDataset(test_image_path, test_mask_path, test_point_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    # --------------evaluation------------------
    logger.info(f"begin evaluation of {args.data_name}_{args.test_folder} using {args.checkpoint_folder}")
    model.eval()
    
    output_folder = join(result_folder)
    os.makedirs(output_folder, exist_ok=True)
    mask_folder = join(output_folder, 'masks')
    field_folder = join(output_folder, 'fields')
    os.makedirs(mask_folder, exist_ok=True)
    if args.if_ccs:
        os.makedirs(field_folder, exist_ok=True)
    iou_scores, dice_scores, sensitivity_scores, specificity_scores, accuracy_scores = 0, 0, 0, 0, 0
    precision_scores = 0
    
    star_field = 0
    cosine_loss = nn.CosineSimilarity(dim=-1)
    with torch.no_grad():
        for i, batched_input in enumerate(tqdm(test_loader)):
            gt2D = torch.stack([x["gt"] for x in batched_input], dim=0)
            gt2D = gt2D.to(device=args.device, dtype=torch.float32)

            inputs = to_device(batched_input, args.device)
            output = model(inputs)
            mask = torch.stack([y["mask"] for y in output], dim=0)
            if args.if_ccs:
                point_coord = torch.stack([x['point_coord'] for x in inputs], dim=0)
                point_coord = point_coord.squeeze(0)
                gt_star_field = Smoothed_VF(point_coord, batched_input[0]['origin_size'])
                star_field_pre = torch.stack([y["vector_field"] for y in output], dim=0)
                star_field_loss = 1 - cosine_loss(star_field_pre, gt_star_field).mean()

            binary_predictions = (mask > 0).float()
            binary_predictions = binary_predictions.squeeze(0)
            gt2D = gt2D.squeeze(0)

            iou = calculate_iou(binary_predictions, gt2D)
            dice = get_dice(binary_predictions, gt2D)
            sensitivity, specificity, accuracy, precision = compute_metrics(gt2D, binary_predictions)
            
            logger.info(
                f"Picture [{inputs[0]['image_path']}]: iou-{iou:.4f},dice-{dice:.4f},sensitivity-{sensitivity:.4f},specificity-{specificity:.4f}, "
                f"accuracy-{accuracy:.4f},precision-{precision:.4f}")
            if args.if_ccs:
                logger.info(f'star_field_loss-{star_field_loss:.4f}')
                star_field += star_field_loss
                field_result = (star_field_pre.squeeze(0)).cpu().numpy()
                field_path = join(field_folder, inputs[0]['image_path']+'.png')
                
                gt_np = gt2D.cpu().numpy()
                kernel = np.ones((10, 10), np.uint8)
                dilated = cv2.dilate(gt_np, kernel, iterations=1)
                eroded = cv2.erode(gt_np, kernel, iterations=1)
                boundary_mask = dilated - eroded
                boundary_mask[::20, ::20] = 1
                field_result = field_result*boundary_mask[:, :, np.newaxis]
                # save the vector field
                Vis_Field(field_result, field_path)

            iou_scores = iou_scores + iou
            dice_scores = dice_scores + dice
            sensitivity_scores += sensitivity
            specificity_scores += specificity
            accuracy_scores += accuracy
            precision_scores +=precision
    

            # save the result
            result = binary_predictions.mul(255).byte().cpu().numpy()
    
            image = Image.fromarray(result, mode='L')
            file_name = inputs[0]['image_path']+'.png'
            image.save(join(mask_folder, file_name))
    mean_star_loss = star_field / len(test_loader)
    mean_iou = iou_scores / len(test_loader)
    mean_dice = dice_scores / len(test_loader)
    mean_sensitivity, mean_specificity, mean_accuracy = sensitivity_scores / len(test_loader), specificity_scores / len(
        test_loader), accuracy_scores / len(test_loader)
    mean_precision = precision_scores / len(test_loader)
    
    logger.info("Evaluation finish")
    logger.info(f"mean_iou - {mean_iou}, mean_dice - {mean_dice}, mean_sensitivity-{mean_sensitivity}, "
                f"mean_specificity-{mean_specificity}, mean_accuracy-{mean_accuracy},mean_precision-{mean_precision},mean_star_loss-{mean_star_loss}")

if __name__ == "__main__":
    main()
