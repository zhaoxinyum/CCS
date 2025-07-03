import math

from build_samccs import samccs_model_registry
from tool.MyDataset import MyDataset
import datetime
import logging
import argparse
import os
import numpy as np
import random

join = os.path.join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from tool.calculate_metric import calculate_dice_batch, shape_loss_batch
from tool.smooth_field_torch import Smoothed_VF
from segment_anything.lora import LoRA_sam


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SAM model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--gpu", type=str, default="0", help="Specify GPU block number to use")
    parser.add_argument(
        "--sam_checkpoint", type=str, default="parameterfault/sam_vit_b_01ec64.pth"
    )
    parser.add_argument("--model_name", type=str, default="SAMccs", help="model (SAM or SAMccs or SAMsloss)")
    parser.add_argument("--dataname", type=str, default="ISIC")

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="weight decay (default: 0.01)"
    )
    return parser.parse_args()


def main():
    seed = 0
    set_seed(seed)
    CONFIG = {'ISIC': (1, 50, 1e-4), 'Refuge': (1, 30, 1e-4), 'Kvasir':(1, 30, 1e-4)}
    args = parse_args()
    if args.model_name == 'SAMccs':
        args.if_ccs = True
        args.sloss = False
    elif args.model_name == 'SAMsloss':
        args.sloss = True
        args.if_ccs = False
    elif args.model_name == 'SAM':
        args.if_ccs = False
        args.sloss = False
    batch_size, epochs, lr = CONFIG[args.dataname]
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S') 
    model_save_path = os.path.join('work_dir', args.model_name + "_runs", args.dataname, time_str)
    
    os.makedirs(model_save_path, exist_ok=True)
    
    logger = setup_logging(join(model_save_path, 'training.log'))
    logger.info(f'{args}')
    logger.info(f'Using device {args.device}')
    logger.info(f'Train Parameter:\n \
                dataset - {args.dataname}\n \
                epochs - {epochs}\n \
                learning rate - {lr}\n \
                batch size - {batch_size}\n \
                random seed - {seed}')

    # ----------- TensorBoard --------------------
    writer = SummaryWriter(log_dir=join(model_save_path, 'logs'))
    # ------------dataset-------------------
    train_image_path = join('dataset', args.dataname, 'train', 'image')
    train_point_path = join('dataset', args.dataname, 'train', 'star_center_point')
    train_mask_path = join('dataset', args.dataname, 'train', 'mask')
    val_image_path = join('dataset', args.dataname, 'val', 'image')
    val_mask_path = join('dataset', args.dataname, 'val', 'mask')
    val_point_path = join('dataset', args.dataname, 'val', 'star_center_point')
    train_dataset = MyDataset(train_image_path, train_mask_path, train_point_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              collate_fn=custom_collate_fn)

    val_dataset = MyDataset(val_image_path, val_mask_path, val_point_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                            collate_fn=custom_collate_fn)

    # -------------load SAM----------------------
    samccs = samccs_model_registry['vit_b'](checkpoint=args.sam_checkpoint, if_ccs=args.if_ccs)
    # Create SAM LoRA
    sam_lora = LoRA_sam(samccs, 4)
    model = sam_lora.sam
    model.to(args.device)
    # ---------------total number of trainable parameters-------------
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'total number of trainable parameters {trainable_num}')

    # -------------loss function and optimizer---------------
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=args.weight_decay
                                 )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=Lr_lambda)
    # --------------train----------------------------
    logger.info(f'Parameter weight saved path:{model_save_path}')
    logger.info('Begin training---------------------------')
    BEST_SCORE = 0
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, args.device, args.if_ccs, args.sloss)
        scheduler.step()
        val_loss, val_score = val(model, val_loader, criterion, calculate_dice_batch, args.device, args.if_ccs)

        writer.add_scalar('Training Loss', train_loss, epoch + 1)
        writer.add_scalar('Validation Loss', val_loss, epoch + 1)
        writer.add_scalar('Validation Score', val_score, epoch + 1)
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}]: Train Loss - {train_loss:.4f}, Validation Loss - {val_loss:.4f}, "
            f"Val_score-{val_score:.4f}")
        
        #save the latest model
        sam_lora.save_lora_parameters(join(model_save_path, "lora_latest_parameters.pth"))

        model.save_parameters(join(model_save_path, "decoder_latest_parameters"))

        if val_score > BEST_SCORE:
            BEST_SCORE = val_score
            sam_lora.save_lora_parameters(join(model_save_path, "lora_best_parameters.pth"))

            model.save_parameters(join(model_save_path, "decoder_best_parameters"))
            logger.info(f'BEST {epoch + 1} saved!')
    # close TensorBoard
    writer.close()


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


# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_loader, optimizer, criterion, device, if_ccs=True, if_sloss=False):
    model.train()
    total_loss = 0
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    cosine_loss = nn.CosineSimilarity(dim=-1)

    for step, batched_input in loop:
        optimizer.zero_grad(set_to_none=True)
        gt2D = torch.stack([x["gt"] for x in batched_input], dim=0)
        gt2D = gt2D.to(device=device, dtype=torch.float32)
        inputs = to_device(batched_input, device)
        # the star center points
        # point_coord = torch.stack([x['point_coord'] for x in inputs], dim=0)
        # point_coord = point_coord.squeeze(0)
        # gt_star_field = Smoothed_VF(point_coord, batched_input[0]['origin_size'])
        # convert points to shape vector field
        point_coords = [x['point_coord'] for x in inputs]
        gt_star_field_list=[]
        for point_coord in point_coords:
            gt_star_field = Smoothed_VF(point_coord, batched_input[0]['origin_size'])
            gt_star_field_list.append(gt_star_field)
        gt_star_fields = torch.stack(gt_star_field_list, dim=0)
        if device == "cpu":
            output = model(inputs)
            mask = torch.stack([y["mask"] for y in output], dim=0)
            star_field_pre = torch.stack([y["vector_field"] for y in output], dim=0)
            bceloss = criterion(mask, gt2D)
            if if_ccs:
                star_field_pre = torch.stack([y["vector_field"] for y in output], dim=0)
                star_field_loss = 1 - cosine_loss(star_field_pre, gt_star_fields).mean()
                loss = 1*bceloss + 0.1* star_field_loss
                # the weights can be adjusted
            elif if_sloss:
                lss = shape_loss_batch(mask, gt_star_fields)
                loss = 1*bceloss + 0.1* lss
                # The weights can be adjusted.
            else:
                loss = bceloss
            loss.backward()
            optimizer.step()
        else:
            scaler = GradScaler()
            output = model(inputs)
            mask = torch.stack([y["mask"] for y in output], dim=0)
            bceloss = criterion(mask, gt2D)
            if if_ccs:
                star_field_pre = torch.stack([y["vector_field"] for y in output], dim=0)
                
                star_field_loss = 1 - cosine_loss(star_field_pre, gt_star_fields).mean()
                loss = 1*bceloss + 0.1* star_field_loss
                # the weights can be adjusted
            elif if_sloss:
                lss = shape_loss_batch(mask, gt_star_fields)
                loss = 1*bceloss + 0.1* lss
                # the weights can be adjusted
            else:
                loss = bceloss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

    return total_loss / len(train_loader)


def val(model, val_loader, criterion, calculatedice, device, if_ccs):
    model.eval()
    total_loss = 0
    total_dice = 0
    cosine_loss = nn.CosineSimilarity(dim=-1)

    with torch.no_grad():
        for batched_input in val_loader:
            gt2D = torch.stack([x["gt"] for x in batched_input], dim=0)
            gt2D = gt2D.to(device=device, dtype=torch.float32)
            inputs = to_device(batched_input, device)
            # point_coord = torch.stack([x['point_coord'] for x in inputs], dim=0)
            # point_coord = point_coord.squeeze(0)
            # gt_star_field = Smoothed_VF(point_coord, batched_input[0]['origin_size'])
            point_coords = [x['point_coord'] for x in inputs]
            gt_star_field_list=[]
            for point_coord in point_coords:
                gt_star_field = Smoothed_VF(point_coord, batched_input[0]['origin_size'])
                gt_star_field_list.append(gt_star_field)
            gt_star_fields = torch.stack(gt_star_field_list, dim=0)

            with autocast():
                output = model(inputs)
                mask = torch.stack([y["mask"] for y in output], dim=0)
                bceloss = criterion(mask, gt2D)
                if if_ccs:
                    star_field_pre = torch.stack([y["vector_field"] for y in output], dim=0)
                    star_field_loss = 1 - cosine_loss(star_field_pre, gt_star_fields).mean()
                    loss = bceloss + 0.1 * star_field_loss
                else:
                    loss = bceloss
                dice = calculatedice(mask, gt2D)
                total_loss += loss.item()
                total_dice += dice
    return total_loss / len(val_loader), total_dice / len(val_loader)


def setup_logging(log_file='training.log'):
    logging.basicConfig(level=logging.INFO)
    # create logger
    logger = logging.getLogger(__name__)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def custom_collate_fn(batch):
    return batch  # return batch list


def Lr_lambda(epoch, warm_up_steps=4, period=4):
    if epoch < warm_up_steps:
        return 0.8 ** (warm_up_steps - epoch)
    else:
        if (epoch - warm_up_steps) < period:
            return (1 + math.cos((epoch - warm_up_steps) * math.pi / period)) / 2
        else:
            return Lr_lambda(epoch - warm_up_steps - period, warm_up_steps=0, period=period * 2)


if __name__ == "__main__":
    main()
