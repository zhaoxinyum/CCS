
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
# from safetensors import safe_open
# from safetensors.torch import save_file, load_file
import yaml

class LoRA_qkv(nn.Module):
    """
    LoRA adaption for attention modules. Only for queries and values

    Arguments:
        qkv: Original block of attention
        linear_a_q: linear block for q
        linear_b_q: linear block for q
        linear_a_v: linear block for v
        linear_b_v: linear block for v

    Return:
        qkv(nn.Module): qkv block with all linear blocks added (equivalent to adding the matrix B*A)
    """

    def __init__(
            self,
            qkv,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        
        super(LoRA_qkv, self).__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x: Tensor):
        qkv = self.qkv(x)
        q_ba = self.linear_b_q(self.linear_a_q(x))
        v_ba = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.d_model] += q_ba #q part
        qkv[:, :, :, -self.d_model:] += v_ba #v part

        return qkv


class LoRA_sam(nn.Module):
    """
    Class that takes the image encoder of SAM and add the lora weights to the attentions blocks

    Arguments:
        sam_model: Sam class of the segment anything model
        rank: Rank of the matrix for LoRA
        lora_layer: List of weights exisitng for LoRA
    
    Return:
        None

    """

    def __init__(self, sam_model , rank: int, lora_layer=None):
        super(LoRA_sam, self).__init__()
        self.rank = rank
        assert rank > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            # In each block, you have an attention block => total blocks -> nb lora layers
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))
        
        self.A_weights = []
        self.B_weights = []

        # freeze parameters of the image encoder
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # if only lora on few layers
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.d_model = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(self.d_model, self.rank, bias=False)
            w_b_linear_q = nn.Linear(self.rank, self.d_model, bias=False)
            w_a_linear_v = nn.Linear(self.d_model, self.rank, bias=False)
            w_b_linear_v = nn.Linear(self.rank, self.d_model, bias=False)
            

            self.A_weights.append(w_a_linear_q)
            self.B_weights.append(w_b_linear_q)
            self.A_weights.append(w_a_linear_v)
            self.B_weights.append(w_b_linear_v)

            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v
            )

        self.reset_parameters()
        self.sam = sam_model
        self.lora_vit = sam_model.image_encoder


    def reset_parameters(self):
        """
        Initialize the LoRA A and B matrices like in the paper
        """
        # Initalisation like in the paper
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)

    # def save_lora_parameters(self, filename):
    #     merged_dict = {f'A_weight_{i}': w_A.weight.detach().cpu() for i, w_A in enumerate(self.A_weights)}
    #     merged_dict.update({f'B_weight_{i}': w_B.weight.detach().cpu() for i, w_B in enumerate(self.B_weights)})
    #     save_file(merged_dict, filename)
    #     print(f'LoRA parameters saved to {filename}')
    #
    # def load_lora_parameters(self, filename):
    #     loaded_dict = load_file(filename)
    #     for i, w_A in enumerate(self.A_weights):
    #         w_A.weight = nn.Parameter(loaded_dict[f'A_weight_{i}'])
    #     for i, w_B in enumerate(self.B_weights):
    #         w_B.weight = nn.Parameter(loaded_dict[f'B_weight_{i}'])
    #     print(f'LoRA parameters loaded from {filename}')
    def save_lora_parameters(self, filename):
        merged_dict = {f'A_weight_{i}': w_A.weight.detach().cpu() for i, w_A in enumerate(self.A_weights)}
        merged_dict.update({f'B_weight_{i}': w_B.weight.detach().cpu() for i, w_B in enumerate(self.B_weights)})
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename):
        loaded_dict = torch.load(filename)
        for i, w_A in enumerate(self.A_weights):
            w_A.weight = nn.Parameter(loaded_dict[f'A_weight_{i}'].to(w_A.weight.device))
        for i, w_B in enumerate(self.B_weights):
            w_B.weight = nn.Parameter(loaded_dict[f'B_weight_{i}'].to(w_B.weight.device))


