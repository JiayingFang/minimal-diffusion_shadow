# Copyright (c) OpenMMLab. All rights reserved.
# Modified from mmagic from OpenMMLab
from enum import Enum

import torch
import torch.nn as nn
from unets import AttentionBlock, normalization
from torch import Tensor

AttentionStatus = Enum('ATTENTION_STATUS', 'READ WRITE DISABLE')


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class AttentionInjection(nn.Module):
    """Wrapper for stable diffusion unet.

    Args:
        module (nn.Module): The module to be wrapped.
    """

    def __init__(self, module: nn.Module, injection_weight=1):
        super().__init__()
        self.attention_status = AttentionStatus.READ
        self.style_cfgs = []
        self.unet = module

        attn_inject = self

        def transformer_forward_replacement(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ):
            b, c, *spatial = hidden_states.shape
            hidden_states = hidden_states.reshape(b, c, -1)
            norm_hidden_states = self.norm(hidden_states)

            attn_output = None
            self_attention_context = norm_hidden_states
            if attn_inject.attention_status == AttentionStatus.WRITE:
                self.bank.append(self_attention_context.detach().clone())
            if attn_inject.attention_status == AttentionStatus.READ:
                if len(self.bank) > 0:
                    self.bank = self.bank * injection_weight
                    attn_output = self.qkv(norm_hidden_states)
                    # replace the qk of self-attention
                    attn_output[:, :2*self.channels] = self.qkv(self.bank[0])[:, :2*self.channels]
                    attn_output = self.attention(attn_output)
                    attn_output = self.proj_out(attn_output)
                self.bank = []
            if attn_output is None:
                attn_output = attn_output = self.qkv(norm_hidden_states)
                attn_output = self.attention(attn_output)
                attn_output = self.proj_out(attn_output)

            hidden_states = attn_output + hidden_states

            return hidden_states.reshape(b, c, *spatial)

        all_modules = torch_dfs(self.unet)

        attn_modules = [
            module for module in all_modules
            if isinstance(module, AttentionBlock)
        ]
        for i, module in enumerate(attn_modules):
            module.forward = transformer_forward_replacement.__get__(
                module, AttentionBlock)
            module.bank = []

    def forward(self,
                x: Tensor,
                t,
                ref_x,
                y) -> Tensor:
        """Forward and add LoRA mapping.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if ref_x is not None:
            self.attention_status = AttentionStatus.WRITE
            self.unet(
                ref_x,
                t,y)
        self.attention_status = AttentionStatus.READ
        output = self.unet(
            x,
            t,y)

        return output