# encoding: utf-8

import torch
from typing import *
from tqdm import tqdm
from math import log

Tensor = torch.Tensor


# import cv2

class DDIM:
    def __init__(self,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.002,
                 max_diffusion_step: int = 100,
                 device: str = 'cuda',
                 scale_mode: Literal["linear", "quadratic", "log"] = "linear",
                 skip_step=1,
                 data_dim: int = 2):

        if scale_mode == "quadratic":
            betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, max_diffusion_step).to(device) ** 2
        elif scale_mode == "log":
            betas = torch.exp(torch.linspace(log(min_beta), log(max_beta), max_diffusion_step).to(device))
        else:
            betas = torch.linspace(min_beta, max_beta, max_diffusion_step).to(device)

        self.skip_step = skip_step
        self.device = device

        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.T = max_diffusion_step

        expand_shape = [-1] + [1] * data_dim

        self.β = betas.view(*expand_shape)  # (T, 1, 1)
        self.α = alphas.view(*expand_shape)  # (T, 1, 1)
        self.αbar = alpha_bars.view(*expand_shape)  # (T, 1, 1)
        self.sqrt_αbar = torch.sqrt(alpha_bars).view(*expand_shape)  # (T, 1, 1)
        self.sqrt_1_m_αbar = torch.sqrt(1 - alpha_bars).view(*expand_shape)  # (T, 1, 1)

        self.σ = 0.0

    def diffusionForwardStep(self, x_t, t, ϵ_t_to_tp1):
        # t: 0 - T-1
        # For DDPM, x_t+1 = √(α_t) * x_t + √(1 - α_t) * ϵ_t:t+1
        # For DDIM, x_t+1 = √(αbar_t-2)
        return torch.sqrt(self.α[t]) * x_t + torch.sqrt(1 - self.α[t]) * ϵ_t_to_tp1

    def diffusionForward(self, x_0, t, ϵ):
        """
        Forward Diffusion Process
        :param x_0: input (B, C, L)
        :param t: time steps (B, )
        :param ϵ: noise (B, C, L)
        :return: x_t: output (B, C, L)
        """
        x_t = self.sqrt_αbar[t] * x_0 + self.sqrt_1_m_αbar[t] * ϵ
        return x_t

    def diffusionBackwardStep(self, x_tp1: Tensor, t: int, next_t: int, ϵ_pred: Tensor):
        """
        Backward Diffusion Process
        :param x_t: input images (B, C, L)
        :param t: time steps
        :param ϵ_pred: predicted noise (B, C, L)
        :param scaling_factor: scaling factor of noise
        :return: x_t-1: output images (B, C, L)
        """
        pred_x0 = (x_tp1 - self.sqrt_1_m_αbar[t] * ϵ_pred) / self.sqrt_αbar[t]
        if t <= self.skip_step * 3:
            return pred_x0
        return self.diffusionForward(pred_x0, next_t, ϵ_pred)

    def diffusionBackwardStepWithx0(self, pred_x0, t, next_t, ϵ_pred: Tensor):
        """
        Backward Diffusion Process
        :param x_t: input images (B, C, L)
        :param t: time steps
        :param ϵ_pred: predicted noise (B, C, L)
        :param scaling_factor: scaling factor of noise
        :return: x_t-1: output images (B, C, L)
        """
        if isinstance(t, int):
            t = torch.tensor(t, device=self.device)
        if isinstance(next_t, int):
            next_t = torch.tensor(next_t, device=self.device)
        # return self.diffusionForward(pred_x0, next_t, ϵ_pred)
        mask = (t <= max(self.skip_step, 5)).to(torch.long).view(-1, 1, 1)
        return pred_x0 * mask + self.diffusionForward(pred_x0, next_t, ϵ_pred) * (1 - mask)

    @torch.no_grad()
    def diffusionBackward(self,
                          noises: List[Tensor],
                          pred_func: Callable,
                          mode: Literal["eps", "x0"] = "x0",
                          verbose=False,
                          **pred_func_args):
        """
        Backward Diffusion Process
        :param unet: UNet
        :param input_T: input images (B, 6, L)
        :param s: initial state (B, 32, L//4)
        :param E: mix context (B, 32, L//4)
        :param mask: mask (B, 1, L), 1 for erased, 0 for not erased, -1 for padding
        :param query_len: query length (B, )
        """
        B = noises[0].shape[0]
        content_list = [noise.clone() for noise in noises]
        tensor_t = torch.arange(self.T, dtype=torch.long, device=self.device).repeat(B, 1)  # (B, T)
        t_list = list(range(self.T - 1, -1, -self.skip_step))
        if t_list[-1] != 0:
            t_list.append(0)
        pbar = tqdm(t_list) if verbose else t_list
        for ti, t in enumerate(pbar):
            if mode == "eps":
                noise_preds = pred_func(content_list, tensor_t[:, t], **pred_func_args)
                t_next = 0 if ti + 1 == len(t_list) else t_list[ti + 1]

                for i in range(len(content_list)):
                    content_list[i] = self.diffusionBackwardStep(content_list[i], t, t_next, noise_preds[i])
            else:
                x0_preds, noise_preds = pred_func(content_list, tensor_t[:, t], **pred_func_args)
                t_next = 0 if ti + 1 == len(t_list) else t_list[ti + 1]
                for i in range(len(content_list)):
                    content_list[i] = self.diffusionBackwardStepWithx0(x0_preds[i], t, t_next, noise_preds[i])
        return content_list