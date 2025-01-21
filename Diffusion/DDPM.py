import torch
from typing import *
from tqdm import tqdm
from math import log

Tensor = torch.Tensor


class DDPM:
    def __init__(self,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.002,
                 max_diffusion_step: int = 100,
                 device: str = 'cuda',
                 scale_mode: Literal["linear", "quadratic", "log"] = "linear"):

        if scale_mode == "quadratic":
            betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, max_diffusion_step).to(device) ** 2
        elif scale_mode == "log":
            betas = torch.exp(torch.linspace(log(min_beta), log(max_beta), max_diffusion_step).to(device))
        else:
            betas = torch.linspace(min_beta, max_beta, max_diffusion_step).to(device)

        self.device = device

        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.T = max_diffusion_step
        self.b = betas.view(-1, 1, 1)  # (T, 1, 1)
        self.a = alphas.view(-1, 1, 1)  # (T, 1, 1)
        self.abar = alpha_bars.view(-1, 1, 1)  # (T, 1, 1)
        self.sqrt_abar = torch.sqrt(alpha_bars).view(-1, 1, 1)  # (T, 1, 1)
        self.sqrt_1_m_abar = torch.sqrt(1 - alpha_bars).view(-1, 1, 1)  # (T, 1, 1)

    def diffusionForward(self, x_0, t, eps_0_to_tp1):
        """
        Forward Diffusion Process
        :param x_0: input (B, C, L)
        :param t: time steps (B, )
        :param epsilon: noise (B, C, L)
        :return: x_t: output (B, C, L)
        """
        x_tp1 = self.sqrt_abar[t] * x_0 + self.sqrt_1_m_abar[t] * eps_0_to_tp1
        return x_tp1

    def diffusionForwardStep(self, x_t, t, eps_t_to_tp1):
        # t: 0 - T-1
        return torch.sqrt(self.a[t]) * x_t + torch.sqrt(1 - self.a[t]) * eps_t_to_tp1

    def diffusionBackwardStep(self, x_tp1: torch.Tensor, t: int, epsilon_pred: torch.Tensor, disable_noise: bool = False):
        """
        Backward Diffusion Process
        :param x_t: input images (B, C, L)
        :param t: time steps
        :param epsilon_pred: predicted noise (B, C, L)
        :param scaling_factor: scaling factor of noise
        :return: x_t-1: output images (B, C, L)
        """
        if isinstance(t, int):
            t = torch.tensor(t).to(x_tp1.device)

        sample_dims = x_tp1.dim() - 1
        beta = self.b[t]#.view(-1, *([1] * sample_dims))
        alpha = self.a[t]#.view(-1, *([1] * sample_dims))
        sqrt_1_minus_alpha_bar = self.sqrt_1_m_abar[t]#.view(-1, *([1] * sample_dims))

        mu = (x_tp1 - beta / sqrt_1_minus_alpha_bar * epsilon_pred) / torch.sqrt(alpha)

        if disable_noise:
            return mu

        stds = torch.sqrt((1 - self.abar[t - 1]) / (1 - self.abar[t]) * beta) * torch.randn_like(x_tp1) * 0.2
        stds = stds * (t.view(-1, *([1] * sample_dims)) > 5).to(torch.long)

        return mu + stds

    @torch.no_grad()
    def diffusionBackward(self,
                          noises: List[Tensor],
                          pred_func: Callable,
                          verbose=False):
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
        pbar = tqdm(range(self.T - 1, -1, -1)) if verbose else range(self.T - 1, -1, -1)
        for t in pbar:
            noise_preds = pred_func(content_list, tensor_t[:, t])

            for i in range(len(content_list)):
                content_list[i] = self.diffusionBackwardStep(content_list[i], t, noise_preds[i])

        return content_list
