from TrainEvalTest.Utils import *
from TrainEvalTest.T2W_DiT.eval import getEvalFunction
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import T2W_DiT, W2G_VAE
from Diffusion import DDIM

def train(
        title: str = "295M",
        dataset_path: str = "",
        lr: float = 2e-4,
        lr_reduce_factor: float = 0.5,
        lr_reduce_patience: int = 30,
        lr_reduce_min: float = 1e-7,
        lr_reduce_threshold: float = 1e-5,
        epochs: int = 1000,
        B: int = 32,
        T: int = 500,
        mov_avg_len: int = 5,
        log_interval: int = 10,
        eval_interval: int = 10,
        beta_min: float = 0.0001,
        beta_max: float = 0.05,
        vae_path: str = "",
        load_weights: str = None
):
    log_dir = f"./Runs/T2W_DiT/{datetime.now().strftime('%Y%m%d_%H%M')[2:]}_{title}/"
    # Dataset & DataLoader
    dataset = RoadNetworkDataset(folder_path=dataset_path,
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 enable_aug=True,
                                 img_H=16,
                                 img_W=16
                                 )

    # Models
    vae = W2G_VAE(N_routes=dataset.N_trajs, L_route=dataset.max_L_route,
                  N_interp=dataset.N_interp, threshold=0.5).to(DEVICE)
    loadModels(vae_path, vae=vae)
    vae.eval()

    DiT = T2W_DiT(D_in=dataset.N_interp * 2,
                  N_routes=dataset.N_trajs,
                  L_route=dataset.max_L_route,
                  L_traj=dataset.max_L_traj,
                  d_context=2,
                  n_layers=8,
                  T=T)

    if load_weights is not None:
        loadModels(load_weights, DiT=DiT)

    # torch.set_float32_matmul_precision("high")
    # torch.compile(DiT)

    DiT = DiT.to(DEVICE)

    eval = getEvalFunction(dataset_path, vae)

    ddim = DDIM(beta_min, beta_max, T, DEVICE, "quadratic", skip_step=10, data_dim=3)
    loss_func = torch.nn.MSELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(DiT.parameters(), lr=lr, amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=lr_reduce_factor, patience=lr_reduce_patience,
                                     min_lr=lr_reduce_min, threshold=lr_reduce_threshold)

    # Prepare Logging
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    mov_avg_loss = MovingAvg(mov_avg_len * len(dataset))
    global_step = 0
    best_loss = float("inf")

    with ProgressManager(len(dataset), epochs, 5, 2, ["Loss", "lr"]) as progress:
        for e in range(epochs):
            total_loss = 0
            for i, batch in enumerate(dataset):
                batch: Dict[str, Tensor]

                with torch.no_grad():
                    latent, _ = vae.encode(batch["routes"])    # (B, N_trajs, L_route, N_interp*2)

                latent_noise = torch.randn_like(latent)

                t = torch.randint(0, T, (B,)).to(DEVICE)
                latent_noisy = ddim.diffusionForward(latent, t, latent_noise)

                latent_noise_pred = DiT(latent_noisy, batch["trajs"], t)

                loss = loss_func(latent_noise_pred, latent_noise) * 100
                loss.backward()
                torch.nn.utils.clip_grad_norm_(DiT.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                global_step += 1
                mov_avg_loss.update(loss.item())

                progress.update(e, i, Loss=mov_avg_loss.get(), lr=optimizer.param_groups[0]['lr'])

                if global_step % log_interval == 0:
                    writer.add_scalar("loss/Loss", mov_avg_loss.get(), global_step)
                    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

            if e % eval_interval == 0:
                figure, eval_loss = eval(DiT, ddim)
                writer.add_figure("Evaluation", figure, global_step)
                writer.add_scalar("loss/eval", eval_loss, global_step)

            saveModels(log_dir + "last.pth", DiT=DiT)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(log_dir + "best.pth", DiT=DiT)

            lr_scheduler.step(total_loss)

            if optimizer.param_groups[0]["lr"] <= lr_reduce_min:
                # Stop training if learning rate is too low
                break

    return os.path.join(log_dir, "last.pth")

if __name__ == "__main__":
    train()
