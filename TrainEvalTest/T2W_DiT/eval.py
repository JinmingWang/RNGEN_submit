from Dataset import RoadNetworkDataset
from TrainEvalTest.Utils import *
from Models import W2G_VAE
from Diffusion import DDIM

from typing import Callable

import torch


def pred_func(noisy_contents: List[Tensor], t: Tensor, model: torch.nn.Module, trajs: Tensor):
    pred = model(*noisy_contents, trajs, t)
    return [pred]


def getEvalFunction(dataset_path: str, vae: W2G_VAE) -> Callable:
    """
    Evaluate the model on the given batch
    :param vae: The VAE model
    :return: The figure and loss
    """
    test_set = RoadNetworkDataset(dataset_path,
                                 batch_size=10,
                                 drop_last=True,
                                 set_name="test",
                                 enable_aug=False,
                                 img_H=16,
                                 img_W=16
                                 )

    batch = test_set[0:10]

    with torch.no_grad():
        latent, _ = vae.encode(batch["routes"])

    latent_noise = torch.randn_like(latent)

    plot_manager = PlotManager(4, 1, 5)

    def eval(DiT, ddim: DDIM) -> Tuple[plt.Figure, Tensor]:

        DiT.eval()

        with torch.no_grad():
            latent_pred = ddim.diffusionBackward([latent_noise], pred_func, mode="eps", model=DiT, trajs=batch["trajs"])[0]
            duplicate_segs, cluster_mat, cluster_means, coi_means = vae.decode(latent_pred)

        loss = torch.nn.functional.mse_loss(duplicate_segs, batch["routes"].flatten(1, 2))

        plot_manager.plotSegments(batch["routes"][0], 0, 0, "Routes", color="red")
        plot_manager.plotSegments(batch["segs"][0], 0, 1, "Segs", color="blue")
        plot_manager.plotSegments(coi_means[0].detach(), 0, 2, "Pred Segs", color="green")
        plot_manager.plotSegments(duplicate_segs[0].detach(), 0, 3, "Pred Duplicate Segs")
        plot_manager.plotTrajs(batch["trajs"][0], 0, 4, "Trajectories")

        DiT.train()

        return plot_manager.getFigure(), loss.item()

    return eval
