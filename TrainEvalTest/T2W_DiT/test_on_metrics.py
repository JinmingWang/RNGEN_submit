from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from Diffusion import DDIM
from tqdm import tqdm
import cv2

import torch

from Dataset import DEVICE, RoadNetworkDataset
from Models import W2G_VAE, T2W_DiT


def segsToHeatmaps(batch_segs: Tensor, batch_trajs: Tensor, traj_lens: Tensor, img_H: int, img_W: int, line_width: int):
    # segs: B * (N_segs, N_interp, 2)
    B = len(batch_segs)

    heatmaps = []

    for i in range(B):
        # Get the bounding box of the segment
        trajs = batch_trajs[i]
        L_traj = traj_lens[i]
        points = torch.cat([trajs[i, :L_traj[i]] for i in range(48)], dim=0)
        min_point = torch.min(points, dim=0, keepdim=True).values
        max_point = torch.max(points, dim=0, keepdim=True).values
        point_range = max_point - min_point

        segs = batch_segs[i]  # (N_segs, N_interp, 2)
        segs = segs[torch.all(segs.flatten(1) != 0, dim=1)]

        segs = (segs - min_point.view(1, 1, 2)) / point_range.view(1, 1, 2)

        segs[..., 0] *= img_W
        segs[..., 1] *= img_H

        heatmap = np.zeros((1, img_H, img_W), dtype=np.float32)

        lons = segs[:, :, 0].cpu().numpy().astype(np.int32)
        lats = segs[:, :, 1].cpu().numpy().astype(np.int32)

        for j in range(len(segs)):
            # Draw the polyline
            cv2.polylines(heatmap[0], [np.stack([lons[j], lats[j]], axis=1)], False,
                          1.0, thickness=line_width)

        heatmaps.append(torch.tensor(heatmap))

    return torch.stack(heatmaps, dim=0).to(DEVICE)


def pred_func(noisy_contents: List[Tensor], t: Tensor, model: torch.nn.Module, trajs: Tensor):
    pred = model(*noisy_contents, trajs, t)
    return [pred]


def test(
        T=500,
        beta_min = 0.0001,
        beta_max = 0.05,
        data_path = "",
        vae_path = "",
        model_path = ""
):
    B = 50
    dataset = RoadNetworkDataset(folder_path=data_path,
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="test",
                                 permute_seq=False,
                                 enable_aug=False,
                                 shuffle=False,
                                 img_H=256,
                                 img_W=256,
                                 need_heatmap=True
                                 )

    vae = W2G_VAE(N_routes=dataset.N_trajs, L_route=dataset.max_L_route,
                  N_interp=dataset.N_interp, threshold=0.7).to(DEVICE)
    loadModels(vae_path, vae=vae)
    vae.eval()

    DiT = T2W_DiT(D_in=dataset.N_interp * 2,
                  N_routes=dataset.N_trajs,
                  L_route=dataset.max_L_route,
                  L_traj=dataset.max_L_traj,
                  d_context=2,
                  n_layers=8,
                  T=T).to(DEVICE)
    loadModels(model_path, DiT=DiT)
    DiT.eval()

    ddim = DDIM(beta_min, beta_max, T, DEVICE, "quadratic", skip_step=10, data_dim=3)

    titles = ["hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse", "diff_seg_len"]

    name = "GraphWalker"

    with open(f"Report_{name}.csv", "w") as f:
        f.write(",".join(titles) + "\n")
        for batch in tqdm(dataset, desc="Testing"):

            with torch.no_grad():
                latent, _ = vae.encode(batch["routes"])
                latent_noise = torch.randn_like(latent)
                latent_pred = ddim.diffusionBackward([latent_noise], pred_func, mode="eps", model=DiT, trajs=batch["trajs"])[0]
                duplicate_segs, cluster_mat, cluster_means, coi_means = vae.decode(latent_pred)

            batch_scores = reportAllMetrics(coi_means, [batch["segs"][b][:batch["N_segs"][b]] for b in range(B)])

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(",".join([f"{s}" for s in scores]) + "\n")


if __name__ == "__main__":
    test()