from TrainEvalTest.W2G_VAE.train import train as train_w2gvae
from TrainEvalTest.T2W_DiT.train import train as train_t2wdit

dataset = "LasVegas"

default_params = {
    "dataset_path": f"Dataset/{dataset}",
    "lr": 2e-4,
    "lr_reduce_factor": 0.5,
    "lr_reduce_patience": 20,
    "lr_reduce_min": 1e-6,
    "lr_reduce_threshold": 1e-7,
    "epochs": 1000,
    "B": 32,
    "mov_avg_len": 5,
    "log_interval": 10,
}

### For fine-tunning
# default_params["lr"] = 1e-4

vae_params = {k: v for k, v in default_params.items()}
vae_params["lr"] = 1e-4
vae_params["kl_weight"] = 1e-6

diffusion_params = {k: v for k, v in default_params.items()}
diffusion_params["T"] = 500
diffusion_params["beta_min"] = 0.0001
diffusion_params["beta_max"] = 0.05
diffusion_params["eval_interval"] = 10


if __name__ == "__main__":
    print("Start Training RGVAE")   # ------------------------------------------- RGVAE
    rgvae_path = train_w2gvae(
        title=dataset,
        **vae_params,
        load_weights="Runs/RGVAE/241222_2243_LasVegas/last.pth"
    )

    print("Start Training TRDiT")   # ------------------------------------------- TRDiT
    train_t2wdit(
        title=dataset,
        **diffusion_params,
        vae_path=rgvae_path,
        load_weights="Runs/TRDiT/241219_0901_Shanghai/last.pth"
    )
