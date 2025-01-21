from TrainEvalTest.T2W_DiT.test_on_metrics import test


weights = {
    "Tokyo": {
        "W2G_VAE": "",
        "T2W_DiT": "",
    },
    "Shanghai": {
        "W2G_VAE": "",
        "T2W_DiT": "",
    },
    "LasVegas": {
        "W2G_VAE": "",
        "T2W_DiT": "",
    }
}


if __name__ == "__main__":
    weight_dataset = "LasVegas"
    load_dataset = "LasVegas"

    print(f"Start Testing T2W_DiT on {load_dataset}")
    test(
        data_path=f"Dataset/{load_dataset}",
        model_path=weights[weight_dataset]["T2W_DiT"],
        vae_path=weights[weight_dataset]["W2G_VAE"]
    )



