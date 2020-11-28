import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(config_path="configs", config_name="config")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    train_dir


if __name__ == "__main__":
    run()
