import os

import torch
import wandb
from omegaconf import OmegaConf

from torch_3dgs.data import read_data
from torch_3dgs.trainer import Trainer
from torch_3dgs.model import GaussianModel
from torch_3dgs.point import get_point_clouds
from torch_3dgs.utils import dict_to_device


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    os.makedirs(config.output_folder, exist_ok=True)
    device = torch.device(config.device)

    data = read_data(config.data_folder, resize_scale=config.resize_scale)
    data = dict_to_device(data, device)

    data_size = data['camera'].shape[0]
    assert data_size >= config.num_training_data

    data_indices = list(range(0, data_size, data_size // config.num_training_data))

    if len(data_indices) < config.num_training_data:
        assert config.num_training_data == len(data_indices) + 1
        data_indices.append(data_size - 1)

    points = get_point_clouds(
        data["camera"][data_indices],
        data["depth"][data_indices],
        data["alpha"][data_indices],
        data["rgb"][data_indices],
    )
    raw_points = points.random_sample(config.num_points)
    model = GaussianModel(sh_degree=4, debug=False)
    model.create_from_pcd(pcd=raw_points)

    wandb.init(
        project="EV-HW1",
        config=OmegaConf.to_container(config, resolve=True),
    )

    trainer = Trainer(
        data=data,
        model=model, 
        device=device,
        num_steps=config.num_steps,
        eval_interval=config.eval_interval,
        l1_weight=config.l1_weight,
        dssim_weight=config.dssim_weight,
        depth_weight=config.depth_weight,
        lr=config.lr,
        results_folder=config.output_folder,
        render_kwargs={
            "tile_size": config.render.tile_size,
        },
        logger=wandb,
    )
    trainer.train()
