from os.path import join
import json
import argparse

from datasets.modelnet import ModelNet40
from models.pointnet import PointNet
from datasets.datamodule import LitDataModule

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb

class WandbCallback(pl.Callback):
    """Logs the input pointclouds and output predictions of a module.
    
    Predictions and targets are logged as class indices."""
    
    def __init__(self, config):
        super().__init__()
        test = ModelNet40(root_dir=join(config['data_dir'], config['dataset']), split='test')
        samples = next(iter(torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)))
        self.points, self.targets = samples
        self.targets = self.targets.squeeze()
        self.points = self.points.transpose(2, 1)

    def on_test_end(self, trainer, pl_module):
        points = self.points.to(device=pl_module.device)
        log_logits, _, _ = pl_module(points)
        preds = torch.argmax(log_logits, dim=-1)

        my_table = wandb.Table(columns=["is", "pointcloud", "target", "prediction"])
        for idx, pointcloud in enumerate(points):
            my_table.add_data(idx, wandb.Object3D(pointcloud.detach().cpu().T.numpy()), ModelNet40.number_to_category[self.targets[idx].item()], ModelNet40.number_to_category[preds[idx].item()])
        
        wandb.log({"ModelNet40_predictions": my_table})


if __name__ == '__main__':

    # ------------
    # args
    # ------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    pl.seed_everything(config['seed'])

    datamodule = LitDataModule(data_dir=join(config['data_dir'], config['dataset']), config=config)

    model = PointNet(config=config)

    wandb_logger = WandbLogger(id=config['run_id'], project=config['project_name'], log_model='all')

    if (not config['ckpts_dir'] is None) and (not config['project_name'] is None) and (not config['run_id']is None) and (not config['last_ckpt'] is None):
        ckpt = join(config['ckpts_dir'], config['project_name'], config['run_id'], "checkpoints", config['last_ckpt'])
    else:
        ckpt = None

    trainer = pl.Trainer(
        default_root_dir=config['ckpts_dir'],
        gpus=config['gpu'],
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        callbacks=[WandbCallback(config)],
        resume_from_checkpoint=ckpt,
        )

    # log gradients and model topology
    wandb_logger.watch(model)

    trainer.fit(model, datamodule)

    trainer.test(model, datamodule=datamodule)
