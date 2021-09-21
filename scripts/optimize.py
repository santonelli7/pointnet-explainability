import json
import argparse
from os.path import join

from models.input_optimization import InputOptimization

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

class WandbCallback(pl.Callback):
    """Logs the input pointclouds and output predictions of a module.
    
    Predictions and targets are logged as class indices."""
    
    def __init__(self, config):
        super().__init__()
        dataset_name = config['dataset'].lower()
        if dataset_name == 'shapenet':
            from datasets.shapenet import ShapeNetDataset as ShapeNet
            self.classes = list(ShapeNet.synth_id_to_category.values())
        elif dataset_name == 'modelnet':
            from datasets.modelnet import ModelNet40 as ModelNet
            self.classes = ModelNet.all_classes
        else:
            raise ValueError(f'Invalid dataset name. Expected `shapenet` or 'f'`modelnet`. Got: `{dataset_name}`')

    def on_epoch_end(self, trainer, pl_module):
        points = pl_module.random_noise_input
        log_probs = self._get_prediction(pl_module, points)

        wandb.log({'pointcloud': wandb.Object3D(points[0].detach().cpu().squeeze().transpose(0,1).numpy())})
        
        if trainer.current_epoch % 1000 == 0:
            table = self._get_class_distribution(log_probs)
            wandb.log({"class_dist" : wandb.plot.bar(table, "class", "probability", title="Class Distribution")})
    
    def on_train_end(self, trainer, pl_module):
        points = pl_module.random_noise_input
        log_probs = self._get_prediction(pl_module, points)
        table = self._get_class_distribution(log_probs)
        wandb.log({"class_dist" : wandb.plot.bar(table, "class", "probability", title="Class Distribution")})

    def _get_prediction(self, pl_module, points):
        pl_module.encoder.eval()
        pl_module.generator.eval()
        pl_module.pointnet.eval()
        with torch.no_grad():
            code, _, _ = pl_module.encoder(points)
            gen_points = pl_module.generator(code)
            log_probs, _, _ = pl_module.pointnet(gen_points)
        return log_probs
    
    def _get_class_distribution(self, log_probs):
        values = torch.exp(log_probs.squeeze()).tolist()
        data = [[label, val] for (label, val) in zip(self.classes, values)]
        table = wandb.Table(data=data, columns = ["class", "probability"])
        return table


def optimization_loop(config):
    pl.seed_everything(config['seed'])

    model = InputOptimization(config)

    wandb_logger = WandbLogger(id=config['run_id'], project=config['project_name'], log_model='all')
    early_stopping  = EarlyStopping(monitor="loss", stopping_threshold=config["threshold"])

    if (not config['ckpts_dir'] is None) and (not config['project_name'] is None) and (not config['run_id']is None) and (not config['last_ckpt'] is None):
        ckpt = join(config['ckpts_dir'], config['project_name'], config['run_id'], "checkpoints", config['last_ckpt'])
    else:
        ckpt = None

    trainer = pl.Trainer(
        default_root_dir=config['ckpts_dir'],
        gpus=config['gpu'],
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        callbacks=[early_stopping, WandbCallback(config)],
        resume_from_checkpoint=ckpt,
        log_every_n_steps=1,
        )

    trainer.fit(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    optimization_loop(config)
