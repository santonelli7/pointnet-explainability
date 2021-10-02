from os.path import join
import json
import argparse

from models.aae import AAE
from datasets.datamodule import LitDataModule

import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


class WandbCallback(pl.Callback):
    """Logs the input pointclouds and output predictions of a module.
    
    Predictions and targets are logged as class indices."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # if torch.cuda.is_available():
        #     self.fixed_noise = torch.cuda.FloatTensor(self.config['num_classes'], self.config['z_size'], 1)
        # else:
        #     self.fixed_noise = torch.FloatTensor(self.config['num_classes'], self.config['z_size'], 1)
        # self.fixed_noise.normal_(mean=self.config['normal_mu'], std=self.config['normal_std'])

    def on_epoch_end(self, trainer, pl_module):
        samples = trainer.datamodule.test_samples
        X, _ = samples
        X = X.to(pl_module.device)

        if X.size(-1) == 3:
            X.transpose_(X.dim() - 2, X.dim() - 1)

        # fake = pl_module(self.fixed_noise).data
        codes, _, _ = pl_module.encoder(X)
        X_rec = pl_module(codes).data

        for k in range(len(X)):

            if self.config['dataset'].lower() == 'shapenet':
                from datasets.shapenet import ShapeNetDataset as ShapeNet
                cls = ShapeNet.synth_id_to_category[ShapeNet.number_to_synth_id[k]]
            elif self.config['dataset'].lower() == 'modelnet':
                from datasets.modelnet import ModelNet40 as ModelNet
                cls = ModelNet.number_to_category[k]
            else:
                raise ValueError(f'Invalid dataset name. Expected `shapenet` or `modelnet`. Got: `{self.config["dataset"].lower()}`')

            if trainer.current_epoch == 0:
                wandb.log({f'{cls}_real': wandb.Object3D(X[k].T.cpu().numpy())})
            # wandb.log({f'{cls}_fixed': wandb.Object3D(fake[k].T.cpu().numpy())})
            wandb.log({f'{cls}_reconstructed': wandb.Object3D(X_rec[k].T.cpu().numpy())})

def main(config):
    pl.seed_everything(config['seed'])

    datamodule = LitDataModule(data_dir=join(config['data_dir'], config['dataset']), config=config)

    model = AAE(config)

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

    main(config)
