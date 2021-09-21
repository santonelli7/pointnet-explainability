import json
from os.path import join

from models.pointnet import PointNet
from models.aae import AAE

import torch
import torch.nn as nn

import pytorch_lightning as pl

class InputOptimization(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # MODELS
        self.pointnet = PointNet.load_from_checkpoint(join(config['ckpts_dir'], config['pointnet_ckpt']), map_location=self.device)
        self.pointnet.freeze()

        with open(join("settings", config["aae_config"])) as f:
            aae_config = json.load(f)
        aae = AAE.load_from_checkpoint(join(config['ckpts_dir'], config['aae_ckpt']), config=aae_config, map_location=self.device)
        aae.freeze()
        self.encoder = aae.encoder
        self.generator = aae.generator

        # LOSS
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

        # INPUT
        self.random_noise_input = torch.randn((1, 3, self.config["num_points"]))
        if torch.cuda.is_available():
            self.random_noise_input = self.random_noise_input.cuda()
        self.random_noise_input.requires_grad = True

        # GROUND TRUTH
        dataset_name = self.config['dataset'].lower()
        if dataset_name == 'shapenet':
            from datasets.shapenet import ShapeNetDataset as ShapeNet
            cls_idx = ShapeNet.category_to_synth_id[ShapeNet.synth_id_to_number[config['expected_class']]]
            classes_idx = list(ShapeNet.synth_id_to_number.values())
        elif dataset_name == 'modelnet':
            from datasets.modelnet import ModelNet40 as ModelNet
            cls_idx = ModelNet.category_to_number[config['expected_class']]
            classes_idx = list(ModelNet.category_to_number.values())
        else:
            raise ValueError(f'Invalid dataset name. Expected `shapenet` or `modelnet`. Got: `{dataset_name}`')

        self.gt_class = (torch.Tensor(classes_idx) == cls_idx).unsqueeze(dim=0).to(torch.float32)
        if torch.cuda.is_available():
            self.gt_class = self.gt_class.cuda()

        self.save_hyperparameters(config)

    def _models_sanity_check(self):
        assert not self.gt_class.requires_grad, "Ground truth does not require gradients"
        assert self.random_noise_input.requires_grad, "Input does require gradients"

        assert (not self.pointnet.training and not self.encoder.training and not self.generator.training), "The models need to be in evaluation mode"

        # PointNet must be freezed
        for param in self.pointnet.parameters():
            assert not param.requires_grad, "The comparator model must be frozen"

        # Encoder must be freezed
        for param in self.encoder.parameters():
            assert not param.requires_grad, "The encoder model must be frozen"

        # Generator must be freezed
        for param in self.generator.parameters():
            assert not param.requires_grad, "The generator model must be frozen"

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(self.random_noise_input)
        return torch.utils.data.DataLoader(dataset, batch_size=1)

    def forward(self, x):
        return

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.config['optimizer']['type'])
        optim = optim([self.random_noise_input],
                        **self.config['optimizer']['hyperparams'])
        return optim

    def training_step(self, batch, batch_idx):
        self.pointnet.eval()
        self.encoder.eval()
        self.generator.eval()
        self._models_sanity_check()

        code, _, _ = self.encoder(self.random_noise_input)
        gen_points = self.generator(code)
        log_probs, _, _ = self.pointnet(gen_points)
        loss = self.kl_div(log_probs, self.gt_class)
        metrics = {'loss': loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrics
