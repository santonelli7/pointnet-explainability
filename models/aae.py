from os.path import join
from itertools import chain

from models.pointnet import PointNet

import torch
import torch.nn as nn

import pytorch_lightning as pl

def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['G']['use_bias']
        self.relu_slope = config['model']['G']['relu_slope']
        self.num_points = config['num_points']
        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=self.num_points * 3, bias=self.use_bias),
        )

    def forward(self, input):
        output = self.model(input.squeeze())
        output = output.view(-1, 3, self.num_points)
        return output


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['D']['use_bias']
        self.relu_slope = config['model']['D']['relu_slope']
        self.dropout = config['model']['D']['dropout']

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
                      bias=self.use_bias),
        )

        self.model = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(256, 1, bias=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.max(dim=2)[0]
        logit = self.model(x)
        return logit


class Comparator(nn.Module):
    def __init__(self, pointnet):
        super().__init__()
        self.feat = pointnet.model.feat
        self.model = nn.Sequential(pointnet.model.fc1,
                            pointnet.model.bn1,
                            nn.ReLU(inplace=True),
                            pointnet.model.fc2,
                            pointnet.model.dropout,
                            pointnet.model.bn2,
                            nn.ReLU(inplace=True),
                            pointnet.model.fc3)

    def forward(self, input):
        x, _, _ = self.feat(input)
        x = self.model(x)
        return x


class Encoder(nn.Module):
    def __init__(self, pointnet):
        super().__init__()
        self.feat = pointnet.model.feat

    def forward(self, input):
        x, trans, trans_feat = self.feat(input)
        return x, trans, trans_feat


class AAE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # MODELS
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        
        pointnet = PointNet.load_from_checkpoint(join(config['ckpts_dir'], config['pointnet_ckpt']))
        pointnet.freeze()

        self.encoder = Encoder(pointnet)
        self.comparator = Comparator(pointnet)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # LOSSES
        if self.config['reconstruction_loss'].lower() == 'chamfer':
            from losses.champfer_loss import ChamferLoss
            self.reconstruction_loss = ChamferLoss()
        elif config['reconstruction_loss'].lower() == 'earth_mover':
            from losses.earth_mover_distance import EMD
            self.reconstruction_loss = EMD()
        else:
            raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                             f'`earth_mover`, got: {config["reconstruction_loss"]}')
        self.comparator_loss = nn.MSELoss(reduction='mean')

        # FAKE TENSOR FOR DISCRIMINATOR TRAINING
        # if torch.cuda.is_available():
        #     self.noise = torch.cuda.FloatTensor(config['batch_size'], config['z_size'])
        # else:
        #     self.noise = torch.FloatTensor(config['batch_size'], config['z_size'])

        self.save_hyperparameters(config)

    def _models_sanity_check(self, optimizer_idx):
        assert (not self.comparator.training and not self.encoder.training), "Both comparator and encoder need to be in evaluation mode"
        assert (self.generator.training and self.discriminator.training), "Both generator and discriminator need to be in training mode"

        # Comparator must be freezed
        for param in self.comparator.parameters():
            assert not param.requires_grad, "The comparator model must be frozen"

        # Encoder must be freezed
        for param in self.encoder.parameters():
            assert not param.requires_grad, "The encoder model must be frozen"

        # Train generator
        if optimizer_idx == 0:
            # Generator requires gradients
            for param in self.generator.parameters():
                assert param.requires_grad, "The generator model requires gradients, do not freeze it"

            # Discriminator does not require gradients
            for param in self.discriminator.parameters():
                assert not param.requires_grad, "The discriminator model does not require gradients while training the generator"
        
        # Train discriminator
        elif optimizer_idx == 1:
            # Discriminator requires gradients
            for param in self.discriminator.parameters():
                assert param.requires_grad, "The discriminator model requires gradients, do not freeze it"
            # Generator does not require gradients
            for param in self.generator.parameters():
                assert not param.requires_grad, "The generator model does not require gradients while training the discriminator"

    def forward(self, z):
        return self.generator(z)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, 1).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + (fake_samples - alpha * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones_like(d_interpolates).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def configure_optimizers(self):
        EGC_optim = getattr(torch.optim, self.config['optimizer']['EGC']['type'])
        EGC_optim = EGC_optim(chain(self.encoder.parameters(), self.comparator.parameters(), self.generator.parameters()),
                        **self.config['optimizer']['EGC']['hyperparams'])

        D_optim = getattr(torch.optim, self.config['optimizer']['D']['type'])
        D_optim = D_optim(self.discriminator.parameters(),
                      **self.config['optimizer']['D']['hyperparams'])
        
        return (
            {'optimizer': EGC_optim, 'frequency': self.config['optimizer']['EGC']['frequency']},
            {'optimizer': D_optim, 'frequency': self.config['optimizer']['D']['frequency']}
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.encoder.eval()
        self.comparator.eval()
        self._models_sanity_check(optimizer_idx)
        X, _ = batch

        # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
        if X.size(-1) == 3:
            X.transpose_(X.dim() - 2, X.dim() - 1)

        codes, _, _ = self.encoder(X)
        assert not codes.requires_grad

        # train generator
        if optimizer_idx == 0:
            X_rec = self(codes)
            assert X_rec.requires_grad

            # points space loss
            pts_loss = torch.mean(
                self.config['reconstruction_coef'] *
                self.reconstruction_loss(X.permute(0, 2, 1) + 0.5,
                                    X_rec.permute(0, 2, 1) + 0.5))

            # adversarial loss
            synth_logit = self.discriminator(X_rec)
            assert synth_logit.requires_grad
            adv_loss = -torch.mean(synth_logit)

            # features space loss
            pre_logits_rec = self.comparator(X_rec)
            assert pre_logits_rec.requires_grad
            pre_logits_X = self.comparator(X)
            assert not pre_logits_X.requires_grad
            feat_loss = self.comparator_loss(pre_logits_rec, pre_logits_X)

            # generator loss
            eg_loss = pts_loss + adv_loss + feat_loss

            metrics = {'eg_loss': eg_loss, 'adv_loss': adv_loss, 'pts_loss': pts_loss, 'feat_loss': feat_loss}
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return eg_loss

        # train discriminator
        elif optimizer_idx == 1:
            X_fake = self(codes)
            assert not X_fake.requires_grad
            synth_logit = self.discriminator(X_fake)
            assert synth_logit.requires_grad
            real_logit = self.discriminator(X)
            assert real_logit.requires_grad

            gradient_penalty = self.compute_gradient_penalty(X.data, X_fake.data)
            d_loss = torch.mean(synth_logit) - torch.mean(real_logit) + self.config['gp_lambda'] * gradient_penalty

            metrics = {'d_loss': d_loss}
            self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return d_loss
