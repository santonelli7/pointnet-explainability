import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

import torchmetrics
import pytorch_lightning as pl

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.relu_(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu_(self.bn4(self.fc1(x)))
        x = F.relu_(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.relu_(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu_(self.bn4(self.fc1(x)))
        x = F.relu_(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu_(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu_(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu_(self.bn1(self.fc1(x)))
        x = F.relu_(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = PointNetCls(k=config['num_classes'], feature_transform=config['feature_transform'])
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.save_hyperparameters(config)

    def forward(self, x):
        softmax_logits, trans, trans_feat = self.model(x)
        return softmax_logits, trans, trans_feat

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.config['optimizer']['type'])
        optim = optim(self.model.parameters(),
                        **self.config['optimizer']['hyperparams'])

        scheduler = getattr(torch.optim.lr_scheduler, self.config['scheduler']['type'])
        scheduler = scheduler(optim,
                        **self.config['scheduler']['hyperparams'])
        
        return {'optimizer': optim, 
                'lr_scheduler': {
                    'scheduler': scheduler},
                    'monitor': 'metric_to_track',
                }

    def training_step(self, batch, batch_idx):
        points, target = self._dataprocess(batch)
        loss, pred = self._shared_step(points, target)
        acc = self.train_accuracy(pred, target)
        metrics = {'acc': acc, 'loss': loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def validation_step(self, batch, batch_idx):
        points, target = self._dataprocess(batch)
        loss, pred = self._shared_step(points, target)
        acc = self.val_accuracy(pred, target)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def test_step(self, batch, batch_idx):
        points, target = self._dataprocess(batch)
        loss, pred = self._shared_step(points, target)
        acc = self.test_accuracy(pred, target)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'points': points, 'target': target, 'pred': pred.argmax(dim=-1)}

    def _dataprocess(self, batch):
        points, target = batch
        target = target.squeeze()
        points = points.transpose(2, 1)
        return points, target

    def _shared_step(self, points, target):
        pred, _, trans_feat = self(points)
        loss = F.nll_loss(pred, target)
        if self.config['feature_transform']:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        return loss, pred
