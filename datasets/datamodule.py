import urllib
import shutil
from os import listdir, remove, makedirs
from os.path import exists, join
from zipfile import ZipFile

import torch
import pytorch_lightning as pl

class LitDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, config):
        super().__init__()
        self.data_dir = data_dir
        self.config = config

    def prepare_data(self):
        # download
        raw_dir = join(self.data_dir, 'raw')
        if not exists(raw_dir):
            makedirs(raw_dir)
        
        if len(listdir(raw_dir)) != 0:
            return

        print(f'Downloading...')
        dataset_name = self.config['dataset'].lower()
        if dataset_name == 'shapenet':
            url = 'https://www.dropbox.com/s/vmsdrae6x5xws1v/shape_net_core_uniform_samples_2048.zip?dl=1'
            dir_name = 'shape_net_core_uniform_samples_2048'
        elif dataset_name == 'modelnet':
            url = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
            dir_name = 'modelnet40_ply_hdf5_2048'
        else:
            raise ValueError(f'Invalid dataset name. Expected `shapenet` or 'f'`modelnet`. Got: `{dataset_name}`')

        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2][:-5]
        file_path = join(self.data_dir, filename)
        with open(file_path, mode='wb') as f:
            d = data.read()
            f.write(d)

        print('Extracting...')
        with ZipFile(file_path, mode='r') as zip_f:
            zip_f.extractall(self.data_dir)

        remove(file_path)

        extracted_dir = join(self.data_dir, dir_name)
        for d in listdir(extracted_dir):
            shutil.move(src=join(extracted_dir, d),
                        dst=raw_dir)

        shutil.rmtree(extracted_dir)

    def setup(self, stage = None):

        dataset_name = self.config['dataset'].lower()
        if dataset_name == 'shapenet':
            from datasets.shapenet import ShapeNetDataset as ShapeNet
            dataset = ShapeNet
        elif dataset_name == 'modelnet':
            from datasets.modelnet import ModelNet40 as ModelNet
            dataset = ModelNet
        else:
            raise ValueError(f'Invalid dataset name. Expected `shapenet` or 'f'`modelnet`. Got: `{dataset_name}`')

        if stage == "fit" or stage is None:
            self.train = dataset(root_dir=self.data_dir,
                                    classes=self.config['classes'])

            self.val = dataset(root_dir=self.data_dir, split='test')
            points_test = []
            cls_test = []
            for gt_cls in range(self.config['num_classes']):
                for points, cls in self.val:
                    if cls == gt_cls:
                        points_test.append(points.numpy())
                        cls_test.append(cls.numpy())
                        break

            dataset = torch.utils.data.TensorDataset(torch.FloatTensor(points_test), torch.LongTensor(cls_test))
            self.test_samples = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset))))

        if stage == "test" or stage is None:
            self.test = dataset(root_dir=self.data_dir, split="test")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            # num_workers=self.config['num_workers'],
            drop_last=True, 
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.config['batch_size']*2)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.config['batch_size']*2)
