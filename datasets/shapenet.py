import urllib
import shutil
import os
from tqdm import tqdm
from os.path import exists, join
from zipfile import ZipFile

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.plyfile import load_ply

class ShapeNetDataset(Dataset):
    synth_id_to_category = {
        '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
        '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
        '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
        '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
        '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
        '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
        '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
        '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
        '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
        '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
        '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
        '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
        '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
        '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
        '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
        '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
        '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
        '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
        '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
    }

    category_to_synth_id = {v: k for k, v in synth_id_to_category.items()}
    synth_id_to_number = {k: i for i, k in enumerate(synth_id_to_category.keys())}
    number_to_synth_id = {num: synth for synth, num in synth_id_to_number.items()}

    def __init__(self, root_dir, classes=[],
                 transform=None, split='train'):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        self.processed_dir = join(self.root_dir, 'processed')
        if os.path.isfile(os.path.join(self.processed_dir, f"{split}.pth")):
            self.data, self.labels = self._load_dataset(split)
        else:
            pc_df = self._get_names()
            if classes:
                if classes[0] not in ShapeNetDataset.synth_id_to_category.keys():
                    classes = [ShapeNetDataset.category_to_synth_id[c] for c in classes]
                pc_df = pc_df[pc_df.category.isin(classes)].reset_index(drop=True)
            else:
                classes = ShapeNetDataset.synth_id_to_category.keys()

            if self.split == 'train':
                pc_names = pd.concat([pc_df[pc_df['category'] == c][:int(0.85*len(pc_df[pc_df['category'] == c]))].reset_index(drop=True) for c in classes])
            elif self.split == 'valid':
                pc_names = pd.concat([pc_df[pc_df['category'] == c][int(0.85*len(pc_df[pc_df['category'] == c])):int(0.9*len(pc_df[pc_df['category'] == c]))].reset_index(drop=True) for c in classes])
            elif self.split == 'test':
                pc_names = pd.concat([pc_df[pc_df['category'] == c][int(0.9*len(pc_df[pc_df['category'] == c])):].reset_index(drop=True) for c in classes])
            else:
                raise ValueError('Invalid split. Should be train, valid or test.')

            self.data = []
            self.labels = []
            for index, row in tqdm(pc_names.iterrows(), total=pc_names.shape[0], desc="Loading shapenet"):
                pc_category, pc_filename = row.values
                pc_filepath = join(self.root_dir, 'raw', pc_category, pc_filename)
                sample = load_ply(pc_filepath)
                if self.transform:
                    sample = self.transform(sample)
                
                self.data.append(sample)
                self.labels.append(ShapeNetDataset.synth_id_to_number[pc_category])

            if not exists(self.processed_dir):
                os.makedirs(self.processed_dir)
            torch.save({"data": self.data,
                        "labels": self.labels}, join(self.processed_dir, f"{split}.pth"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_set = self.data[idx]
        cls = self.labels[idx]
        return point_set, cls

    def _get_names(self) -> pd.DataFrame:
        filenames = []
        for category_id in ShapeNetDataset.synth_id_to_category.keys():
            for f in os.listdir(join(self.root_dir, "raw", category_id)):
                if f not in ['.DS_Store']:
                    filenames.append((category_id, f))
        return pd.DataFrame(filenames, columns=['category', 'filename'])

    def _load_dataset(self, split):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = torch.load(join(self.processed_dir, f"{split}.pth"), map_location=device)
        data = dataset["data"]
        labels = dataset["labels"]
        return data, labels
