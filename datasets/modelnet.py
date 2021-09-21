import os
from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class ModelNet40(Dataset):

    all_classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle',
               'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door',
               'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
               'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person',
               'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa',
               'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
               'wardrobe', 'xbox']

    number_to_category = {i: c for i, c in enumerate(all_classes)}
    category_to_number = {c: i for i, c in enumerate(all_classes)}

    def __init__(self, root_dir='./modelnet40', classes=[],
                 transform=[], split='train', valid_percent=10, percent_supervised=0.0):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split (string): `train` or `test`
            valid_percent (int): Percent of train (from the end) to use as valid set.
        """
        self.root_dir = root_dir
        self.raw_dir = f'{self.root_dir}/raw'
        self.transform = transform
        self.split = split.lower()
        self.valid_percent = valid_percent
        self.percent_supervised = percent_supervised

        if self.split in ('train', 'valid'):
            self.files_list = os.path.join(self.raw_dir, 'train_files.txt')
        elif self.split == 'test':
            self.files_list = os.path.join(self.raw_dir, 'test_files.txt')
        else:
            raise ValueError('Incorrect split')

        self.processed_dir = f'{self.root_dir}/processed'
        if os.path.isfile(os.path.join(self.processed_dir, f"{split}.pth")):
            self.data, self.labels = self._load_dataset(split)
        else:
            data, labels = self._read_files()

            if classes:
                if classes[0] in ModelNet40.all_classes:
                    classes = np.asarray([ModelNet40.category_to_number[c] for c in classes])
                filter = [label in classes for label in labels]
                data = data[filter]
                labels = labels[filter]
            else:
                classes = np.arange(len(ModelNet40.all_classes))

            if self.split in ('train', 'valid'):
                new_data, new_labels = [], []
                if self.percent_supervised > 0.0:
                    data_sup, labels_sub = [], []
                for c in classes:
                    pc_in_class = sum(labels.flatten() == c)

                    if self.split == 'train':
                        portion = slice(0, int(pc_in_class * (1 - (self.valid_percent / 100))))
                    else:
                        portion = slice(int(pc_in_class * (1 - (self.valid_percent / 100))), pc_in_class)

                    new_data.append(data[labels.flatten() == c][portion])
                    new_labels.append(labels[labels.flatten() == c][portion])

                    if self.percent_supervised > 0.0:
                        n_max = int(self.percent_supervised * (portion.stop - 1))
                        data_sup.append(data[labels.flatten() == c][:n_max])
                        labels_sub.append(labels[labels.flatten() == c][:n_max])
                data = np.vstack(new_data)
                labels = np.vstack(new_labels)
                if self.percent_supervised > 0.0:
                    self.data_sup = np.vstack(data_sup)
                    self.labels_sup = np.vstack(labels_sub)
            self.data = data
            self.labels = labels

            if not os.path.exists(self.processed_dir):
                os.makedirs(self.processed_dir)
            torch.save({"data": self.data,
                        "labels": self.labels}, os.path.join(self.processed_dir, f"{split}.pth"))

    def _load_dataset(self, split):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataset = torch.load(os.path.join(self.processed_dir, f"{split}.pth"), map_location=device)
        data = dataset["data"]
        labels = dataset["labels"]
        return data, labels

    def _read_files(self) -> pd.DataFrame:
        with open(self.files_list) as f:
            files = [os.path.join(self.raw_dir, line.rstrip().rsplit('/', 1)[1]) for line in f]

        data, labels = [], []
        for file in tqdm(files, total=len(files), desc="Reading files "):
            with h5py.File(file) as f:
                data.extend(f['data'][:])
                labels.extend(f['label'][:])

        return np.asarray(data), np.asarray(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_set = self.data[idx]
        cls = self.labels[idx]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
        point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls
