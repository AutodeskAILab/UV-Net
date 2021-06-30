import pathlib
import random
import string

import dgl
import numpy as np
import torch
from dgl.data.utils import load_graphs, save_graphs
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets import util


def _get_filenames(root_dir, filelist):
    with open(str(root_dir / f"{filelist}"), "r") as f:
        file_list = [x.strip() for x in f.readlines()]

    files = list(
        x
        for x in root_dir.rglob(f"*.bin")
        if x.stem in file_list
        #if util.valid_font(x) and x.stem in file_list
    )
    return files


CHAR2LABEL = {char: i for (i, char) in enumerate(string.ascii_lowercase)}


def _char_to_label(char):
    return CHAR2LABEL[char.lower()]


class SolidLetters(Dataset):
    @classmethod
    def num_classes(cls):
        return 26

    def __init__(
        self,
        root_dir,
        split="train",
        center_and_scale=True,
    ):
        """
        Load and create the SolidMNIST dataset
        :param root_dir: Root path to the dataset (UV bin files are expected to be in a 'bin' subfolder)
        :param split: string Whether train, val or test set
        :param center_and_scale: Center and scale the UV grids
        """
        assert split in ("train", "val", "test")
        path = pathlib.Path(root_dir)

        self.center_and_scale = center_and_scale

        if split in ("train", "val"):
            bin_files = _get_filenames(path, filelist="train.txt")
            # The first character of filename must be the alphabet
            labels = [_char_to_label(fn.stem[0]) for fn in bin_files]
            train_files, val_files, train_labels, val_labels = train_test_split(
                bin_files, labels, test_size=0.2, random_state=42, stratify=labels,
            )
            if split == "train":
                self.graph_files = train_files
                self.labels = train_labels
            elif split == "val":
                self.graph_files = val_files
                self.labels = val_labels
        elif split == "test":
            self.graph_files = _get_filenames(path, filelist="test.txt")
            self.labels = [_char_to_label(fn.stem[0]) for fn in self.graph_files]

        self.graphs = []
        print(f"Loading {split} data...")
        for fn in tqdm(self.graph_files):
            self.graphs.append(load_graphs(str(fn))[0][0])
        if self.center_and_scale:
            for i in range(len(self.graphs)):
                self.graphs[i].ndata["x"] = util.center_and_scale_uvgrid(
                    self.graphs[i].ndata["x"]
                )

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph, torch.tensor([self.labels[idx]]).long()

    def _collate(self, batch):
        graphs, labels = map(list, zip(*batch))
        labels = torch.cat(labels, dim=0)
        bg = dgl.batch(graphs)
        return bg, labels

    def get_dataloader(self, batch_size=64, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=0,  # Can be set to non-zero on Linux
            drop_last=True,
        )
