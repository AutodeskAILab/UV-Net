import numpy as np
import pathlib
from torch.utils.data import Dataset, DataLoader
import dgl
import torch
from dgl.data.utils import load_graphs
import json
from datasets import util
from tqdm import tqdm


class MFCADDataset(Dataset):
    @staticmethod
    def num_classes():
        return 16

    def __init__(
        self,
        root_dir,
        split="train",
        center_and_scale=True,
    ):
        """
        Load the MFCAD dataset from:
        Weijuan Cao, Trevor Robinson, Yang Hua, Flavien Boussuge,
        Andrew R. Colligan, and Wanbin Pan. "Graph representation
        of 3d cad models for machining feature recognition with deep
        learning." In Proceedings of the ASME 2020 International
        Design Engineering Technical Conferences and Computers
        and Information in Engineering Conference, IDETC-CIE.
        ASME, 2020.

        :param root_dir: Root path to the dataset
        :param split: string Whether train, val or test set
        """
        path = pathlib.Path(root_dir)
        assert split in ("train", "val", "test")

        with open(str(str(path.joinpath("split.json"))), "r") as read_file:
            filelist = json.load(read_file)

        if split == "train":
            split_filelist = filelist["train"]
        elif split == "val":
            split_filelist = filelist["validation"]
        else:
            split_filelist = filelist["test"]

        self.center_and_scale = center_and_scale

        self.files = []
        for fn in split_filelist:
            self.files.append(path.joinpath("graph").joinpath(fn + ".bin"))

        # Load labels from the json files in the subfolder
        print(f"Loading {split} data...")
        self.graphs = []
        for fn in tqdm(self.files):
            graph = load_graphs(str(fn))[0][0]
            label_file = path.joinpath("labels").joinpath(fn.stem + "_ids.json")
            with open(str(label_file), "r") as read_file:
                labels_data = json.load(read_file)
            label = []
            for face in labels_data["body"]["faces"]:
                index = face["segment"]["index"]
                label.append(index)
            graph.ndata["y"] = torch.tensor(label).long()
            self.graphs.append(graph)

        self.num_face_channels = self.graphs[0].ndata["x"].size(3)
        self.num_edge_channels = self.graphs[0].edata["x"].size(2)

        if self.center_and_scale:
            for i in range(len(self.graphs)):
                self.graphs[i].ndata["x"] = util.center_and_scale_uvsolid(
                    self.graphs[i].ndata["x"]
                )

        print("Done loading {} files".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph

    def _collate(self, batch):
        bg = dgl.batch(batch)
        return bg

    def get_dataloader(self, batch_size=128, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=0,  # Can be set to non-zero on Linux
            drop_last=True,
        )
