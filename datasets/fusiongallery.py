import numpy as np
import pathlib
from torch.utils.data import Dataset, DataLoader
import dgl
import torch
from dgl.data.utils import load_graphs
import json
from datasets import util
from tqdm import tqdm


class FusionGalleryDataset(Dataset):
    @staticmethod
    def num_classes():
        return 8

    def __init__(
        self,
        root_dir,
        split="train",
        center_and_scale=True,
    ):
        """
        Load the Fusion Gallery dataset from:
        Joseph G. Lambourne, Karl D. D. Willis, Pradeep Kumar Jayaraman, Aditya Sanghi,
        Peter Meltzer, Hooman Shayani. "BRepNet: A topological message passing system
        for solid models," CVPR 2021.

        :param root_dir: Root path to the dataset
        :param split: string Whether train, val or test set
        """
        path = pathlib.Path(root_dir)
        assert split in ("train", "val", "test")

        with open(str(path.joinpath("train_test.json")), "r") as read_file:
            filelist = json.load(read_file)

        # NOTE: Using a held out out validation set may be better.
        # But it's not easy to perform stratified sampling on some rare classes
        # which only show up on a few solids.
        if split in ("train", "val"):
            split_filelist = filelist["train"]
        else:
            split_filelist = filelist["test"]

        self.center_and_scale = center_and_scale

        all_files = []
        # Load graphs and store their filenames for loading labels next
        for fn in split_filelist:
            all_files.append(path.joinpath("graph").joinpath(fn + ".bin"))

        # Load labels from the json files in the subfolder
        self.files = []
        self.graphs = []
        print(f"Loading {split} data...")
        for fn in tqdm(all_files):
            if not fn.exists():
                continue
            graph = load_graphs(str(fn))[0][0]
            label = np.loadtxt(
                path.joinpath("breps").joinpath(fn.stem + ".seg"), dtype=np.int, ndmin=1
            )
            if label.size != graph.number_of_nodes():
                # Skip files where the number of faces and labels don't match
                # print(
                #     f"WARN: number of faces  and labels do not match in {fn.stem}: {label.size} vs. {graph.number_of_nodes()}"
                # )
                continue
            self.files.append(fn)
            graph.ndata["y"] = torch.tensor(label).long()
            self.graphs.append(graph)

        if self.center_and_scale:
            for i in range(len(self.graphs)):
                self.graphs[i].ndata["x"] = util.center_and_scale_uvsolid(
                    self.graphs[i].ndata["x"]
                )

    def __len__(self):
        return len(self.graphs)

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
