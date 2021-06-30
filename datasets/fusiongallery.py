import numpy as np
from datasets.base import BaseDataset
import pathlib
from torch.utils.data import Dataset, DataLoader
import torch
import json
from datasets import util
from tqdm import tqdm


class FusionGalleryDataset(BaseDataset):
    @staticmethod
    def num_classes():
        return 8

    def __init__(
        self, root_dir, split="train", center_and_scale=True, random_rotate=False,
    ):
        """
        Load the Fusion Gallery dataset from:
        Joseph G. Lambourne, Karl D. D. Willis, Pradeep Kumar Jayaraman, Aditya Sanghi,
        Peter Meltzer, Hooman Shayani. "BRepNet: A topological message passing system
        for solid models," CVPR 2021.

        Args:
            root_dir (str): Root path of dataset
            split (str, optional): Data split to load. Defaults to "train".
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        path = pathlib.Path(root_dir)
        self.path = path
        assert split in ("train", "val", "test")

        with open(str(path.joinpath("train_test.json")), "r") as read_file:
            filelist = json.load(read_file)

        # NOTE: Using a held out validation set may be better.
        # But it's not easy to perform stratified sampling on some rare classes
        # which only show up on a few solids.
        if split in ("train", "val"):
            split_filelist = filelist["train"]
        else:
            split_filelist = filelist["test"]

        self.random_rotate = random_rotate

        # Call base class method to load all graphs
        print(f"Loading {split} graphs...")
        all_files = [path.joinpath("graph").joinpath(fn + ".bin") for fn in split_filelist]
        self.load_graphs(all_files, center_and_scale)
        print("Done loading {} files".format(len(self.files)))

    def load_one_graph(self, file_path):
        # Load the graph use base class method
        graph = super().load_one_graph(file_path)
        # Additionally load the label and store it as node data
        label = np.loadtxt(
            self.path.joinpath("breps").joinpath(file_path.stem + ".seg"), dtype=np.int, ndmin=1
        )
        graph.ndata["y"] = torch.tensor(label).long()
        return graph
