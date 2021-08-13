import numpy as np
from datasets.base import BaseDataset
import pathlib
import torch
import json
from sklearn.model_selection import train_test_split


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

        # Locate the labels directory.  In s1.0.0 this would be  self.path / "breps"
        # but in s2.0.0 this is self.path / "breps/seg"
        self.seg_path = self.path / "breps/seg"
        if not self.seg_path.exists():
            self.seg_path = self.path / "breps"

        assert split in ("train", "val", "test")

        with open(str(path.joinpath("train_test.json")), "r") as read_file:
            filelist = json.load(read_file)

        # NOTE: Using a held out validation set may be better.
        # But it's not easy to perform stratified sampling on some rare classes
        # which only show up on a few solids.
        if split in ("train", "val"):
            full_train_filelist = filelist["train"]
            train_filesplit, val_filesplit = train_test_split(
                full_train_filelist, test_size=0.2, random_state=42
            )
            if split == "train":
                split_filelist = train_filesplit
            else:
                split_filelist = val_filesplit
        else:
            split_filelist = filelist["test"]

        self.random_rotate = random_rotate

        # Call base class method to load all graphs
        print(f"Loading {split} data...")
        all_files = [path.joinpath("graph").joinpath(fn + ".bin") for fn in split_filelist]
        self.load_graphs(all_files, center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def load_one_graph(self, file_path):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        # Additionally load the label and store it as node data
        label = np.loadtxt(
            self.seg_path.joinpath(file_path.stem + ".seg"), dtype=np.int, ndmin=1
        )
        if sample["graph"].number_of_nodes() != label.shape[0]:
            return None
        sample["graph"].ndata["y"] = torch.tensor(label).long()
        return sample
