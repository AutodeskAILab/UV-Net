from datasets.base import BaseDataset
import pathlib
import torch
import json


class MFCADDataset(BaseDataset):
    @staticmethod
    def num_classes():
        return 16

    def __init__(
        self, root_dir, split="train", center_and_scale=True, random_rotate=False,
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

        Args:
            root_dir (str): Root path of dataset
            split (str, optional): Data split to load. Defaults to "train".
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        path = pathlib.Path(root_dir)
        self.path = path
        assert split in ("train", "val", "test")

        with open(str(str(path.joinpath("split.json"))), "r") as read_file:
            filelist = json.load(read_file)

        if split == "train":
            split_filelist = filelist["train"]
        elif split == "val":
            split_filelist = filelist["validation"]
        else:
            split_filelist = filelist["test"]

        self.random_rotate = random_rotate

        all_files = []
        for fn in split_filelist:
            all_files.append(path.joinpath("graph").joinpath(fn + ".bin"))

        # Load graphs
        print(f"Loading {split} data...")
        self.load_graphs(all_files, center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def load_one_graph(self, file_path):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        # Additionally load the label and store it as node data
        label_file = self.path.joinpath("labels").joinpath(file_path.stem + "_ids.json")
        with open(str(label_file), "r") as read_file:
            labels_data = json.load(read_file)
        label = []
        for face in labels_data["body"]["faces"]:
            index = face["segment"]["index"]
            label.append(index)
        sample["graph"].ndata["y"] = torch.tensor(label).long()
        return sample
