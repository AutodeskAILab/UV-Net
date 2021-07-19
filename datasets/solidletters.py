import pathlib
import string

import torch
from sklearn.model_selection import train_test_split

from datasets.base import BaseDataset


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


class SolidLetters(BaseDataset):
    @staticmethod
    def num_classes():
        return 26

    def __init__(
        self,
        root_dir,
        split="train",
        center_and_scale=True,
        random_rotate=False,
    ):
        """
        Load the SolidLetters dataset

        Args:
            root_dir (str): Root path to the dataset
            split (str, optional): Split (train, val, or test) to load. Defaults to "train".
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        assert split in ("train", "val", "test")
        path = pathlib.Path(root_dir)

        self.random_rotate = random_rotate

        if split in ("train", "val"):
            file_paths = _get_filenames(path, filelist="train.txt")
            # The first character of filename must be the alphabet
            labels = [_char_to_label(fn.stem[0]) for fn in file_paths]
            train_files, val_files = train_test_split(
                file_paths, test_size=0.2, random_state=42, stratify=labels,
            )
            if split == "train":
                file_paths = train_files
            elif split == "val":
                file_paths = val_files
        elif split == "test":
            file_paths = _get_filenames(path, filelist="test.txt")

        print(f"Loading {split} data...")
        self.load_graphs(file_paths, center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def load_one_graph(self, file_path):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        # Additionally get the label from the filename and store it in the sample dict
        sample["label"] = torch.tensor([_char_to_label(file_path.stem[0])]).long()
        return sample

    def _collate(self, batch):
        collated = super()._collate(batch)
        collated["label"] =  torch.cat([x["label"] for x in batch], dim=0)
        return collated
