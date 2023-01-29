import dgl
import pathlib
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from datasets.base import BaseDataset, BaseContrastiveDataset
from datasets.solidletters import _get_filenames, _char_to_label


class SolidLettersContrastive(BaseContrastiveDataset):
    def __init__(
            self,
            root_dir,
            split="train",
            center_and_scale=True,
            shape_type="upper",
            prob_full_graph=0.10,
            size_percentage=1.0,
    ):
        assert shape_type in ("upper", "lower", "both")
        super().__init__(split, prob_full_graph)
        path = pathlib.Path(root_dir)

        if split in ("train", "val"):
            file_paths = _get_filenames(path, filelist="train.txt")
            print(f"Found {len(file_paths)} bin files")
            # The first character of filename must be according to shape_type
            if shape_type != "both":
                file_paths = [fn for fn in file_paths if shape_type in fn.stem]
            print(f"Left with {len(file_paths)} bin files after filtering by shape type:", shape_type)
            labels_to_stratify = [_char_to_label(fn.stem[0]) for fn in file_paths]
            train_files, val_files = train_test_split(
                file_paths, test_size=0.2, random_state=42, stratify=labels_to_stratify,
            )
            if split == "train":
                file_paths = train_files
            elif split == "val":
                file_paths = val_files
            labels = [torch.tensor([_char_to_label(fn.stem[0])]).long() for fn in file_paths]
        elif split == "test":
            file_paths = _get_filenames(path, filelist="test.txt")
            labels = [torch.tensor([_char_to_label(fn.stem[0])]).long() for fn in file_paths]
        self.labels = labels

        if size_percentage < 1.0:
            k = int(size_percentage * len(file_paths))
            index_list = set(random.sample(list(range(len(file_paths))), k))
            file_paths = [x for i, x in enumerate(file_paths) if i in index_list]
            self.labels = [x for i, x in enumerate(self.labels) if i in index_list]

        print(f"Loading {split} data...")
        self.load_graphs(file_paths, center_and_scale)

    @staticmethod
    def num_classes():
        # Only used during evaluation
        return 26
