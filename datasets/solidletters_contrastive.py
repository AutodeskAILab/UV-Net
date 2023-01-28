import dgl
import pathlib
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from datasets.base import BaseDataset
from datasets.solidletters import _get_filenames, _char_to_label


class SolidLettersContrastive(BaseDataset):
    def __init__(
            self,
            root_dir,
            split="train",
            center_and_scale=True,
            random_rotate=False,
            shape_type="upper",
            prob_full_graph=0.10,
    ):
        assert shape_type in ("upper", "lower", "both")
        self.split = split
        self.random_rotate = random_rotate
        self.prob_full_graph = prob_full_graph
        assert split in ("train", "val", "test")
        path = pathlib.Path(root_dir)

        self.random_rotate = random_rotate

        if split in ("train", "val"):
            file_paths = _get_filenames(path, filelist="train.txt")
            print(f"Found {len(file_paths)} bin files")
            # The first character of filename must be according to shape_type
            file_paths = [fn for fn in file_paths if shape_type in fn.stem]
            print(f"Left with {len(file_paths)} bin files after filtering by shape type:", shape_type)
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
            labels = [_char_to_label(fn.stem[0]) for fn in file_paths]
        self.labels = labels

        print(f"Loading {split} data...")
        self.load_graphs(file_paths, center_and_scale)

    def num_classes(self):
        return 0

    def _collate(self, batch):
        collated = {"graph": dgl.batch([x["graph"] for x in batch]),
                    "label": torch.stack([x["label"] for x in batch]),
                    "filename": [x["filename"] for x in batch]}
        if self.split in ("train", "val"):
            collated["graph2"] = dgl.batch([x["graph2"] for x in batch])
        else:
            collated["graph2"] = None
        return collated

    def apply_transformation(self, graph):
        all_transformations = ("sub_graph", "sub_graph_2hops", "drop_nodes", "drop_edges")
        transformation_type = random.choice(all_transformations)
        #
        # if transformation_type == "all_no_sub_graph":
        #     all_transformations = ["sub_graph_2hops", "drop_nodes", "drop_edges"]
        #     transformation_type = random.choice(all_transformations)
        #
        # if transformation_type == "all_no_sub_graph_2hops":
        #     all_transformations = ["sub_graph_2hops", "drop_nodes", "drop_edges"]
        #     transformation_type = random.choice(all_transformations)
        #
        # if transformation_type == "all_no_drop_nodes":
        #     all_transformations = ["sub_graph", "sub_graph_2hops", "drop_edges"]
        #     transformation_type = random.choice(all_transformations)
        # if transformation_type == "all_no_drop_edges":
        #     all_transformations = ["sub_graph", "sub_graph_2hops", "drop_nodes"]
        #     transformation_type = random.choice(all_transformations)
        # if transformation_type == "full_graph":
        #     return graph.clone()
        if transformation_type == "sub_graph":
            graph2 = self.get_subgraph(graph, num_nodes=1, hops=1, normalize=True)
            return graph2
        elif transformation_type == "sub_graph_2hops":
            graph2 = self.get_subgraph(graph, num_nodes=1, hops=2, normalize=True)
            return graph2
        # elif transformation_type == "sub_graph_non_normalize":
        #     graph2 = get_subgraph(graph, num_nodes=1, hops=1, normalize=False)
        #     return graph2
        # elif transformation_type == "sub_graph_2hops_non_normalize":
        #     graph2 = get_subgraph(graph, num_nodes=1, hops=2, normalize=False)
        #     return graph2
        # elif transformation_type == "rotation":
        #     graph2 = graph.clone()
        #     graph2.ndata["x"] = font_util.random_rotate_uvsolid(
        #         graph2.ndata["x"],
        #         axis=np.array([0, 0, 1], dtype=np.float32,),
        #         min_angle_radians=-np.pi / 4,
        #         max_angle_radians=+np.pi / 4,
        #     )
        #     return graph2
        # elif transformation_type == "remove_nodes":
        #     graph2 = graph.clone()
        #     remove_nodes_num = int(0.7 * graph2.number_of_nodes())
        #     remove_choice1 = np.random.choice(
        #         list(range(graph2.number_of_nodes())), remove_nodes_num
        #     ).astype(np.int64)
        #     graph2.remove_nodes(torch.tensor(remove_choice1))
        #     return graph2
        # elif transformation_type == "remove_edges":
        #     graph2 = graph.clone()
        #     remove_edges_num = int(0.7 * graph2.number_of_edges())
        #     remove_choice1 = np.random.choice(
        #         list(range(graph2.number_of_edges())), remove_edges_num
        #     ).astype(np.int64)
        #     graph2.remove_edges(torch.tensor(remove_choice1))
        #     return graph2
        # elif transformation_type == "translate":
        #     graph2 = graph.clone()
        #     graph2.ndata["x"] = font_util.translate_uvsolid(graph2.ndata["x"])
        #     return graph2
        elif transformation_type == "drop_nodes":
            graph2 = graph.clone()
            nodes_to_drop = []
            for i in range(graph2.number_of_nodes()):
                if random.random() <= 0.4:
                    nodes_to_drop.append(i)
            graph2.remove_nodes(nodes_to_drop)
            return graph2
        elif transformation_type == "drop_edges":
            graph2 = graph.clone()
            edges_to_drop = []
            for i in range(graph2.number_of_edges()):
                if random.random() <= 0.4:
                    edges_to_drop.append(i)
            graph2.remove_edges(edges_to_drop)
            return graph2

    def __getitem__(self, idx):
        graph_and_filename = self.data[idx]
        filename = graph_and_filename["filename"]
        graph = graph_and_filename["graph"]
        graph2 = self.apply_transformation(graph.clone())
        if self.split == "train" and random.uniform(0, 1) > self.prob_full_graph:
            graph = self.apply_transformation(graph.clone())
        else:
            graph = graph.clone()
        graph.ndata.pop("_ID", None)
        graph.edata.pop("_ID", None)
        graph2.ndata.pop("_ID", None)
        graph2.edata.pop("_ID", None)
        return {"graph": graph, "graph2": graph2, "label": torch.tensor([self.labels[idx]], dtype=torch.long),
                "filename": filename}

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0, drop_last=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=drop_last,
        )

    def get_subgraph(self, graph, num_nodes=1, hops=2, normalize=False):
        assert num_nodes >= 1
        assert hops >= 0
        # node_idx = random.sample(graph.nodes().cpu().numpy().tolist(), 1)
        subgraph_nodes = [random.sample(graph.nodes().cpu().numpy().tolist(), 1)[0]]
        for _ in range(hops):
            neighbors = []
            for idx in subgraph_nodes:
                # out_edges returns a tuple of src and dst nodes
                neighbors.extend(graph.out_edges(idx)[1].cpu().numpy().tolist())
            for ngh in neighbors:
                subgraph_nodes.append(ngh)
        subgraph_nodes = list(set(subgraph_nodes))
        subgraph = graph.subgraph(subgraph_nodes)
        # subgraph.copy_from_parent()
        if normalize:
            subgraph = self.center_and_scale_graph(subgraph)
        return subgraph
