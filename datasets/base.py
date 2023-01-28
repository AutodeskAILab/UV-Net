from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor
import dgl
from dgl.data.utils import load_graphs
from datasets import util
from tqdm import tqdm
from abc import abstractmethod


class BaseDataset(Dataset):
    @staticmethod
    @abstractmethod
    def num_classes():
        pass

    def load_graphs(self, file_paths, center_and_scale=True):
        self.data = []
        for fn in tqdm(file_paths, desc="Loading graphs"):
            if not fn.exists():
                continue
            sample = self.load_one_graph(fn)
            if sample is None:
                continue
            if sample["graph"].edata["x"].size(0) == 0:
                # Catch the case of graphs with no edges
                continue
            self.data.append(sample)
        if center_and_scale:
            self.center_and_scale()
        self.convert_to_float32()

    def load_one_graph(self, file_path):
        graph = load_graphs(str(file_path))[0][0]
        sample = {"graph": graph, "filename": file_path.stem}
        return sample

    def center_and_scale_graph(self, graph):
        graph.ndata["x"], center, scale = util.center_and_scale_uvgrid(
            graph.ndata["x"], return_center_scale=True
        )
        graph.edata["x"][..., :3] -= center
        graph.edata["x"][..., :3] *= scale
        return graph

    def center_and_scale(self):
        for i in range(len(self.data)):
            self.data[i]["graph"] = self.center_and_scale_graph(self.data[i]["graph"])

    def convert_to_float32(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"] = self.data[i]["graph"].ndata["x"].type(FloatTensor)
            self.data[i]["graph"].edata["x"] = self.data[i]["graph"].edata["x"].type(FloatTensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.random_rotate:
            rotation = util.get_random_rotation()
            sample["graph"].ndata["x"] = util.rotate_uvgrid(sample["graph"].ndata["x"], rotation)
            sample["graph"].edata["x"] = util.rotate_uvgrid(sample["graph"].edata["x"], rotation)
        return sample

    def _collate(self, batch):
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        return {"graph": batched_graph, "filename": batched_filenames}

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,  # Can be set to non-zero on Linux
            drop_last=True,
        )



def get_random_subgraph(graph, num_nodes=1, include_neighbors=True, normalize=True):
    """
    Gets a random subgraph from the given graph
    :param graph: DGL graph
    :param num_nodes: Number of nodes to randomly extract in the subgraph
    :param include_neighbors: Whether to include one-ring neighbors of the extracted nodes so that we get connected patches
    :param normalize: Whether to normalize the graph into a cube of size 2
    """
    node_idx = random.sample(graph.nodes().cpu().numpy().tolist(), num_nodes)
    subgraph_nodes = []
    for idx in node_idx:
        subgraph_nodes.append(idx)
        if include_neighbors:
            # out_edges returns a tuple of src and dst nodes
            neighbors = graph.out_edges(idx)[1]
            for ngh in neighbors.cpu().numpy().tolist():
                subgraph_nodes.append(ngh)
    subgraph_nodes = list(set(subgraph_nodes))
    subgraph = graph.subgraph(subgraph_nodes)
    # subgraph.copy_from_parent()
    if normalize:
        subgraph.ndata["x"] = font_util.center_and_scale_uvsolid(subgraph.ndata["x"])
    return subgraph


def get_random_subgraph_one_neighbour(
    graph, num_nodes=1, include_neighbors=True, normalize=True
):
    """
    Gets a random subgraph from the given graph
    :param graph: DGL graph
    :param num_nodes: Number of nodes to randomly extract in the subgraph
    :param include_neighbors: Whether to include one-ring neighbors of the extracted nodes so that we get connected patches
    :param normalize: Whether to normalize the graph into a cube of size 2
    """
    node_idx = random.sample(graph.nodes().cpu().numpy().tolist(), num_nodes)
    subgraph_nodes = []
    for idx in node_idx:
        subgraph_nodes.append(idx)
        if include_neighbors:
            # out_edges returns a tuple of src and dst nodes
            neighbors = graph.out_edges(idx)[1]
            for ngh in neighbors.cpu().numpy().tolist():
                subgraph_nodes.append(ngh)
                break
    subgraph_nodes = list(set(subgraph_nodes))
    subgraph = graph.subgraph(subgraph_nodes)
    # subgraph.copy_from_parent()
    if normalize:
        subgraph.ndata["x"] = font_util.center_and_scale_uvsolid(subgraph.ndata["x"])
    return subgraph


def get_subgraph(graph, num_nodes=1, hops=2, normalize=False):
    assert num_nodes >= 1
    assert hops >= 0
    node_idx = random.sample(graph.nodes().cpu().numpy().tolist(), 1)
    subgraph_nodes = [random.sample(graph.nodes().cpu().numpy().tolist(), 1)[0]]
    for h in range(hops):
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
        subgraph.ndata["x"] = font_util.center_and_scale_uvsolid(subgraph.ndata["x"])
    return subgraph
