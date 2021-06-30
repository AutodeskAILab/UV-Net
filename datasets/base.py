from torch.utils.data import Dataset, DataLoader
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
        self.files = []
        self.graphs = []
        for fn in tqdm(file_paths):
            if not fn.exists():
                continue
            graph = self.load_one_graph(fn)
            self.graphs.append(graph)
            self.files.append(fn)
        if center_and_scale:
            self.center_and_scale()
    
    def load_one_graph(self, file_path):
        graph = load_graphs(str(file_path))[0][0]
        return graph

    def center_and_scale(self):
        for i in range(len(self.graphs)):
            self.graphs[i].ndata["x"] = util.center_and_scale_uvgrid(
                self.graphs[i].ndata["x"]
            )

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.random_rotate:
            rotation = util.get_random_rotation()
            graph.ndata["x"] = util.rotate_uvgrid(graph.ndata["x"], rotation)
            graph.edata["x"] = util.rotate_uvgrid(graph.edata["x"], rotation)
        return graph

    def _collate(self, batch):
        bg = dgl.batch(batch)
        return bg

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,  # Can be set to non-zero on Linux
            drop_last=True,
        )
