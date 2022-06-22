import argparse
import numpy as np
import pathlib
import os
import os.path as osp
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import dgl
from dgl.data.utils import load_graphs


def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)


def bounding_box_uvsolid(inp: torch.Tensor):
    pts = inp[:, :, :, :3].reshape((-1, 3))
    mask = inp[:, :, :, 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    return bounding_box_pointcloud(pts)


def plot_uvsolid(uvsolid: torch.Tensor, ax, normals=False):
    """
    Plot the loaded UV solid features to a MPL 3D Axes
    :param uvsolid torch.Tensor: Features loaded from *.feat file of shape [#faces, #u, #v, 10]
    :param ax matplotlib Axes3D: 3D Axes object for plotting
    """
    assert len(uvsolid.shape) == 4  # faces x #u x #v x 10
    bbox = bounding_box_uvsolid(uvsolid)
    bbox_diag = torch.norm(bbox[1] - bbox[0]).item()
    num_faces = uvsolid.size(0)
    for i in range(num_faces):
        pts = uvsolid[i, :, :, :3].cpu().detach().numpy().reshape((-1, 3))
        nor = uvsolid[i, :, :, 3:6].cpu().detach().numpy().reshape((-1, 3))
        mask = uvsolid[i, :, :, 6].cpu().detach().numpy().reshape(-1)
        point_indices_inside_faces = mask == 1
        pts = pts[point_indices_inside_faces, :]
        if normals:
            nor = nor[point_indices_inside_faces, :]
            ax.quiver(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                nor[:, 0],
                nor[:, 1],
                nor[:, 2],
                length=0.075 * bbox_diag,
            )
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])


def plot_uvsolid_edges(graph: dgl.DGLGraph, ax, tangents=False):
    """
    Plot the loaded UV solid's edge features to a MPL 3D Axes
    :param graph: dgl.DGLGraph: DGL Graph containing the graph with UV-grids as node features and 1D UV-grids as edge features
    :param ax matplotlib Axes3D: 3D Axes object for plotting
    """
    face_feat = graph.ndata["x"]
    if graph.edata.get("x") is None:
        print("Edge features not found")
        return
    edge_feat = graph.edata["x"]
    if edge_feat.shape[0] == 0:
        return
    assert edge_feat.shape[2] in (3, 6), edge_feat.shape  # edges x #u x 3/6
    bbox = bounding_box_uvsolid(face_feat)
    bbox_diag = torch.norm(bbox[1] - bbox[0]).item()
    num_edges = graph.edata["x"].size(0)
    for i in range(num_edges):
        pts = graph.edata["x"][i, :, :3].cpu().detach().numpy().reshape((-1, 3))
        if tangents:
            tgt = graph.edata["x"][i, :, 3:6].cpu().detach().numpy().reshape((-1, 3))
            ax.quiver(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                tgt[:, 0],
                tgt[:, 1],
                tgt[:, 2],
                length=0.075 * bbox_diag,
            )
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])


def plot_faceadj_graph(graph: dgl.DGLGraph, ax):
    """
    Plot the face-adj graph to a MPL 3D Axes
    :param graph: dgl.DGLGraph: DGL Graph containing the graph with UV-grids as node features
    :param ax matplotlib Axes3D: 3D Axes object for plotting
    """
    assert len(graph.ndata["x"].shape) == 4  # faces x #u x #v x 10
    src, dst = graph.edges()
    for i in range(src.size(0)):
        center_idx = graph.ndata["x"].size(1) // 2
        src_pt = graph.ndata["x"][src[i], center_idx, center_idx, :3]
        dst_pt = graph.ndata["x"][dst[i], center_idx, center_idx, :3]
        ax.plot(
            (src_pt[0], dst_pt[0]),
            (src_pt[1], dst_pt[1]),
            zs=(src_pt[2], dst_pt[2]),
            color="k",
            linewidth=2,
            marker="o",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Visualize UV-grids and face adj graphs for testing"
    )
    parser.add_argument("dir", type=str, default=None, help="Directory of bin files")
    parser.add_argument(
        "--hide_plots",
        action="store_true",
        help="Whether to hide the plots, and only save them",
    )
    parser.add_argument(
        "--plot_face_normals",
        action="store_true",
        help="Whether to plot face normals",
    )
    parser.add_argument(
        "--plot_edge_tangents",
        action="store_true",
        help="Whether to plot edge tangents",
    )
    args, _ = parser.parse_known_args()

    if args.dir is None:
        raise ValueError("Expected a valid directory to be provided")
    folder = pathlib.Path(args.dir)
    bin_files = folder.glob("*.bin")

    for f in bin_files:
        graph = load_graphs(str(f))[0][0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plt.gca().view_init(35, 90)
        ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])

        plot_uvsolid(
            graph.ndata["x"],
            ax,
            normals=args.plot_face_normals,
        )
        plot_faceadj_graph(
            graph,
            ax,
        )
        plot_uvsolid_edges(graph, ax, tangents=args.plot_edge_tangents)
        plt.savefig(folder.joinpath(f.stem + ".jpg"))
        if not args.hide_plots:
            plt.show()