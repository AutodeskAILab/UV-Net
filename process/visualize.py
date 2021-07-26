import argparse
import numpy as np
from occwl.viewer import Viewer
from occwl.io import load_step
from occwl.edge import Edge
from occwl.solid import Solid

import torch
import dgl
from dgl.data.utils import load_graphs


def draw_face_uvgrids(solid, graph, viewer):
    face_uvgrids = graph.ndata["x"].view(-1, 7)
    points = []
    normals = []
    for idx in range(face_uvgrids.shape[0]):
        # Don't draw points outside trimming loop
        if face_uvgrids[idx, -1] == 0:
            continue
        points.append(face_uvgrids[idx, :3].cpu().numpy())
        normals.append(face_uvgrids[idx, 3:6].cpu().numpy())

    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)

    bbox = solid.box()
    max_length = max(bbox.x_length(), bbox.y_length(), bbox.z_length())

    # Draw the points
    viewer.display_points(
        points, color=(51.0 / 255.0, 0, 1), marker="point", scale=2*max_length
    )

    # Draw the normals
    for pt, nor in zip(points, normals):
        viewer.display(Edge.make_line_from_points(pt, pt + nor * 0.05 * max_length), color=(51.0 / 255.0, 0, 1))


def draw_edge_uvgrids(solid, graph, viewer):
    edge_uvgrids = graph.edata["x"].view(-1, 6)
    points = []
    tangents = []
    for idx in range(edge_uvgrids.shape[0]):
        points.append(edge_uvgrids[idx, :3].cpu().numpy())
        tangents.append(edge_uvgrids[idx, 3:6].cpu().numpy())

    points = np.asarray(points, dtype=np.float32)
    tangents = np.asarray(tangents, dtype=np.float32)

    bbox = solid.box()
    max_length = max(bbox.x_length(), bbox.y_length(), bbox.z_length())

    # Draw the points
    viewer.display_points(points, color=(1, 0, 1), marker="point", scale=2*max_length)

    # Draw the tangents
    for pt, tgt in zip(points, tangents):
        viewer.display(Edge.make_line_from_points(pt, pt + tgt * 0.1 * max_length), color=(1, 0, 1))


def draw_graph_edges(solid, graph, viewer):
    src, dst = graph.edges()
    num_u = graph.ndata["x"].shape[1]
    num_v = graph.ndata["x"].shape[2]
    bbox = solid.box()
    max_length = max(bbox.x_length(), bbox.y_length(), bbox.z_length())

    for s, d in zip(src, dst):
        src_pt = graph.ndata["x"][s, num_u // 2, num_v // 2, :3].cpu().numpy()
        dst_pt = graph.ndata["x"][d, num_u // 2, num_v // 2, :3].cpu().numpy()
        # Make a cylinder for each edge connecting a pair of faces
        up_dir = dst_pt - src_pt
        height = np.linalg.norm(up_dir)
        if height > 1e-3:
            v.display(
                Solid.make_cylinder(
                    radius=0.01 * max_length, height=height, base_point=src_pt, up_dir=up_dir
                ),
                color="BLACK",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Visualize UV-grids and face adj graphs for testing"
    )
    parser.add_argument("solid", type=str, help="Solid STEP file")
    parser.add_argument("graph", type=str, help="Graph BIN file")
    args = parser.parse_args()

    solid = load_step(args.solid)[0]
    graph = load_graphs(args.graph)[0][0]

    v = Viewer(backend="wx")
    # Draw the solid
    v.display(solid, transparency=0.5, color=(0.2, 0.2, 0.2))
    # Draw the face UV-grids
    draw_face_uvgrids(solid, graph, viewer=v)
    # Draw the edge UV-grids
    draw_edge_uvgrids(solid, graph, viewer=v)
    # Draw face-adj graph edges
    draw_graph_edges(solid, graph, viewer=v)

    v.fit()
    v.show()
