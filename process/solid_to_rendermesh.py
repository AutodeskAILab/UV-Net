import argparse
import pathlib

import numpy as np
from occwl.entity_mapper import EntityMapper
from occwl.compound import Compound
from tqdm import tqdm
import trimesh


def triangulate_with_face_mapping(solid, triangle_face_tol=0.01, angle_tol_rads=0.1):
    # Triangulate faces
    solid.triangulate_all_faces(
        triangle_face_tol=triangle_face_tol, angle_tol_rads=angle_tol_rads
    )

    verts = []
    tris = []
    # Store B-rep face index along with triangles
    mapper = EntityMapper(solid)
    tri_mapping = []
    vert_counter = 0
    for face in solid.faces():
        face_index = mapper.face_index(face)
        face_verts, face_tris = face.get_triangles()
        if len(face_tris) == 0:
            continue
        face_tris += vert_counter
        vert_counter += face_verts.shape[0]
        face_mapping = np.ones(face_tris.shape[0]) * face_index
        verts.append(face_verts)
        tris.append(face_tris)
        tri_mapping.append(face_mapping)
    if len(verts) == 0:
        return None, None, None
    verts = np.concatenate(verts, axis=0).astype(np.float32)
    if len(tris) == 0:
        return None, None, None
    tris = np.concatenate(tris, axis=0).astype(np.int32)
    tri_mapping = np.concatenate(tri_mapping, axis=-1).astype(np.int32)
    return verts, tris, tri_mapping


def process_one_file(fn, args):
    fn_stem = fn.stem
    output_path = pathlib.Path(args.output)
    output_filename = output_path.joinpath(fn_stem + ".stl")
    if output_filename.exists():
        return
    solid = Compound.load_from_step(fn)

    verts, tris, tri_mapping = triangulate_with_face_mapping(
        solid, args.triangle_face_tol, args.angle_tol_rads
    )

    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles = tris, alpha=0.8)
    # plt.show()

    # Write to numpy compressed archive
    # np.savez(
    #     str(output_path.joinpath(fn_stem + ".npz")),
    #     vertices=verts,
    #     triangles=tris,
    #     triangle_mapping=tri_mapping,
    # )
    trimesh.Trimesh(vertices=verts, faces=tris).export(str(output_filename))


def process(args):
    input_path = pathlib.Path(args.input)
    step_files = list(input_path.glob(f"{args.filename_pattern}.st*p"))
    for fn in tqdm(step_files):
        process_one_file(fn, args)


def main():
    parser = argparse.ArgumentParser(
        "Convert solid models to render (non-watertight) meshes"
    )
    parser.add_argument("input", type=str, help="Input folder of STEP files")
    parser.add_argument("output", type=str, help="Output folder of NPZ mesh files")
    parser.add_argument(
        "--triangle_face_tol",
        type=float,
        default=0.01,
        help="Tolerance between triangle and surface relative to each B-rep face",
    )
    parser.add_argument(
        "--angle_tol_rads",
        type=float,
        default=0.1,
        help="Tolerance angle between normals/tangents at triangle vertices (in radians)",
    )
    parser.add_argument(
        "--filename_pattern",
        type=str,
        default="*",
        help="Filename regex pattern to filter input files. Defaults to '*'",
    )

    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
