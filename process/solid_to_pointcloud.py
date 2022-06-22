import argparse
from itertools import repeat
from multiprocessing.pool import Pool
import pathlib
import signal

import numpy as np
import trimesh
from occwl.compound import Compound
from tqdm import tqdm

from process.solid_to_rendermesh import triangulate_with_face_mapping


def process_one_file(arguments):
    fn, args = arguments
    if fn.stat().st_size == 0:
        return None
    fn_stem = fn.stem
    output_path = pathlib.Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    try:
        solid = Compound.load_from_step(fn)
    except Exception as e:
        print(e)
        return

    verts, tris, tri_mapping = triangulate_with_face_mapping(solid)

    mesh = trimesh.Trimesh(vertices=verts, faces=tris)
    points, face_indices = trimesh.sample.sample_surface(mesh, args.num_points)
    points_to_face_mapping = tri_mapping[face_indices]

    # import matplotlib.pyplot as plt
    # from matplotlib.colors import Normalize
    # from matplotlib.cm import tab20
    # from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # colors = tab20(points_to_face_mapping)
    # norm = Normalize(
    #     vmin=np.amin(points_to_face_mapping), vmax=np.amax(points_to_face_mapping)
    # )
    # ax.scatter(
    #     points[:, 0], points[:, 1], points[:, 2], c=colors, norm=norm,
    # )
    # plt.show()

    # Write to numpy compressed archive
    np.savez(
        str(output_path.joinpath(fn_stem + ".npz")),
        points=points,
        point_mapping=points_to_face_mapping,
    )


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process(args):
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    step_files = list(input_path.glob("*.st*p"))
    # for fn in tqdm(step_files):
    #     process_one_file(fn, args)
    pool = Pool(processes=args.num_processes, initializer=initializer)
    try:
        results = list(tqdm(pool.imap(process_one_file, zip(step_files, repeat(args))), total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    print(f"Processed {len(results)} files.")


def main():
    parser = argparse.ArgumentParser("Convert solid models to point clouds")
    parser.add_argument("input", type=str, help="Input folder of STEP files")
    parser.add_argument(
        "output", type=str, help="Output folder of NPZ point cloud files"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=2048,
        help="Number of points in the point cloud",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes to use",
    )
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
