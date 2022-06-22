# Processing your own data

We provide scripts to process your own STEP file data into the DGL bin format that UV-Net consumes, point clouds in NPZ format and render meshes (non-watertight meshes) in STL format.

Example usage:

```
cd /path/to/uv_net
python -m process.solid_to_graph /path/to/input/step_files /path/to/output/bin_graphs
```

Other scripts can be run similarly. For more details, run the script with the `--help` argument.