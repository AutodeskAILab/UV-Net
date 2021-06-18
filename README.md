# UV-Net: Learning from Boundary Representations

This repository contains code for the paper:

["UV-Net: Learning from Boundary Representations."](https://arxiv.org/abs/2006.10211) Pradeep Kumar Jayaraman, Aditya Sanghi, Joseph G. Lambourne, Karl D.D. Willis, Thomas Davies, Hooman Shayani, Nigel Morris. CVPR 2021.

![Teaser](docs/img/Teaser.png)

UV-Net is a neural network designed to operate directly on Boundary representation (B-rep) data from 3D CAD models. The B-rep format is widely used in the design, simulation and manufacturing industries to enable sophisticated and precise CAD modeling operations. However, B-rep data presents some unique challenges when used with neural networks due to the complexity of the data structure and its support for various disparate geometric and topological entities.

In UV-Net, we represent the geometry stored in the edges (curves) and faces (surfaces) of the B-rep using 1D and 2D UV-grids, a structured set of points sampled by taking uniform steps in the parameter domain. 1D and 2D convolutional neural networks can be applied on these UV-grids to encode the edge and face geometry.

![UVGrid](docs/img/UVGrid.png)

The topology is represented using a face-adjacency graph where features from the face UV-grids are stored as node features, and features from the edge UV-grids are stored as edge features. A graph neural network is then used to message pass these features to obtain embeddings for faces, edges and the entire solid model.

![MessagePassing](docs/img/MessagePassing.png)

## Data
The network consumes [DGL](https://dgl.ai/)-based face-adjacency graphs, where each B-rep face is mapped to a node, and each B-rep edge is mapped to a edge. The face UV-grids are expected as node features and edge UV-grids as edge features. For example, the UV-grid features from our face-adjacency graph representation can be accessed as follows:

```python
from dgl.data.utils import load_graphs

graph = load_graphs(filename)[0]
graph.ndata["x"]  # num_facesx10x10x7 face UV-grids (we use 10 samples along the u- and v-directions of the surface)
                  # The first three channels are the point coordinates, next three channels are the surface normals, and
                  # the last channel is a trimming mask set to 1 if the point is in the visible part of the face and 0 otherwise
graph.edata["x"]  # num_edgesx10x6 edge UV-grids (we use 10 samples along the u-direction of the curve)
                  # The first three channels are the point coordinates, next three channels are the curve tangents
```

### SolidLetters

SolidLetters is a synthetic dataset of ~96k solids created by extruding and filleting fonts. It has class labels (alphabets), and style labels (font name and upper/lower case) for each solid.

The dataset of face-adjacency graphs can be downloaded from here:


We also provide solid models in SMT format that can be processed with the Autodesk Fusion 360 software, and in STEP format that can be read using OpenCascade and its Python bindings [pythonOCC](https://github.com/tpaviot/pythonocc-core).

To train the UV-Net classification model on the data:

1. Extract the graphs to a folder, say `/path/to/solidletters/`. Please refer to the license in `/path/to/solidletters/license.pdf`.

2. There should be three subfolders:

- `/path/to/solidletters/smt` contains the solid models in `.smt` format that can be read by a proprietory Autodesk solid modeling kernel and the Fusion 360 software.
- `/path/to/solidletters/step` contains the solid models in `.step` format that can be read using OpenCascade and its Python bindings [pythonOCC](https://github.com/tpaviot/pythonocc-core).
- `/path/to/solidletters/bin` contains the derived face-adjacency graphs in DGL's `.bin` format with UV-grids stored as node and edge features. This is the data that gets passed to UV-Net for training and testing.

3. Pass the `solidletters` folder to the `--dataset_path` argument in the classification script and set `--dataset` to `solidletters`.

### MFCAD

The original solid model data is available here in STEP format: github.com/hducg/MFCAD
We provide pre-processed DGL graphs in `.bin` format to train UV-Net on this dataset.

1. Download and extract the data to a folder, say `/path/to/mfcad/` from here:

2. Pass the `mfcad` folder to the `--dataset_path` argument in the segmentation script and set `--dataset` to `mfcad`.

### Fusion 360 Gallery segmentation

We provide pre-processed DGL graphs in `.bin` format to train UV-Net on the Fusion 360 Gallery segmentation task.

1. Download and extract the dataset to a folder, say `/path/to/fusionallery/` from here:

2. Pass the `fusiongallery` folder to the `--dataset_path` argument in the segmentation script and set `--dataset` to `fusiongallery`.


## Processing your own data
See guide [here](process/README.md) to convert your own solid model data (in STEP format) to `.bin` files that are understood by UV-Net.

## Training

The classification model can be trained using:
```
python classification.py train --dataset solidletters --dataset_path /path/to/solidletters --epochs 100 --batch_size 64
```

Only the SolidLetters dataset is currently supported for classification.

The segmentation model can be trained similarly:
```
python segmentation.py train --dataset mfcad --dataset_path /path/to/graphs --epochs 100 --batch_size 64
```

The MFCAD and Fusion 360 Gallery segmentation datasets are supported.

The logs will be stored in a folder called `classification_logs` or `segmentation_logs` based on the experiment, and can be monitored with Tensorboard:

```
tensorboard --logdir *_logs
```

## Testing
The best checkpoints based on the smallest validation loss is saved in the `classification_checkpoints` or `segmentation_checkpoints` folder. The checkpoints can be used to test the model as follows:

```
python segmentation.py test --dataset mfcad --dataset_path /path/to/dataset --checkpoint ./segmentation_checkpoints/best-epoch=65-val_loss=0.00.ckpt
```

## Citation

```
@inproceedings{jayaraman2021uvnet,
 title = {UV-Net: Learning from Boundary Representations},
 author = {Pradeep Kumar Jayaraman and Aditya Sanghi and Joseph G. Lambourne and Karl D.D. Willis and Thomas Davies and Hooman Shayani and Nigel Morris},
 eprint = {2006.10211},
 eprinttype = {arXiv},
 eprintclass = {cs.CV},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 year = {2021}
}
```