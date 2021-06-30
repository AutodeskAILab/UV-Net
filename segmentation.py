import argparse
import pathlib

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.fusiongallery import FusionGalleryDataset
from datasets.mfcad import MFCADDataset
from uvnet.models import Segmentation

parser = argparse.ArgumentParser("UV-Net solid model face segmentation")
parser.add_argument(
    "traintest", choices=("train", "test"), help="Whether to train or test"
)
parser.add_argument(
    "--dataset", choices=("mfcad", "fusiongallery"), help="Segmentation dataset"
)
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--random_rotate",
    action="store_true",
    help="Whether to randomly rotate the solids in 90 degree increments along the canonical axes",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="segmentation",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", dirpath=str(results_path), filename="best", save_last=True,
)

trainer = Trainer.from_argparse_args(
    args, callbacks=[checkpoint_callback], logger=TensorBoardLogger(str(results_path)),
)

if args.dataset == "mfcad":
    Dataset = MFCADDataset
elif args.dataset == "fusiongallery":
    Dataset = FusionGalleryDataset

if args.traintest == "train":
    # Train/val
    model = Segmentation(num_classes=Dataset.num_classes())
    train_data = Dataset(
        root_dir=args.dataset_path, split="train", random_rotate=args.random_rotate
    )
    val_data = Dataset(root_dir=args.dataset_path, split="val")
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    trainer.fit(model, train_loader, val_loader)
else:
    # Test
    model = Segmentation.load_from_checkpoint(args.checkpoint)
    test_data = Dataset(
        root_dir=args.dataset_path, split="test", random_rotate=args.random_rotate
    )
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    results = trainer.test(model=model, test_dataloaders=[test_loader], verbose=False)
    print(f"Segmentation IoU (%) on test set: {results[0]['test_iou'] * 100.0}")
