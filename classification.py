import argparse
import pathlib

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.solidletters import SolidLetters
from uvnet.models import Classification

parser = argparse.ArgumentParser("UV-Net solid model classification")
parser.add_argument(
    "traintest", choices=("train", "test"), help="Whether to train or test"
)
parser.add_argument("--dataset", choices=("solidletters",), help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
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
    args,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(results_path.joinpath(args.experiment_name)),
)

if args.dataset == "solidletters":
    Dataset = SolidLetters
else:
    raise ValueError("Unsupported dataset")


if args.traintest == "train":
    # Train/val
    model = Classification(num_classes=Dataset.num_classes())
    train_data = Dataset(root_dir=args.dataset_path, split="train")
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
    test_data = Dataset(root_dir=args.dataset_path, split="test")
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = Classification.load_from_checkpoint(args.checkpoint)
    results = trainer.test(model=model, test_dataloaders=[test_loader], verbose=False)
    print(
        f"Classification accuracy (%) on test set: {results[0]['test_acc_epoch'] * 100.0}"
    )
