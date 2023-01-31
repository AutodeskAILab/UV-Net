import argparse
import numpy as np
import pathlib
import time
from datasets.solidletters_contrastive import SolidLettersContrastive
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from uvnet.models import Contrastive

parser = argparse.ArgumentParser("UV-Net self-supervision with contrastive learning")
parser.add_argument(
    "traintest", choices=("train", "test"), help="Whether to train or test"
)
parser.add_argument("--dataset", choices=("solidletters",), help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--size_percentage", type=float, default=1, help="Percentage of data to load")
parser.add_argument("--temperature", type=float, default=0.1, help="Temperature to use in NTXentLoss")
parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension for UV-Net's embeddings")
parser.add_argument("--out_dim", type=int, default=64, help="Output dimension for SimCLR projection head")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size; larger batches are needed for SimCLR")
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
    default="contrastive",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Define a path to save the results based date and time. E.g.
# results/args.experiment_name/0430/123103
month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_last=True,
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
)

if args.dataset == "solidletters":
    Dataset = SolidLettersContrastive
else:
    raise ValueError("Unsupported dataset")

if args.traintest == "train":
    # Train/val
    seed_everything(workers=True)
    print(
        f"""
-----------------------------------------------------------------------------------
UV-Net Contrastive Learning
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    model = Contrastive(latent_dim=args.latent_dim, out_dim=args.out_dim, temperature=args.temperature)
    train_data = Dataset(root_dir=args.dataset_path, split="train", size_percentage=args.size_percentage,)
    val_data = Dataset(root_dir=args.dataset_path, split="val", size_percentage=args.size_percentage)
    train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = val_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    trainer.fit(model, train_loader, val_loader)
else:
    # Test
    assert args.checkpoint is not None, "Expected the --checkpoint argument to be provided"
    model = Contrastive.load_from_checkpoint(args.checkpoint)
    if args.gpus is not None:
        model = model.cuda()

    test_data = Dataset(root_dir=args.dataset_path, split="test", size_percentage=args.size_percentage)
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False
    )
    test_outputs = model.get_embeddings_from_dataloader(test_loader)

    # K-means clustering on embeddings
    cluster_ami = model.clustering(test_outputs, num_clusters=test_data.num_classes(), standardize=False)
    print(f"Clustering AMI score on test set: {cluster_ami:2.3f}")
    cluster_ami = model.clustering(test_outputs, num_clusters=test_data.num_classes(), standardize=True)
    print(f"Clustering AMI score on standardized test set: {cluster_ami:2.3f}")

    # Linear SVM classification on embeddings
    train_data = Dataset(root_dir=args.dataset_path, split="train", size_percentage=args.size_percentage)
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False
    )
    train_outputs = model.get_embeddings_from_dataloader(train_loader)
    svm_acc = model.linear_svm_classification(train_outputs, test_outputs)
    print(f"Linear SVM classification accuracy (%) on test set: {svm_acc * 100.0:2.3f}")
