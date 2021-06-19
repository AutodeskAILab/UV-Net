import argparse

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
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)

args = parser.parse_args()

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="classification_checkpoints",
    filename="best-{epoch}-{val_loss:.2f}",
    save_last=True,
)
trainer = Trainer(
    gpus=1,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=1,
    max_epochs=args.epochs,
    logger=TensorBoardLogger("classification_logs"),
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
    train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=True)
    val_loader = val_data.get_dataloader(batch_size=args.batch_size, shuffle=False)
    trainer.fit(model, train_loader, val_loader)
else:
    # Test
    test_data = Dataset(root_dir=args.dataset_path, split="test")
    test_loader = test_data.get_dataloader(batch_size=args.batch_size, shuffle=False)
    model = Classification.load_from_checkpoint(args.checkpoint)
    results = trainer.test(model=model, test_dataloaders=[test_loader], verbose=False)
    print(
        f"Classification accuracy (%) on test set: {results[0]['test_acc_epoch'] * 100.0}"
    )