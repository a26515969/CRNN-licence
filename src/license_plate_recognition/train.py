import os
from pathlib import Path

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

import license_plate_recognition.data.alphabet as alphabet
from license_plate_recognition.data.dataset import LicencePlateDataset
import license_plate_recognition.data.utils as utils
from license_plate_recognition.trainer import OCRModule
from license_plate_recognition.data.dataset import custom_collate_fn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, dir_okay=True),
    default="data/train",
    show_default=True,
    help="Directory with data.",
)
@click.option(
    "--val-split",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.05,
    show_default=True,
    help="Proportion of the train dataset to include in the validation split.",
)
@click.option(
    "--n-epochs", type=int, default=10, show_default=True, help="Upper epoch limit."
)
@click.option(
    "--batch-size", type=int, default=8, show_default=True, help="Batch size."
)############################################################################################原本64 測試時先改成8
@click.option(
    "--lr",
    type=float,
    default=0.001,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "--ckpt-dir",
    type=click.Path(exists=True, dir_okay=True),
    default="checkpoints",
    show_default=True,
    help="Directory to save the model file.",
)
@click.option(
    "--ckpt-name",
    type=str,
    default="best_checkpoint",
    show_default=True,
    help="Checkpoint filename.",
)
@click.option(
    "--logs-dir",
    type=click.Path(),
    default="logs",
    show_default=True,
    help="Save directory for TensorBoard Logger.",
)
@click.option(
    "--exp-name",
    type=str,
    default="experiment1",
    show_default=True,
    help="Experiment name.",
)
@click.option(
    "--label-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to the label mapping txt file.",
)
def train(
    input_dir: Path,
    val_split: float,
    n_epochs: int,
    batch_size: int,
    lr: float,
    ckpt_dir: Path,
    ckpt_name: str,
    logs_dir: Path,
    exp_name: str,
    label_file: Path,
) -> None:
    """Script that trains a model and saves it to a file."""

    # create dictionary
    dictionary = utils.Dictionary()
    for char in (
        alphabet.blank + alphabet.provinces + alphabet.alphabets + alphabet.numbers
    ):
        dictionary.add_char(char)

    # create label converter
    converter = utils.LabelConverter(dictionary=dictionary)
    
    # create datasets
    click.echo("Loading datasets...")
    train_dataset = LicencePlateDataset(
    img_dir=input_dir,
    label_file=label_file or os.path.join(input_dir, "labels.txt"),
    dictionary=dictionary,
    train=True,
    transform=transforms.ToTensor(),
    )
    test_dataset = LicencePlateDataset(
    img_dir="data/test",
    label_file="data/test/labels.txt" ,
    dictionary=dictionary,
    train=False,
    transform=transforms.ToTensor(),
    )

    # split train dataset into train and valid sets
    
    n_val = int(val_split * len(train_dataset))
    n_train = len(train_dataset) - n_val
    train_dataset, val_dataset = random_split(  # type: ignore
        train_dataset, [n_train, n_val]
    )
    
    click.echo(f"Number of images in train set: {len(train_dataset)}")
    click.echo(f"Number of images in valid set: {len(val_dataset)}")
    click.echo(f"Number of images in test set: {len(test_dataset)}")

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    module = OCRModule(
        learning_rate=lr, dictionary_size=len(dictionary), label_converter=converter
    )
    ####改成儲存在資料夾李############################################################################################################
    base_dir = ckpt_dir
    existing = [d for d in os.listdir(base_dir) if d.startswith("exp")]
    exp_id = len(existing) + 1
    exp_dir = os.path.join(base_dir, f"exp{exp_id}")
    os.makedirs(exp_dir, exist_ok=True)
    ############################################################################################################################################
    logger = pl.loggers.TensorBoardLogger(save_dir=logs_dir, name=exp_name)
    #checkpoint_callback = ModelCheckpoint(
        #monitor="val_loss",mode="min", save_top_k=1, dirpath=ckpt_dir, filename=ckpt_name
    #)
    checkpoint_callback = ModelCheckpoint(
    dirpath=exp_dir,         # 存放目錄
    filename="{epoch}-{val_loss:.2f}",    # 固定名稱，產出 checkpoints/last_checkpoint.ckpt
    save_top_k=1,                  # 儲存一個檔案
    save_last=True                 # 明確指定存最後一個
)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        max_epochs=n_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,ckpt_path=None)#ckpt_path="checkpoints/last.ckpt
    trainer.test(
        model=module,
        dataloaders=test_dataloader,
        ckpt_path=os.path.join(exp_dir, "last.ckpt")
        
    )
###################原本的是ckpt_path=f"{os.path.join(ckpt_dir, ckpt_name)}.ckpt",