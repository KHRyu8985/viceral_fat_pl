import autorootcwd
import lightning.pytorch as pytorch_lightning
from monai.utils import set_determinism
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandFlipd,
    CropForegroundd,
    Compose,
    Spacingd,
    EnsureType,
    AsDiscrete,
    AsDiscreted,
    KeepLargestConnectedComponent,
)

from src.models.swin_unetr import SwinUNETRv2
from src.models.segresnet import SegResNet  # Add this import
from src.models.unetr import UNETR
from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    LearningRateFinder,
    StochasticWeightAveraging,
)

from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss  # Add DiceCELoss import
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import yaml
import click
import numpy as np
import nibabel as nib
from pathlib import Path
import csv
from dvclive.lightning import DVCLiveLogger

print_config()


def load_data_splits(yaml_path, fold_number=1):
    # Read the YAML file
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    # Extract train, val, test splits from the specified fold
    fold_key = f"fold_{fold_number}"
    fold = data["cross_validation_splits"][fold_number-1][fold_key]
    train_split = fold["train"]
    val_split = fold["val"]
    test_split = fold["test"]

    # Add folder path to each entry in the splits and create dictionaries
    base_dir = os.path.dirname(yaml_path)
    train_split = [
        {
            "image": os.path.join(base_dir, entry, "CT.nii.gz"),
            "label": os.path.join(base_dir, entry, "vf.nii.gz"),
            "seg": os.path.join(base_dir, entry, "combined_seg.nii.gz"),
        }
        for entry in train_split
    ]
    val_split = [
        {
            "image": os.path.join(base_dir, entry, "CT.nii.gz"),
            "label": os.path.join(base_dir, entry, "vf.nii.gz"),
            "seg": os.path.join(base_dir, entry, "combined_seg.nii.gz"),
        }
        for entry in val_split
    ]
    test_split = [
        {
            "image": os.path.join(base_dir, entry, "CT.nii.gz"),
            "label": os.path.join(base_dir, entry, "vf.nii.gz"),
            "seg": os.path.join(base_dir, entry, "combined_seg.nii.gz"),
        }
        for entry in test_split
    ]
    print(f"Loaded data splits from {yaml_path} for fold {fold_number}")
    print(f"Train: {len(train_split)} subjects")
    print(f"Validation: {len(val_split)} subjects")
    print(f"Test: {len(test_split)} subjects")

    return train_split, val_split, test_split


class FatSegmentModel(pytorch_lightning.LightningModule):
    """Proposed model"""

    def __init__(
        self,
        arch_name="UNETR",
        loss_fn="DiceFocalLoss",
        batch_size=1,
        lr=1e-3,
        patch_size=(96, 96, 96),
        fold_number=1,  # Add fold_number as an argument
    ):
        super().__init__()

        self.fold_number = fold_number  # Store the fold number

        if arch_name == "SwinUNETR":
            self._model = SwinUNETRv2(
                img_size=patch_size,
                in_channels=1,
                out_channels=2,
                feature_size=48,
                use_checkpoint=True,
                label_nc=8,
            )
            weight = torch.load("weights/model_swinvit.pt", weights_only=True)
            # Extract only the SwinTransformer (swinViT) weights
            swin_vit_weights = {
                k: v for k, v in weight.items() if k.startswith("swinViT")
            }

            # Load the SwinTransformer weights into the model
            self._model.swinViT.load_state_dict(swin_vit_weights, strict=False)

            print(
                "Using pretrained self-supervised Swin UNETR SwinTransformer weights!"
            )

        elif arch_name == "SegResNet":  # Add this condition
            self._model = SegResNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                init_filters=16,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                dropout_prob=0.2,
                label_nc=8,
            )
            print("Using SegResNet architecture!")

        elif arch_name == "UNETR":
            self._model = UNETR(
                in_channel=1,
                out_channel=2,
                img_size=(96, 96, 96),
                label_nc=8,
            )

        else:
            raise ValueError(f"Unsupported architecture name: {arch_name}")

        if loss_fn == "DiceLoss":
            self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        elif loss_fn == "DiceCELoss":
            self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        elif loss_fn == "DiceFocalLoss":
            self.loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True)
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")

        self.post_pred = Compose(
            [
                EnsureType("tensor", device="cpu"),
                AsDiscrete(argmax=True, to_onehot=2),
            ]
        )
        self.post_label = Compose(
            [EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)]
        )
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.hausdorff_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean"
        )
        self.mean_iou_metric = MeanIoU(include_background=False, reduction="mean")
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.batch_size = batch_size
        self.lr = lr
        self.patch_size = patch_size
        self.results_folder = Path("results")  # Define the results folder
        self.test_step_outputs = []

    def forward(self, x, seg):
        # Modify the forward method to include seg
        return self._model(x, seg)

    def prepare_data(self):
        # set up the correct data path
        train_files, val_files, test_files = load_data_splits(
            yaml_path="data/KU-PET-CT/data_splits.yaml", fold_number=self.fold_number
        )

        set_determinism(seed=0)

        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label", "seg"]),
                EnsureChannelFirstd(keys=["image", "label", "seg"]),
                Orientationd(keys=["image", "label", "seg"], axcodes="RAS"),
                # Spacingd(
                #    keys=["image", "label", "seg"],
                #    pixdim=(1.5, 1.5, 2.0),
                #    mode=("bilinear", "nearest", "nearest"),
                # ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-200,
                    a_max=100,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                AsDiscreted(
                    keys=["seg"],
                    to_onehot=8,
                ),
                CropForegroundd(keys=["image", "label", "seg"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label", "seg"],
                    label_key="label",
                    spatial_size=self.patch_size,
                    pos=1,
                    neg=1,
                    num_samples=6,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=["image", "label", "seg"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label", "seg"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label", "seg"]),
                EnsureChannelFirstd(keys=["image", "label", "seg"]),
                Orientationd(keys=["image", "label", "seg"], axcodes="RAS"),
                # Spacingd(
                #    keys=["image", "label", "seg"],
                #    pixdim=(1.5, 1.5, 2.0),
                #    mode=("bilinear", "nearest", "nearest"),
                # ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-200,
                    a_max=100,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                AsDiscreted(
                    keys=["seg"],
                    to_onehot=8,
                ),
                CropForegroundd(keys=["image", "label", "seg"], source_key="image"),
            ]
        )

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=0.1,
            num_workers=4,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=0.1,
            num_workers=4,
        )

        self.test_ds = CacheDataset(
            data=test_files,
            transform=val_transforms,
            cache_rate=0.1,
            num_workers=4,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=4)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_ds, batch_size=1, num_workers=4)
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels, segs = (
            batch["image"],
            batch["label"],
            batch["seg"],
        )  # Include seg
        output = self.forward(images, segs)  # Pass seg to forward
        loss = self.loss_function(output, labels)
        metrics = loss.item()
        self.log(
            "train_loss",
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, segs = (
            batch["image"],
            batch["label"],
            batch["seg"],
        )  # Include seg

        inputs = torch.cat((images, segs), dim=1)

        roi_size = self.patch_size
        sw_batch_size = 4
        outputs = sliding_window_inference(
            inputs,
            roi_size,
            sw_batch_size,
            lambda x: self.forward(
                x[:, :1, ...], x[:, 1:, ...]
            ),  # Split before forward
        )

        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        self.dice_metric(y_pred=outputs, y=labels)
        self.hausdorff_metric(y_pred=outputs, y=labels)
        self.mean_iou_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        mean_val_hausdorff = self.hausdorff_metric.aggregate().item()
        mean_val_iou = self.mean_iou_metric.aggregate().item()

        self.dice_metric.reset()
        self.hausdorff_metric.reset()
        self.mean_iou_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)

        log_dict = {
            "val_dice": mean_val_dice,
            "val_hausdorff": mean_val_hausdorff,
            "val_iou": mean_val_iou,
            "val_loss": mean_val_loss,
        }

        self.log_dict(log_dict)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}, "
            f"hausdorff: {mean_val_hausdorff:.4f}, "
            f"iou: {mean_val_iou:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.validation_step_outputs.clear()  # free memory

    def save_results(self, inputs, outputs, labels, filename_prefix="results"):
        # Ensure outputs and labels are numpy arrays
        save_folder = self.results_folder / "test"
        os.makedirs(save_folder, exist_ok=True)

        inputs_np = inputs.detach().cpu().numpy().squeeze()
        outputs_np = outputs.detach().cpu().numpy().squeeze()[1]
        labels_np = labels.detach().cpu().numpy().squeeze()[1]

        # inputs_np = np.moveaxis(inputs_np, 0, -1)
        # outputs_np = np.moveaxis(outputs_np, 0, -1)
        # labels_np = np.moveaxis(labels_np, 0, -1)

        # Save inputs as NIfTI
        inputs_nifti = nib.Nifti1Image(
            inputs_np,
            np.array([[0.98, 0, 0, 0], [0, 0.98, 0, 0], [0, 0, 2.8, 0], [0, 0, 0, 1]]),
        )
        nib.save(inputs_nifti, save_folder / f"{filename_prefix}_inputs.nii.gz")

        # Save outputs as NIfTI
        outputs_nifti = nib.Nifti1Image(
            outputs_np,
            np.array([[0.98, 0, 0, 0], [0, 0.98, 0, 0], [0, 0, 2.8, 0], [0, 0, 0, 1]]),
        )
        nib.save(outputs_nifti, save_folder / f"{filename_prefix}_outputs.nii.gz")

        # Save labels as NIfTI
        labels_nifti = nib.Nifti1Image(
            labels_np,
            np.array([[0.98, 0, 0, 0], [0, 0.98, 0, 0], [0, 0, 2.8, 0], [0, 0, 0, 1]]),
        )
        nib.save(labels_nifti, save_folder / f"{filename_prefix}_labels.nii.gz")

        print(f"Results saved to: {save_folder}")
        print(f"Inputs: {save_folder / f'{filename_prefix}_inputs.nii.gz'}")
        print(f"Outputs: {save_folder / f'{filename_prefix}_outputs.nii.gz'}")
        print(f"Labels: {save_folder / f'{filename_prefix}_labels.nii.gz'}")

    def test_step(self, batch, batch_idx):
        images, labels, segs = (
            batch["image"],
            batch["label"],
            batch["seg"],
        )  # Include seg
        roi_size = self.patch_size
        sw_batch_size = 4
        # Concatenate images and segs along the channel dimension
        inputs = torch.cat((images, segs), dim=1)

        outputs = sliding_window_inference(
            inputs,
            roi_size,
            sw_batch_size,
            lambda x: self.forward(
                x[:, :1, ...], x[:, 1:, ...]
            ),  # Split before forward
        )

        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        filename = batch["image"].meta["filename_or_obj"][0]
        patient_id = int(
            filename.split("/")[-2]
        )  # Converts '01404213' from the path to an integer

        # Save results
        self.save_results(
            images, outputs[0], labels[0], filename_prefix=f"Subj_{patient_id}"
        )

        self.dice_metric(y_pred=outputs, y=labels)
        self.hausdorff_metric(y_pred=outputs, y=labels)
        self.mean_iou_metric(y_pred=outputs, y=labels)

        # Calculate metrics for this batch
        dice_score = self.dice_metric.aggregate().item()
        hausdorff_score = self.hausdorff_metric.aggregate().item()
        mean_iou_score = self.mean_iou_metric.aggregate().item()
        # Extract patient ID from filename
        d = {
            "test_dice": dice_score,
            "test_hausdorff": hausdorff_score,
            "test_iou": mean_iou_score,
            "patient_id": patient_id,
        }
        self.test_step_outputs.append(d)

        self.dice_metric.reset()
        self.hausdorff_metric.reset()
        self.mean_iou_metric.reset()

        return d

    def on_test_epoch_end(self):
        # Calculate mean metrics
        dice_scores = [x["test_dice"] for x in self.test_step_outputs]
        hausdorff_scores = [x["test_hausdorff"] for x in self.test_step_outputs]
        iou_scores = [x["test_iou"] for x in self.test_step_outputs]

        mean_dice = np.mean(dice_scores)
        mean_hausdorff = np.mean(hausdorff_scores)
        mean_iou = np.mean(iou_scores)

        # Save detailed results to CSV
        results_file = self.results_folder / "test" / "test_results.csv"
        with open(results_file, "w", newline="") as csvfile:
            fieldnames = ["dice_score", "hausdorff_score", "iou_score", "patient_id"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.test_step_outputs:
                # Create a new dict without the filename
                result_with_filename = {
                    "dice_score": result["test_dice"],
                    "hausdorff_score": result["test_hausdorff"],
                    "iou_score": result["test_iou"],
                    "patient_id": result["patient_id"],
                }
                writer.writerow(result_with_filename)
        # Write summary row at the end of test results
        with open(results_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(
                {
                    "dice_score": f"{mean_dice:.4f} ± {np.std(dice_scores):.4f}",
                    "hausdorff_score": f"{mean_hausdorff:.4f} ± {np.std(hausdorff_scores):.4f}",
                    "iou_score": f"{mean_iou:.4f} ± {np.std(iou_scores):.4f}",
                    "patient_id": f"AVG ± STD",
                }
            )

        print(f"\nTest Results Summary:")
        print(f"Mean Dice Score: {mean_dice:.4f}")
        print(f"Mean Hausdorff Distance: {mean_hausdorff:.4f}")
        print(f"Mean IoU Score: {mean_iou:.4f}")
        print(f"Detailed results saved to: {results_file}")

        # Clear the outputs
        self.test_step_outputs.clear()


@click.command()
@click.option(
    "--arch_name",
    type=click.Choice(["UNet", "SegResNet", "UNETR", "SwinUNETR"]),
    default="UNETR",
    help="Choose the architecture name for the model.",
)
@click.option(
    "--loss_fn",
    type=click.Choice(["DiceLoss", "DiceCELoss", "DiceFocalLoss"]),
    default="DiceFocalLoss",
    help="Choose the loss function for training.",
)
@click.option(
    "--max_epochs",
    type=int,
    default=300,
    help="Set the maximum number of training epochs.",
)
@click.option(
    "--check_val_every_n_epoch",
    type=int,
    default=10,
    help="Set the frequency of validation checks (in epochs).",
)
@click.option(
    "--gpu_number", type=int, default=0, help="Set the GPU index to use for training."
)
@click.option(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to a checkpoint file to load for inference.",
)
@click.option("--fold_number", type=int, default=1, help="Specify the fold number for training.")
def main(
    arch_name, loss_fn, max_epochs, check_val_every_n_epoch, gpu_number, checkpoint_path, fold_number
):
    # set up loggers and checkpoints
    log_dir = f"results/proposed_{arch_name}_{fold_number}"
    os.makedirs(log_dir, exist_ok=True)

    # Create a list with a single GPU index
    devices = [gpu_number]

    # Set up callbacks
    callbacks = [
        StochasticWeightAveraging(
            swa_lrs=[1e-4], annealing_epochs=5, swa_epoch_start=100
        )
    ]
    dvc_logger = DVCLiveLogger(log_model=True, dir=log_dir, report="html")

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        devices=devices,
        max_epochs=max_epochs,
        logger=dvc_logger,  # Use DVC logger instead of CSV logger
        enable_checkpointing=True,
        benchmark=True,
        accumulate_grad_batches=5,
        precision="bf16-mixed",
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=1,
        callbacks=callbacks,
        default_root_dir=log_dir,
    )

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        # Load model from checkpoint and perform inference
        print(f"Loading checkpoint from {checkpoint_path}")
        net = FatSegmentModel.load_from_checkpoint(
            checkpoint_path, arch_name=arch_name, loss_fn=loss_fn
        )
        net.results_folder = Path(log_dir)  # Set results folder path

        net.prepare_data()
        print(f"Inferring and testing model")
        trainer.test(model=net, dataloaders=net.test_dataloader())

    else:
        # Initialize model for training
        net = FatSegmentModel(arch_name=arch_name, loss_fn=loss_fn, batch_size=1, fold_number=fold_number)
        net.results_folder = Path(log_dir)  # Set results folder path

        # Train the model
        trainer.fit(net)
        trainer.save_checkpoint(os.path.join(log_dir, "final_model.ckpt"))
        trainer.test(model=net, dataloaders=net.test_dataloader())


if __name__ == "__main__":
    main()
