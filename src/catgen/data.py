from __future__ import annotations

from pathlib import Path

import lightning as L
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"


class ImgDataset(Dataset):
    def __init__(self, paths, image_size=64, augment=False):
        self.paths = list(paths)
        tfs = []
        if augment:
            tfs.append(transforms.RandomHorizontalFlip())
        tfs += [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
        self.tf = transforms.Compose(tfs)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def read_split(name: str) -> list[Path]:
    fp = SPLITS / name
    if not fp.exists():
        raise FileNotFoundError(f"{fp} missing; run scripts/prepare_data.py")
    return [RAW / line.strip() for line in fp.read_text().splitlines() if line.strip()]


class CatDataModule(L.LightningDataModule):
    def __init__(self, split="train_3000.txt", image_size=64, batch_size=32, num_workers=2, augment=True):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_ds = ImgDataset(
            read_split(self.hparams.split),
            image_size=self.hparams.image_size,
            augment=self.hparams.augment,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            persistent_workers=self.hparams.num_workers > 0,
        )
