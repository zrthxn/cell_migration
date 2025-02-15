from argparse import ArgumentParser
from pathlib import Path


class TrainArguments(ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("--network", choices=["sequencenet", "tstransformer"], required=True)
        self.add_argument("--parameters", type=str, required=True)
        self.add_argument("--series", action="store", nargs="*", required=True)
        self.add_argument("--batch_size", type=int, default=64)
        self.add_argument("--early_stopping", type=bool, default=False)
        self.add_argument("--limit", type=int, default=None)
        self.add_argument("--epochs", type=int, default=100)
        self.add_argument("--train_val_split", type=float, default=0.9)
        self.add_argument("--plot_dir", type=Path, default=".")
        self.add_argument("--save_to", type=Path, default=None)


class ValidationArguments(ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("--network", choices=["sequencenet", "tstransformer"], required=True)
        self.add_argument("--checkpoint", type=Path, required=True)
        self.add_argument("--parameters", type=str, required=True)
        self.add_argument("--series", action="store", nargs="*", required=True)
        self.add_argument("--limit", type=int, default=None)
        self.add_argument("--plot_dir", type=Path, default=".")
