from argparse import ArgumentParser


class TrainArguments(ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("--network", choices=["sequencenet", "transformer"], required=True)
        self.add_argument("--parameters", type=str, required=True)
        self.add_argument("--series", action="store", nargs="*", required=True)
        self.add_argument("--limit_dataset", type=int, default=None)
        self.add_argument("--train_val_split", type=float, default=0.9)
        self.add_argument("--plot_dir", type=str, default=".")
        self.add_argument("--save_to", type=str, default=None)

class ValidationArguments(ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("--checkpoint", type=str, required=True)
        self.add_argument("--parameters", type=str, required=True)
        self.add_argument("--series", action="store", nargs="*", required=True)
        self.add_argument("--plot_dir", type=str, default=".")
