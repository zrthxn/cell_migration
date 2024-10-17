from argparse import ArgumentParser


class TrainArguments(ArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("--parameters", type=str, required=True)
        self.add_argument("--series", action="nargs+", required=True)
        self.add_argument("--limit_dataset", type=int, default=None)
        self.add_argument("--test_val_split", type=float, default=0.9)
        self.add_argument("--plot_dir", type=str, required=True)
