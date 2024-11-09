import sys
import numpy as np
from bayesflow.trainers import Trainer
from bayesflow.diagnostics import plot_losses, plot_recovery
from bayesflow.diagnostics import plot_sbc_ecdf, plot_sbc_histograms

from models import SequenceNetworkAmortizer, TimeseriesTransformerAmortizer
from arguments import TrainArguments
from utils.dataloaders import load_dataset


args, _ = TrainArguments().parse_known_args(sys.argv[1:])

params, series = load_dataset(args.parameters, args.series, args.limit)

_, num_params = params.shape
prior_mean = np.mean(params, axis=(0,1))
prior_std = np.std(params, axis=(0,1))

series_mean = np.mean(series, axis=(0,2))[np.newaxis, :, np.newaxis]
series_std = np.std(series, axis=(0,2))[np.newaxis, :, np.newaxis]

# Normalize Parameters and series
params = (params - prior_mean)  / prior_std
series = (series - series_mean) / series_std

SPLIT = int(args.train_val_split * len(params))
train = params[:SPLIT], series[:SPLIT]
val = params[SPLIT:], series[SPLIT:]

training_data = { "prior_draws": train[0], "sim_data": train[1] }
validation_data = { "prior_draws": val[0], "sim_data": val[1] }

# Choose network type
if args.network == "sequencenet":
    amortizer = SequenceNetworkAmortizer(num_params)
elif args.network == "transformer":
    amortizer = TimeseriesTransformerAmortizer(series.shape[1] + 1, num_params)
else:
    raise ValueError("Unknown network type!")

trainer = Trainer(amortizer, configurator=amortizer.configurator, memory=True, checkpoint_path=args.save_to)
history = trainer.train_offline(training_data, 
    epochs=args.epochs, 
    batch_size=64, 
    early_stopping=True, 
    validation_sims=validation_data, 
    save_checkpoint=True)

plot_losses(np.log(history["train_losses"]), np.log(history["val_losses"]), moving_average=True)\
    .savefig(args.plot_dir / "training_losses.png")

# Generate posterior draws for all simulations
prior = amortizer.configurator(validation_data)
posterior = amortizer.sample(prior, n_samples=100)

# Create ECDF plot
plot_sbc_ecdf(posterior, prior["parameters"])\
    .savefig(args.plot_dir / "training_sbc_ecdf.png")
plot_sbc_ecdf(posterior, prior["parameters"], stacked=True, difference=True)\
    .savefig(args.plot_dir / "training_sbc_ecdf_stacked.png")
plot_sbc_histograms(posterior, prior["parameters"])\
    .savefig(args.plot_dir / "training_sbc_ecdf_histogram.png")

# TODO:TODO: De-normalize validation data using series mean/std and prior mean/std
# TODO:TODO: Plot both norm and de-norm values
# TODO:TODO: density contour plot instead of points
# TODO: Recovery with cell and fish ids, abuse recovery plot maybe to show uncertainty in each param se
plot_recovery(posterior, prior["parameters"])\
    .savefig(args.plot_dir / "training_recovery.png")
