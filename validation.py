import os
import sys
import numpy as np
from random import randint
from pathlib import Path
from bayesflow.simulation import Prior
from bayesflow.trainers import Trainer
from bayesflow.diagnostics import plot_recovery
from bayesflow.diagnostics import plot_sbc_ecdf, plot_sbc_histograms

from models import SequenceNetworkAmortizer, TimeseriesTransformerAmortizer
from arguments import ValidationArguments
from utils.dataloaders import load_dataset


args, _ = ValidationArguments().parse_known_args(sys.argv[1:])

params, series = load_dataset(args.parameters, args.series)

prior = Prior(prior_fun=lambda: params[randint(0, len(params) - 1)], param_names=["P1", "P2", "P3"])
prior_mean, prior_std = prior.estimate_means_and_stds()

series_mean = np.mean(series, axis=2)[:, :, np.newaxis]
series_std = np.std(series, axis=2)[:, :, np.newaxis]

# Normalize Parameters and series
params = (params - prior_mean)  / prior_std
series = (series - series_mean) / series_std

if args.limit:
    params = params[:args.limit]
    series = series[:args.limit]

# Choose network type
if args.network == "sequencenet":
    amortizer = SequenceNetworkAmortizer(num_params=len(prior.param_names))
elif args.network == "transformer":
    amortizer = TimeseriesTransformerAmortizer(input_dim=series.shape[1]+1, num_params=len(prior.param_names))
else:
    raise ValueError("Unknown network type!")

trainer = Trainer(amortizer=amortizer, configurator=amortizer.configurator, memory=True, checkpoint_path=args.checkpoint)
trainer.load_pretrained_network()

# Generate posterior draws for all simulations
validation_sims = trainer.configurator({ "prior_draws": params, "sim_data": series })
post_samples = amortizer.sample(validation_sims, n_samples=100)

# # Create ECDF plot
# plot_sbc_ecdf(post_samples, validation_sims["parameters"], param_names=prior.param_names)\
#     .savefig(Path(args.plot_dir) / "validation_sbc_ecdf.png")
# plot_sbc_ecdf(post_samples, validation_sims["parameters"], param_names=prior.param_names, stacked=True, difference=True)\
#     .savefig(Path(args.plot_dir) / "validation_sbc_ecdf_stacked.png")
# plot_sbc_histograms(post_samples, validation_sims["parameters"], param_names=prior.param_names)\
#     .savefig(Path(args.plot_dir) / "validation_sbc_ecdf_histogram.png")

# TODO:TODO: De-normalize validation data using series mean/std and prior mean/std
# TODO:TODO: Plot both norm and de-norm values
# TODO:TODO: density contour plot instead of points
# TODO: Recovery with cell and fish ids, abuse recovery plot maybe to show uncertainty in each param se
plot_recovery(post_samples, validation_sims["parameters"], param_names=prior.param_names)\
    .savefig(Path(args.plot_dir) / "validation_recovery.png")