import os
import sys
import numpy as np
from bayesflow.trainers import Trainer

from arguments import ValidationArguments
from utils.dataloaders import load_dataset
from models import SequenceNetworkAmortizer, TimeseriesTransformerAmortizer
from plotting import plot_recovery

args, _ = ValidationArguments().parse_known_args(sys.argv[1:])

if not os.path.exists(args.plot_dir):
    os.makedirs(args.plot_dir)

params, series = load_dataset(args.parameters, args.series, args.limit)

_, num_params = params.shape
prior_mean = np.mean(params, axis=(0,1))
prior_std = np.std(params, axis=(0,1))

series_mean = np.mean(series, axis=(0,2))[np.newaxis, :, np.newaxis]
series_std = np.std(series, axis=(0,2))[np.newaxis, :, np.newaxis]

# Normalize Parameters and series
params = (params - prior_mean)  / prior_std
series = (series - series_mean) / series_std

# Choose network type
if args.network == "sequencenet":
    amortizer = SequenceNetworkAmortizer(num_params)
elif args.network == "tstransformer":
    amortizer = TimeseriesTransformerAmortizer(series.shape[1] + 1, num_params)
else:
    raise ValueError("Unknown network type!")

trainer = Trainer(amortizer, configurator=amortizer.configurator, memory=True, checkpoint_path=args.checkpoint)
trainer.load_pretrained_network()

# Generate posterior draws for all simulations
prior = amortizer.configurator({ "prior_draws": params, "sim_data": series })
posterior = amortizer.sample(prior, n_samples=100)

posterior = posterior * prior_std + prior_mean
prior["parameters"] = prior["parameters"] * prior_std + prior_mean

# TODO:TODO: Plot both norm and de-norm values
plot_recovery(posterior, prior["parameters"], plot_type="kde", ranges=[[0, 1.5], [0, 6], [0, 15]])\
    .savefig(args.plot_dir / "validation_recovery.png")
