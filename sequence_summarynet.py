import os
import sys
import numpy as np
from random import randint
from pathlib import Path
from bayesflow.simulation import Prior
from bayesflow.networks import InvertibleNetwork, SequenceNetwork
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.trainers import Trainer
from bayesflow.diagnostics import plot_losses, plot_recovery
from bayesflow.diagnostics import plot_sbc_ecdf, plot_sbc_histograms

from utils.arguments import TrainArguments
from utils.dataloaders import load_dataset

RNG = np.random.default_rng(2023)
HOME = os.getenv("HOME")


args, _ = TrainArguments().parse_known_args(sys.argv[1:])

params, series = load_dataset(args.parameters, args.series, args.limit_dataset)

prior = Prior(prior_fun=lambda: params[randint(0, len(params) - 1)], param_names=["P1", "P2", "P3"])

prior_mean, prior_std = prior.estimate_means_and_stds()
prior.plot_prior2d().savefig(Path(args.plot_dir) / "prior.png")
print(f"Prior(mean={prior_mean}, std={prior_std})")

series_mean = np.mean(series, axis=(0,2))[np.newaxis, :, np.newaxis]
series_std = np.std(series, axis=(0,2))[np.newaxis, :, np.newaxis]
print(f"Series(mean={series_mean}, std={series_std})")

SPLIT = int(args.train_val_split * len(params))
train = params[:SPLIT], series[:SPLIT]
val = params[SPLIT:], series[SPLIT:]

training_data = { "prior_draws": train[0], "sim_data": train[1] }
validation_data = { "prior_draws": val[0], "sim_data": val[1] }

def configure_input(input_dict):
    
    """ Function to configure the simulated quantities (i.e., simulator outputs)
        into a neural network-friendly (BayesFlow) format.
    """
    
    # Extract prior draws and z-standardize with previously computed means
    # prior draws are the parameters we want to estimate
    params = input_dict["prior_draws"].astype(np.float32)
    params = (params - prior_mean) / prior_std

    data = input_dict['sim_data']
    data = (data - series_mean) / series_std
    data = data.transpose(0, 2, 1)

    return {
        "parameters": params,
        "summary_conditions": data.astype(np.float32)
    }


summary_net = SequenceNetwork()
inference_net = InvertibleNetwork(num_params=len(prior.param_names), num_coupling_layers=4)
amortizer = AmortizedPosterior(inference_net, summary_net, name="lstm_amortizer")

trainer = Trainer(amortizer=amortizer, configurator=configure_input, memory=True, checkpoint_path=args.save_to)
history = trainer.train_offline(training_data, 
    epochs=100, 
    batch_size=64, 
    early_stopping=True, 
    validation_sims=validation_data, 
    save_checkpoint=True)

plot_losses(np.log(history["train_losses"]), np.log(history["val_losses"]), moving_average=True)\
    .savefig(Path(args.plot_dir) / "losses.png")

trainer.diagnose_latent2d()\
    .savefig(Path(args.plot_dir) / "diagnose_latent2d.png")

# Generate posterior draws for all simulations
validation_sims = trainer.configurator(validation_data)
post_samples = amortizer.sample(validation_sims, n_samples=100)

# Create ECDF plot
plot_sbc_ecdf(post_samples, validation_sims["parameters"], param_names=prior.param_names)\
    .savefig(Path(args.plot_dir) / "sbc_ecdf.png")
plot_sbc_ecdf(post_samples, validation_sims["parameters"], param_names=prior.param_names, stacked=True, difference=True)\
    .savefig(Path(args.plot_dir) / "sbc_ecdf_stacked.png")
plot_sbc_histograms(post_samples, validation_sims["parameters"], param_names=prior.param_names)\
    .savefig(Path(args.plot_dir) / "sbc_ecdf_stacked.png")

# TODO:TODO: De-normalize validation data using series mean/std and prior mean/std
# TODO:TODO: Plot both norm and de-norm values
# TODO:TODO: density contour plot instead of points
# TODO: Recovery with cell and fish ids, abuse recovery plot maybe to show uncertainty in each param se
plot_recovery(post_samples, validation_sims["parameters"], param_names=prior.param_names)\
    .savefig(Path(args.plot_dir) / "recovery.png")
    