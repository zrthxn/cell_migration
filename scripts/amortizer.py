# %%
import os
import numpy as np
from random import randint

RNG = np.random.default_rng(2023)
HOME = os.getenv("HOME")

# %%
SIZE = 400_000

params = np.loadtxt(f"{HOME}/p_planarian/cell_migration_bayesflow_3_1M/parameters.txt")

series = [
    np.loadtxt(f"{HOME}/p_planarian/cell_migration_bayesflow_3_1M/dac.txt"),
    np.loadtxt(f"{HOME}/p_planarian/cell_migration_bayesflow_3_1M/msd.txt")
]

# Since series can be different length, we slice them to the same length
SLICETO = min([ s.shape[1] for s in series ])
series = [ s[:, :SLICETO] for s in series ]
series = np.array(series)

# Put in to the right shape
series = np.swapaxes(series, 0, 1)

params = params[:SIZE]
series = series[:SIZE]

print(params.shape)
print(series.shape)

# %%
series_mean = np.mean(series, axis=(0,2))[np.newaxis, :, np.newaxis]
series_std = np.std(series, axis=(0,2))[np.newaxis, :, np.newaxis]

# %%
from bayesflow.simulation import Prior

prior = Prior(prior_fun=lambda: params[randint(0, len(params) - 1)], param_names=["P1", "P2", "P3"])

prior_means, prior_stds = prior.estimate_means_and_stds()

# %%
# As per default, the plot_prior2d function will obtain 1000 draws from the joint prior.
f = prior.plot_prior2d()

# %%
from bayesflow.networks import InvertibleNetwork, TimeSeriesTransformer
from bayesflow.amortizers import AmortizedPosterior

# %%
summary_net = TimeSeriesTransformer(input_dim=series.shape[1]+1)

# %%
inference_net = InvertibleNetwork(num_params=len(prior.param_names), num_coupling_layers=4)

# %%
amortizer = AmortizedPosterior(inference_net, summary_net, name="covid_amortizer")

# %%
from bayesflow.trainers import Trainer


def configure_input(input_dict):

    """ Function to configure the simulated quantities (i.e., simulator outputs)
        into a neural network-friendly (BayesFlow) format.
    """

    # Extract prior draws and z-standardize with previously computed means
    # prior draws are the parameters we want to estimate
    params = input_dict["prior_draws"].astype(np.float32)
    params = (params - prior_means) / prior_stds

    x = input_dict['sim_data']
    x = (x - series_mean) / series_std
    x = x.transpose(0, 2, 1)
    batch_size, num_timesteps, x_dim  = x.shape
    # add time encoding to the data x
    time_encoding = np.linspace(0, 1, num_timesteps)
    time_encoding_batched = np.zeros((batch_size, num_timesteps, x_dim+1))
    time_encoding_batched[:,:,:x_dim] = x
    time_encoding_batched[:,:,x_dim] = time_encoding

    # x = input_dict['sim_data']
    # batch_size, _, num_timesteps  = x.shape
    # # add time encoding to the data x
    # time_encoding = np.linspace(0, 1, num_timesteps)
    # time_encoding_batched = np.tile(time_encoding, (batch_size, 1, 1))

    return {
        "parameters": params,
        "summary_conditions": time_encoding_batched.astype(np.float32),  # for the transformer
        # "summary_conditions": x.astype(np.float32)  # for the sequence network
        # "summary_conditions": np.concatenate((x, time_encoding_batched), axis=1).reshape(batch_size, num_timesteps, -1)
    }

trainer = Trainer(amortizer=amortizer, generative_model=None, configurator=configure_input, memory=True)

# %%
# train, val = reader.train_val_split(0.9)
SPLIT = int(0.9 * len(params))
train = params[:SPLIT], series[:SPLIT]
val = params[SPLIT:], series[SPLIT:]

offline_data = {
    "prior_draws": train[0],
    "sim_data": train[1],
}

validation_data = {
    "prior_draws": val[0],
    "sim_data": val[1],
}

history = trainer.train_offline(offline_data, epochs=100, batch_size=64, early_stopping=True, validation_sims=validation_data)

# %%
import bayesflow.diagnostics as diag

f = diag.plot_losses(np.log(history["train_losses"]), np.log(history["val_losses"]), moving_average=True)

# %%
f = trainer.diagnose_latent2d()

# %%
# f = trainer.diagnose_sbc_histograms()

# %%
validation_sims = trainer.configurator(validation_data)

# Generate posterior draws for all simulations
post_samples = amortizer.sample(validation_sims, n_samples=100)

# %%
# Create ECDF plot
f = diag.plot_sbc_ecdf(post_samples, validation_sims["parameters"], param_names=prior.param_names)

# %%
f = diag.plot_sbc_ecdf(
    post_samples, validation_sims["parameters"], stacked=True, difference=True, legend_fontsize=12, fig_size=(6, 5)
)

# %%
f = diag.plot_sbc_histograms(post_samples, validation_sims["parameters"], param_names=prior.param_names)

# %%
from diagnostics import plot_recovery

post_samples = amortizer.sample(validation_sims, n_samples=1000)
f = diag.plot_recovery(post_samples, validation_sims["parameters"], param_names=prior.param_names)


# from upycli import command

# @command
# def train(
#   params: str,  
# ):
#     ...