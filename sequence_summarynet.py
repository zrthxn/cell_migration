import os
import numpy as np
from dataloaders import LineReader

from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.networks import InvertibleNetwork, SequenceNetwork
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.trainers import Trainer
import bayesflow.diagnostics as diag


RNG = np.random.default_rng(2023)
HOME = os.getenv("HOME")

reader = LineReader(
    nlines=100000,
    fparams=f"{HOME}/Datasets/cell_migration/parameters.txt",
    fseries=[
        f"{HOME}/Datasets/cell_migration/dac.txt",
        f"{HOME}/Datasets/cell_migration/msd.txt"
    ]
)


prior = Prior(prior_fun=reader.sample_params, param_names=["ix", "A", "B", "C", "D", "E"])
simulator = Simulator(simulator_fun=reader.simulate_series)

prior_means, prior_stds = prior.estimate_means_and_stds()
model = GenerativeModel(prior, simulator, name="basic_covid_simulator")

_ = model(10)
# print(prior_means, prior_stds)

# As per default, the plot_prior2d function will obtain 1000 draws from the joint prior.
f = prior.plot_prior2d()




summary_net = SequenceNetwork()

inference_net = InvertibleNetwork(num_params=len(prior.param_names), num_coupling_layers=4)

amortizer = AmortizedPosterior(inference_net, summary_net, name="covid_amortizer")



def configure_input(input_dict):
    """Function to configure the simulated quantities (i.e., simulator outputs)
    into a neural network-friendly (BayesFlow) format.
    """

    # Extract prior draws and z-standardize with previously computed means
    params = input_dict["prior_draws"].astype(np.float32)
    params = (params - prior_means) / prior_stds
    
    # Convert data to logscale
    data = input_dict["sim_data"].astype(np.float32)
    # data_mean = data.mean(axis=2)
    # data_std = data.std(axis=2)
    # print(data.shape)
    # print(data_mean.shape)
    # print(data_std)
    # data = (data - data_mean) / data_std
    # print(data)

    # Remove a batch if it contains nan, inf or -inf
    # idx_keep = np.all(np.isfinite(data), axis=(1, 2))
    # if not np.all(idx_keep):
    #     print("Invalid value encountered...removing from batch")
    

    # Extract prior draws and z-standardize with previously computed means
    # prior draws are the parameters we want to estimate
    params = input_dict["prior_draws"].astype(np.float32)
    params = (params - prior_means) / prior_stds

    data = input_dict['sim_data']
    data = (data - series_mean) / series_std
    
    data = data.transpose(0, 2, 1)
    batch_size, num_timesteps, x_dim  = data.shape
    
    # add time encoding to the data x
    time_encoding = np.linspace(0, 1, num_timesteps)
    time_encoding_batched = np.zeros((batch_size, num_timesteps, x_dim+1))
    time_encoding_batched[:,:,:x_dim] = data
    time_encoding_batched[:,:,x_dim] = time_encoding

    # x = input_dict['sim_data']
    # batch_size, _, num_timesteps  = x.shape
    
    # # add time encoding to the data x
    # time_encoding = np.linspace(0, 1, num_timesteps)
    # time_encoding_batched = np.tile(time_encoding, (batch_size, 1, 1))

    return {
        "parameters": params,
        "summary_conditions": data.astype(np.float32)  # for the sequence network
    }

trainer = Trainer(amortizer=amortizer, generative_model=model, configurator=configure_input, memory=True)

amortizer.summary()

offline_data = model(len(reader.series))
history = trainer.train_offline(offline_data, epochs=30, batch_size=32, validation_sims=200)


f = diag.plot_losses(np.log(history["train_losses"]), np.log(history["val_losses"]), moving_average=True)

f = trainer.diagnose_latent2d()

# f = trainer.diagnose_sbc_histograms()

# Generate some validation data
validation_sims = trainer.configurator(model(batch_size=300))

# Generate posterior draws for all simulations
post_samples = amortizer.sample(validation_sims, n_samples=100)

# Create ECDF plot
f = diag.plot_sbc_ecdf(post_samples, validation_sims["parameters"], param_names=prior.param_names)

f = diag.plot_sbc_histograms(post_samples, validation_sims["parameters"], param_names=prior.param_names)

post_samples = amortizer.sample(validation_sims, n_samples=1000)
f = diag.plot_recovery(post_samples, validation_sims["parameters"], param_names=prior.param_names)
