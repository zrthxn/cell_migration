#%%
import json
import pandas as pd
import numpy as np
from bayesflow.trainers import Trainer
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from models import SequenceNetworkAmortizer, TimeseriesTransformerAmortizer
from utils.arrays import loosestack

#%%
class args:
    network = "tstransformer"
    checkpoint = f".checkpoints/tstransformer-maxi"
    parameters = "/Users/alisamar/Datasets/cell_migration/maxi/parameters.txt"
    
    # Experimental dataset
    series = [
        "/Users/alisamar/Datasets/cell_migration/experimental/msd.txt",
        "/Users/alisamar/Datasets/cell_migration/experimental/dac.txt"]
    
    cell_ids = "/Users/alisamar/Datasets/cell_migration/experimental/parameters.txt"
    emb_ids = "/Users/alisamar/Datasets/cell_migration/experimental/cell_emb_ids.json"
    gdata = "/Users/alisamar/Datasets/cell_migration/experimental/gdata.json"

#%%
params = np.loadtxt(args.parameters)[:400_000]

series = loosestack([np.loadtxt(f) for f in args.series])
series = np.swapaxes(series, 0, 1)

_, num_params = params.shape
num_samples, _, timesteps = series.shape

#%%
cell_ids = np.loadtxt(args.cell_ids).astype(int)
gdata = pd.read_json(args.gdata, orient="records")

#%%
# Normalize
prior_mean = np.mean(params, axis=(0,1))
prior_std = np.std(params, axis=(0,1))

series_mean = np.mean(series, axis=(0,2))[np.newaxis, :, np.newaxis]
series_std = np.std(series, axis=(0,2))[np.newaxis, :, np.newaxis]

# Normalize Parameters and series
params = (params - prior_mean)  / prior_std
series = (series - series_mean) / series_std

#%%
# Choose network type
if args.network == "sequencenet":
    amortizer = SequenceNetworkAmortizer(num_params)
elif args.network == "tstransformer":
    amortizer = TimeseriesTransformerAmortizer(series.shape[1] + 1, num_params)
else:
    raise ValueError("Unknown network type!")

trainer = Trainer(amortizer, configurator=amortizer.configurator, memory=True, checkpoint_path=args.checkpoint)
trainer.load_pretrained_network()

#%%
# Generate posterior draws for all simulations
prior = amortizer.configurator({ "sim_data": series, "prior_draws": params })
posterior = amortizer.sample(prior, n_samples=100)

#%%
posterior = posterior * prior_std + prior_mean
prior["parameters"] = prior["parameters"] * prior_std + prior_mean

#%%
# Uncertainty Plot
param_names = [ f"P{i+1}" for i in range(num_params) ]
# Make date+embryo id unique
date_emb_ids = cell_ids[:, 0]*1000 + cell_ids[:, 1]
unique_ids, counts = np.unique(date_emb_ids, return_counts=True)

f, axs = plt.subplots(ncols=len(unique_ids), nrows=num_params, figsize=(40, 15), width_ratios=counts / num_samples)

for param_post, pname, pix in zip(posterior.swapaxes(0,2), param_names, list(range(num_params))):
    postrange = ((param_post - prior_mean)/ prior_std) * (prior_std * 1.1) + prior_mean
    lower, upper = postrange.min(), postrange.max()
    for eix, embryoid in enumerate(unique_ids):

        labels = cell_ids[(cell_ids[:,0]*1000 + cell_ids[:,1]) == embryoid][:, 2]
        
        axs[pix][eix].boxplot(
            param_post[:, date_emb_ids == embryoid],
            tick_labels=list(labels))

        date = gdata[gdata.datenum == embryoid//1000]["#date"].unique()[0]
        axs[pix][eix].text(1, upper*0.95, f"[Date {embryoid//1000} - {date}]", horizontalalignment="left")
        axs[pix][eix].text(1, upper*0.9, f"Embyro {embryoid%1000}", horizontalalignment="left")
        axs[pix][eix].set_ylim(lower, upper)
        # axs[pix][eix].grid()
                
        if eix != 0:
            axs[pix][eix].set_yticks([])
        
        axs[pix][0].set_ylabel(pname)

f.tight_layout()
f.savefig(f"{args.checkpoint}/uncertainty.png")

# %%
