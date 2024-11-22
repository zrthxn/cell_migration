import numpy as np
from bayesflow.networks import InvertibleNetwork, TimeSeriesTransformer
from bayesflow.amortizers import AmortizedPosterior


class TimeseriesTransformerAmortizer(AmortizedPosterior):
    
    name = "transformer_amortizer"
    
    def __init__(self, input_dim: int, num_params: int):
        summary_net = TimeSeriesTransformer(input_dim)
        inference_net = InvertibleNetwork(num_params, num_coupling_layers=4)
        super().__init__(inference_net, summary_net, name=self.name)
    
    @staticmethod
    def configurator(input_dict):
        """ Function to configure the simulated quantities (i.e., simulator outputs)
            into a neural network-friendly (BayesFlow) format.
        """

        # Extract prior draws and z-standardize with previously computed means
        # prior draws are the parameters we want to estimate
        params = input_dict["prior_draws"].astype(np.float32)

        series = input_dict["sim_data"].astype(np.float32)
        series = series.transpose(0, 2, 1)
        
        # add time encoding to the data
        batchsize, num_timesteps, _  = series.shape
        time_encoding = np.linspace(0, 1, num_timesteps, dtype=np.float32).reshape((-1, 1))
        series = np.append(series, np.tile(time_encoding, (batchsize, 1, 1)), axis=2)

        return {
            "parameters": params,
            "summary_conditions": series,
        }