import numpy as np
from bayesflow.networks import InvertibleNetwork, TimeSeriesTransformer
from bayesflow.amortizers import AmortizedPosterior


class TimeseriesTransformerAmortizer(AmortizedPosterior):
    
    name = "transformer_amortizer"
    
    summary_net: TimeSeriesTransformer
    inference_net: InvertibleNetwork
    
    def __init__(self, input_dim, num_params):
        self.summary_net = TimeSeriesTransformer(input_dim)
        self.inference_net = InvertibleNetwork(num_params, num_coupling_layers=4)
        self(self.inference_net, self.summary_net, name=self.name)
    
    @staticmethod
    def configurator(input_dict):
        """ Function to configure the simulated quantities (i.e., simulator outputs)
            into a neural network-friendly (BayesFlow) format.
        """

        # Extract prior draws and z-standardize with previously computed means
        # prior draws are the parameters we want to estimate
        params = input_dict["prior_draws"].astype(np.float32)

        x = input_dict['sim_data']
        x = x.transpose(0, 2, 1)
        batch_size, num_timesteps, x_dim  = x.shape
        
        # add time encoding to the data x
        time_encoding = np.linspace(0, 1, num_timesteps)
        time_encoding_batched = np.zeros((batch_size, num_timesteps, x_dim+1))
        time_encoding_batched[:,:,:x_dim] = x
        time_encoding_batched[:,:,x_dim] = time_encoding

        return {
            "parameters": params,
            "summary_conditions": time_encoding_batched.astype(np.float32),
        }
