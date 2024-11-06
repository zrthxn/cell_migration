import numpy as np
from bayesflow.networks import InvertibleNetwork, SequenceNetwork
from bayesflow.amortizers import AmortizedPosterior


class SequenceNetworkAmortizer(AmortizedPosterior):
    
    name = "lstm_amortizer"
    
    summary_net: SequenceNetwork
    inference_net: InvertibleNetwork
    
    def __init__(self, num_params):
        self.summary_net = SequenceNetwork()
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

        data = input_dict['sim_data']
        data = data.transpose(0, 2, 1)

        return {
            "parameters": params,
            "summary_conditions": data.astype(np.float32)
        }
