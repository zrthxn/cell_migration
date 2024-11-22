import numpy as np
from bayesflow.networks import InvertibleNetwork, SequenceNetwork
from bayesflow.amortizers import AmortizedPosterior


class SequenceNetworkAmortizer(AmortizedPosterior):
    
    name = "lstm_amortizer"
    
    def __init__(self, num_params: int):
        summary_net = SequenceNetwork()
        inference_net = InvertibleNetwork(num_params, num_coupling_layers=4)
        super().__init__(inference_net, summary_net, name=self.name)
    
    @staticmethod
    def configurator(input_dict):
        """ Function to configure the simulated quantities (i.e., simulator outputs)
            into a neural network-friendly (BayesFlow) format.
        """
    
        params = input_dict["prior_draws"].astype(np.float32)

        data = input_dict["sim_data"].astype(np.float32)
        data = data.transpose(0, 2, 1)

        return {
            "parameters": params,
            "summary_conditions": data
        }
