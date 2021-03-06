import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractclassmethod
from communication_util.general_tools import *


class metric(ABC):
    """
    Interface for metrics to be used in the viterbi algorithm implementation
    """
    def __init__(self, received):
        self.received = received

    @classmethod
    def metric(cls, index):
        """
        Given an index, this function should return the vector of conditional probabilities of the channel states
        represented by the Viterbi Algorithm trellis.
        :param index:
        :return:
        """
        pass

class GaussianChannelMetric(metric):
    """
    Metric for LTI AWGN Channel with quantization
    :param survivor_paths:
    :param index:
    :param transmit_alphabet:
    :param channel_output:
    :param cir:
    :return:
    """

    def __init__(self, csi, received, quantization_level=None):
        metric.__init__(self, received)
        self.parameters = csi/np.linalg.norm(csi)
        self.quantization_level = quantization_level

    def metric(self, index, states):
        costs = []
        for state in states:
            channel_output = self.received[0, index]
            predicted = np.dot(np.asarray(state), self.parameters.T)
            if self.quantization_level is not None:
                cost = quantizer(np.linalg.norm((predicted - channel_output)), self.quantization_level)
                costs.append(cost)
            else:
                costs.append(np.linalg.norm((predicted - channel_output)))
        return np.asarray(costs)


class NeuralNetworkMixtureModelMetric(metric):

    """
    Metric for ViteribNet based detector
    :param survivor_paths:
    :param index:
    :param transmit_alphabet:
    :param channel_output:
    :param cir:
    :return:
    """
    def __init__(self, neual_net, mm, received, input_length=1):
        metric.__init__(self, received)
        self.nn = neual_net
        self.mm = mm
        self.nn_input_size = input_length-1

    def metric(self, index, state=None):
        #   Be careful using the PyTorch parser with scalars!
        torch_input = torch.tensor([self.received[0, index]]).float()
        nn = self.nn(torch_input).flatten().detach().numpy()
        # mm = self.mm(self.received[0, index])
        #   Need to change sign to align with argmin used in viterbi
        # return -nn*mm
        return - nn



