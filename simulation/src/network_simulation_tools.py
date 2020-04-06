import numpy as np
import copy
from testing.circle_laws import *
class Network:
    def __init__(self, num_elements, network_size):
        """
        Assume that the final size is always 1
        :param num_elements:
        :param network_size:
        """
        if network_size[-1] != 1:
            network_size.apend([1])
        previous_size = 1
        self.network_size = network_size
        self.network_channels = []
        for elements in network_size:
            step_surfaces = []
            for previous_surface in range(previous_size):
                step_surfaces.append(IRS(previous_surface,  num_elements))
            self.network_channels.append(step_surfaces)
            previous_size = elements
        self.covariance = np.zeros((num_elements, num_elements))

    def get_covariance(self, path=[0]):
        for ind, matrix in enumerate(self.network_channels):
            if len(path)  == len(self.network_channels):
                self.channel_from_path(path)
            if len(path) < len(self.network_channels):
                path.append(ind)
                self.get_covariance(path=copy.deepcopy(path))
        return self.covariance()


    def channel_from_path(self, path):
        for ind, surface_ind in enumerate(path):
            if ind == 0:
                matrix = self.network_channels[ind][surface_ind]
            else:
                matrix = matrix@self.network_channels[ind][surface_ind]
        self.covariance += matrix

class IRS:
    def __init__(self, num_previous_surfaces, num_elements):
        self.phases = 1j*np.ones(num_elements)
        self.channels = []
        for i in range(num_previous_surfaces):
            self.channels.append(c_rand(num_elements,num_elements))
