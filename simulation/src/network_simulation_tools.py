import copy
from src.circle_laws import *
import cvxpy as cp
from src.lin_alg import *


class Network:
    def __init__(self, num_elements, network_size, eavesdropper=False):
        """
        Assume that the final size is always 1
        :param num_elements:
        :param network_size:
        """
        self.paths_check = 0
        if network_size[-1] != 1 and not eavesdropper:
            network_size.append(1)
        elif network_size[-1] != 2 and eavesdropper:
            network_size.append(2)
        previous_size = 1
        self.network_size = network_size
        self.network_channels = []
        for elements in network_size:
            step_surfaces = []
            for irs in range(elements):
                step_surfaces.append(IRS(previous_size,  num_elements))
            self.network_channels.append(step_surfaces)
            previous_size = elements
        self.covariance = 1j*np.zeros((num_elements, num_elements))
        self.transmit_covariance_eve = 1j*np.zeros((num_elements, num_elements))
        self.channel = 1j*np.zeros((num_elements, num_elements))
        self.channel_eve = 1j*np.zeros((num_elements, num_elements))
        self.get_channel()



    def get_channel_covariance(self, eve=False):
        """
        Numerical Error will usually give some imaginary components here
        :return:
        """
        self.covariance_eve = self.channel_eve @ np.conj(self.channel_eve.T)
        self.covariance = self.channel @ np.conj(self.channel.T)
        if eve:
            return self.covariance, self.covariance_eve
        else:
            return self.covariance


    def get_transmit_covariance(self, transmit_powers, eve=False):
        self.paths_check = 0
        self.transmit_powers = transmit_powers
        self.transmit_covariance = self.channel @ self.transmit_powers @ np.conj(self.channel.T)
        self.transmit_covariance_eve = self.channel_eve @ self.transmit_powers @ np.conj(self.channel_eve.T)
        if not eve:
            return self.transmit_covariance
        if eve:
            return self.transmit_covariance, self.transmit_covariance_eve


    def get_channel(self, path=[], i=0):
        if len(path) == len(self.network_channels):
            self.channel_from_path(path)
        else:
            for ind in range(self.network_size[i]):
                if len(path) < len(self.network_channels):
                    self.get_channel(path=path+[ind], i=i+1)


    def channel_from_path(self, path):
        matrix = None
        previous = 0
        self.paths_check += 1
        for ind, surface_ind in enumerate(path):
            if ind == 0:
                irs = self.network_channels[ind][surface_ind]
                matrix = irs.channels[path[ind]]
            else:
                irs = self.network_channels[ind][surface_ind]
                matrix = irs.channels[path[previous]]@matrix
            previous = surface_ind
        if path[-1] == 0:
            self.channel += matrix
        elif path[-1] == 1:
            self.channel_eve += matrix

    def get_capacity(self, secrecy=False):
        if secrecy:
            return capacity(self.transmit_covariance) - capacity(self.transmit_covariance_eve)
        else:
            return capacity(self.transmit_covariance)


class IRS:
    def __init__(self, num_previous_surfaces, num_elements):
        self.phases = 1j*-1j*np.ones(num_elements)  # Strange but seems to be the way to initialize to real here
        self.channels = []
        for i in range(num_previous_surfaces):
            self.channels.append(c_rand(num_elements,num_elements))


def water_filling(covariance_matrix, power_constraint, sigma_square=1e-2):
    """
    May not be using cvx correctly but currently this only works for very small problems
    :param covariance_matrix:
    :param power_constraint:
    :param sigma_square:
    :return:
    """
    e_values = np.real(np.linalg.eig(covariance_matrix)[0])  # Remove img residual from numerical errors
    variables = cp.Variable(covariance_matrix.shape[0])
    constraint = [cp.sum(variables) <= power_constraint]
    constraint += [variables >= 0]
    # utility = cp.sum(cp.log(1+variables*e_values/sigma_square))
    utility = []
    utility += [cp.log(1+variables[ind]*e_values[ind]/sigma_square) for ind in range(e_values.size)]
    prob = cp.Problem(cp.Maximize(cp.sum(utility)), constraint)
    prob.solve(verbose=True)
    return variables.value

# def water_filling_eavesdropper(covariance_bob, covariance_eve):
#     # First find all eigenvalues and see