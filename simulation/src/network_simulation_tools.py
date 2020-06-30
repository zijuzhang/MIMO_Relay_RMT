#   A collection of tools for generating simulation environments for MIMO, IRS networks
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
        self.network_surfaces = []
        for ind, elements in enumerate(network_size):
            step_surfaces = []
            for irs in range(elements):
                step_surfaces.append(IRS(previous_size,  num_elements, receiver=((ind+1)==len(network_size))))
            self.network_surfaces.append(step_surfaces)
            previous_size = elements
        self.covariance = 1j*np.zeros((num_elements, num_elements))
        self.covariance_eve = 1j*np.zeros((num_elements, num_elements))
        self.channel = 1j*np.zeros((num_elements, num_elements))
        self.channel_eve = 1j*np.zeros((num_elements, num_elements))
        self.update_phases()


    def get_channel(self, path=[], i=0):
        if len(path) == len(self.network_surfaces):
            self.channel_from_path(path)
        else:
            for ind in range(self.network_size[i]):
                if len(path) < len(self.network_surfaces):
                    self.get_channel(path=path+[ind], i=i+1)

    def channel_from_path(self, path):
        matrix = None
        previous = 0
        self.paths_check += 1
        for ind, surface_ind in enumerate(path):
            if ind == 0:
                irs = self.network_surfaces[ind][surface_ind]
                matrix = irs.phases@irs.channels[path[ind]]
            else:
                irs = self.network_surfaces[ind][surface_ind]
                matrix = irs.phases@irs.channels[previous]@matrix
            previous = surface_ind
        if path[-1] == 0:
            self.channel += matrix
        elif path[-1] == 1:
            self.channel_eve += matrix

    def get_capacity(self, transmit_covariance, secrecy=False):
        if secrecy:
            return capacity(self.channel@transmit_covariance@hermetian(self.channel)) \
                   - capacity(self.channel_eve@transmit_covariance@hermetian(self.channel_eve))
        else:
            return capacity(self.channel@transmit_covariance@hermetian(self.channel))

    def update_phases(self):
        for step in self.network_surfaces:
            for irs in step:
                irs.set_phase()
        self.channel = 1j*np.zeros(self.channel.shape)
        self.get_channel()
        self.covariance_eve = self.channel_eve @ np.conj(self.channel_eve.T)
        self.covariance = self.channel @ np.conj(self.channel.T)


class IRS:
    def __init__(self, num_previous_surfaces, num_elements, receiver=False):
        self.receiver = receiver
        self.size = num_elements
        self.channels = []
        for i in range(num_previous_surfaces):
            self.channels.append(c_rand(num_elements, num_elements))

    def set_phase(self):
        """
        Change the phase shifts of the irs
        :return:
        """
        if not self.receiver:
            self.phases = random_phase(self.size)
        else:
            self.phases = np.diag(-1j*1j*np.ones(self.size))


def water_filling(channel_matrix, power_constraint, sigma_square=1, vals=False):
    """
    Use of CVX to solve the convex water-filling problem for power allocation in a MIMO channel with CSI at
    the transmitter.
    :param covariance_matrix:
    :param power_constraint:
    :param sigma_square:
    :return:
    """
    U, Sigma, V_H = np.linalg.svd(channel_matrix)
    variables = cp.Variable(channel_matrix.shape[1])
    constraint = [cp.sum(variables) <= power_constraint]
    constraint += [variables >= 0]
    # utility = cp.sum(cp.log(1+variables*e_values/sigma_square))
    utility = []
    utility += [cp.log(1+variables[ind]*Sigma[ind]/sigma_square) for ind in range(variables.size)]
    prob = cp.Problem(cp.Maximize(cp.sum(utility)), constraint)
    prob.solve()
    covariance_matrix = hermetian(V_H)@np.diag(variables.value)@V_H
    check = np.trace(covariance_matrix)
    if vals:
        return variables.value
    else:
        return channel_matrix@covariance_matrix@hermetian(channel_matrix)
