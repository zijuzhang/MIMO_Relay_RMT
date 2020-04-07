import copy
from src.circle_laws import *
import cvxpy as cp

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
            for irs in range(elements):
                step_surfaces.append(IRS(previous_size,  num_elements))
            self.network_channels.append(step_surfaces)
            previous_size = elements
        self.covariance = 1j*np.zeros((num_elements, num_elements))

    def get_covariance(self, path=[0]):
        for ind, matrix in enumerate(self.network_channels[path[-1]]):
            if len(path) == len(self.network_channels)-1:
                self.channel_from_path(path)
            if len(path) < len(self.network_channels)-1:
                path.append(ind)
                self.get_covariance(path=copy.deepcopy(path))
        return self.covariance@np.conj(self.covariance.T)


    def channel_from_path(self, path):
        for ind, surface_ind in enumerate(path):
            if ind == 0:
                irs = self.network_channels[ind][surface_ind]
                matrix = irs.channels[path[ind]]
            else:
                irs = self.network_channels[ind][surface_ind]
                matrix = irs.channels[path[ind]]@matrix
        self.covariance += matrix

class IRS:
    def __init__(self, num_previous_surfaces, num_elements):
        self.phases = 1j*-1j*np.ones(num_elements)  #Strange but seems to be the way to initialize to real here
        self.channels = []
        for i in range(num_previous_surfaces):
            self.channels.append(c_rand(num_elements,num_elements))


def water_filling(covariance_matrix, power_constraint, sigma_square=1e-2):
    e_values = np.linalg.eig(covariance_matrix)[0]
    variables = cp.Variable(covariance_matrix.shape[0])
    constraint = [cp.sum(variables) <= power_constraint]
    utility = cp.sum(cp.log(1+variables*e_values/sigma_square))
    prob = cp.Problem(cp.Maximize(utility), constraint)
    prob.solve()
    return variables.value

