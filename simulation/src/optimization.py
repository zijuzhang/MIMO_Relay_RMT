"""
Optimization tools for selecting the phases as IRS elements
"""
import numpy as np

def optimize_phases(num_elements, info):
    opt_phases = np.ones(num_elements)
    return opt_phases