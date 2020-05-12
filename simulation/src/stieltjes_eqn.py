# A collection of equations for different analytic results
import numpy as np

#   The equations corresponding to the LOS + Scattering Stieltjes Transform
def arg_function(s_val, g_k, sigma=1, rho=1):
    x_s = np.sqrt(s_val)
    return x_s - (pow(sigma, 2)*rho*x_s*g_k)/(rho-pow(sigma, 2)*s_val*np.power(g_k, 2))


def final_stieltjes_deformed(s_values, g_k, func_params):
    sigma = func_params[0]
    rho = func_params[1]
    phi = func_params[2]
    x_arg = arg_function(s_values, g_k, sigma=sigma, rho=rho)
    # Correct
    G_BB = 1/2 + (1-phi)/(2*np.power(x_arg, 2)) -\
           np.sqrt(np.power((1-phi), 2)/(4*np.power(x_arg, 4)) - (1+phi)/(2*np.power(x_arg, 2)) + 1/4)
    # Correct
    # G_BB = np.sqrt(np.power((1-phi), 2)/(4*np.power(x_arg, 4)) - (1+phi)/(2*np.power(x_arg, 2))
    #                + 1/4) - 1/2 - (1-phi)/(2*np.power(x_arg, 2))
    return (1/np.sqrt(s_values))*x_arg*G_BB



