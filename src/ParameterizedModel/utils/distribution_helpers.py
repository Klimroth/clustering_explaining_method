######################################
## Collection of utility functions  ##
## used for probability functions   ##
######################################

import torch
from torch.distributions import kl_divergence

def kl(pair):
    kl = kl_divergence(
        pair['guide'],
        pair['model']
    )
    return kl.sum()
            
def get_gamma_std(param_1, scale_param):
    if isinstance(scale_param, torch.Tensor):

        if len(scale_param.shape) == 0:
            if len(param_1.shape) == 0:
                param_2 = scale_param
            else:
                param_2 = scale_param.expand(len(param_1))
        else:
            assert scale_param.shape == param_1.shape, \
                "Error in get_param_2; expected param_1 and param_2 to have the same shape, but got shapes {} and {}.".format(
                    param_1.shape,
                    scale_param.shape
                )
            param_2 = scale_param

    else:
        if len(param_1.shape) == 0:
            param_2 = scale_param
        else:
            param_2 = (torch.ones(len(param_1)) * scale_param)

    return param_2

def mean_std_to_gamma_params(mean, std):
    if isinstance(mean, float):
        mean = torch.tensor(mean, dtype=torch.float64).unsqueeze(-1)

    if isinstance(std, float):
        std = torch.tensor(std, dtype=torch.float64).unsqueeze(-1)

    std = get_gamma_std(mean, std)

    return mean_var_to_gamma_params(mean, std ** 2)

def mean_var_to_gamma_params(mean, var):
    alpha = mean ** 2 / var
    beta = mean / var

    return (alpha.to(torch.float64), beta.to(torch.float64))

def mean_std_to_beta_params(mu, sigma):
    alpha = mu**2 * ((1 - mu) / sigma**2 - 1 / mu)
    beta = alpha * (1 / mu - 1)

    return alpha, beta