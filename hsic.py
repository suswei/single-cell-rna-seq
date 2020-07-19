#The original code is refered to https://github.com/romain-lopez/HCV

from scipy.special import gamma
import torch

def bandwidth(d):
    """
    in the case of Gaussian random variables and the use of a RBF kernel,
    this can be used to select the bandwidth according to the median heuristic
    """
    gz = 2 * gamma(0.5 * (d + 1)) / gamma(0.5 * d)
    return 1. / (2. * gz ** 2)


def K(x1, x2, gamma=1.):
    dist_table = torch.unsqueeze(x1, 0) - torch.unsqueeze(x2, 1)
    return torch.t(torch.exp(-gamma * torch.sum(dist_table ** 2, -1)))


def hsic(z, s):
    # use a gaussian RBF for every variable

    d_z = z.shape[1]
    d_s = s.shape[1]

    zz = K(z, z, gamma=bandwidth(d_z))
    ss = K(s, s, gamma=bandwidth(d_s))

    hsic = 0
    hsic += torch.mean(zz * ss)
    hsic += torch.mean(zz) * torch.mean(ss)
    hsic -= 2 * torch.mean(torch.mean(zz, dim=1) * torch.mean(ss, dim=1))
    return torch.sqrt(hsic)

