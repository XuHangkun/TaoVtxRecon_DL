import numpy as np

def xyz2rthetaphi(xyz):
    """
    Args:
        xyz : tensor [N,3]
    Return
        rtp : tensor [N, 3]
    """
    r = np.sqrt(np.sum(xyz * xyz, axis = 1))
    theta = np.arccos(xyz[:,-1]/r) * 180 / np.pi
    rho = np.sqrt(np.sum(xyz[:, :-1] * xyz[:, :-1]))
    phi = np.arccos(xyz[:,0]/rho) * 180 / np.pi
    phi[xyz[:,1] < 0] = 360 - phi[xyz[:,1] < 0]
    return np.concatenate(
        [r.reshape([len(r),1]), theta.reshape([len(theta),1]), phi.reshape([len(phi), 1])],
        axis = 1)
