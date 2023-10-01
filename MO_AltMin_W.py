import torch
import cmath
import random
import numpy as np
from .manopt import sig_manif

def MO_AltMin_W(Wopt, Fopt, H, NRF, noisevar, epsilon):
    """
    Compute the full combiner Wfull, RF combiner WRF and baseband combiner WBB
    based on the optimization method.

    Parameters:
    - Wopt (torch.Tensor): Optimal combining matrix of size (Nr, Ns)
    - Fopt (torch.Tensor): Optimal precoding matrix
    - H (torch.Tensor): Channel matrix
    - NRF (int): Number of RF chains
    - noisevar (float): Noise variance
    - epsilon (float): Convergence criterion

    Returns:
    - Wfull (torch.Tensor): Full combiner matrix
    - WRF (torch.Tensor): RF combiner matrix
    - WBB (torch.Tensor): Baseband combiner matrix
    """
    
    Nr, Ns = Wopt.shape
    y = [0, 0]  # Initialize with two elements
    
    # Initialize WRF with random phase
    WRF = torch.tensor([[cmath.exp(1j * random.uniform(0, 2 * cmath.pi)) for _ in range(NRF)] for _ in range(Nr)], dtype=torch.complex64)
    
    while True:  # Start with an infinite loop
        # Compute WMMSE
        Wmmse = torch.linalg.solve(Fopt.T.conj() @ H.T.conj() @ H @ Fopt + noisevar * Ns * torch.eye(Ns), Fopt.T.conj() @ H.T.conj()).T
        
        # Compute Ess and Eyy
        Ess = torch.eye(Ns) / Ns
        Eyy = H @ Fopt @ Ess @ Fopt.T.conj() @ H.T.conj() + noisevar * torch.eye(Nr)
        
        # Compute WBB
        WBB = torch.linalg.solve(WRF.T.conj() @ Eyy @ WRF, WRF.T.conj() @ Eyy @ Wmmse)
        
        # Update y[0]
        y[0] = torch.linalg.norm(Wopt - WRF @ WBB, 'fro').item() ** 2
        
        # Update WRF and y[1] using manifold optimization
        WRF, y[1] = sig_manif(Wopt, WRF, WBB)
        
        # Check for convergence
        if abs(y[0] - y[1]) <= epsilon:
            break  # Exit the loop if the condition is met
    
    # Compute the full combiner and normalize it
    Wfull = WRF @ WBB
    Wfull = Wfull / torch.sqrt(torch.sum(torch.diag(Wfull.T.conj() @ Wfull)) / NRF)
    
    return Wfull, WRF, WBB