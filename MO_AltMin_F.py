import torch
import cmath
import random

from .manopt import sig_manif

def MO_AltMin_F(Fopt, NRF, epsilon):
    """
    Compute the full precoder Ffull, RF precoder FRF and baseband precoder FBB
    based on the optimization method.

    Parameters:
    - Fopt (torch.Tensor): Optimal precoding matrix of size (Nt, Ns)
    - NRF (int): Number of RF chains
    - epsilon (float): Convergence criterion

    Returns:
    - Ffull (torch.Tensor): Full precoder matrix
    - FRF (torch.Tensor): RF precoder matrix
    - FBB (torch.Tensor): Baseband precoder matrix
    """
    
    Nt, Ns = Fopt.shape
    y = [0, 0]  # Initialize with two elements
    
    # Initialize FRF with random phase
    FRF = torch.tensor([[cmath.exp(1j * random.uniform(0, 2 * cmath.pi)) for _ in range(NRF)] for _ in range(Nt)], dtype=torch.complex64)
    
    while True:  # Start with an infinite loop
        # Compute FBB
        FBB = torch.linalg.pinv(FRF) @ Fopt
        
        # Update y[0]
        y[0] = torch.linalg.norm(Fopt - FRF @ FBB, 'fro').item() ** 2
        
        # Update FRF and y[1] using manifold optimization
        FRF, y[1] = sig_manif(Fopt, FRF, FBB)
        
        # Check for convergence
        if abs(y[0] - y[1]) <= epsilon:
            break  # Exit the loop if the condition is met
    
    Ffull = FRF @ FBB
    Ffull = Ffull / torch.sqrt(torch.sum(torch.diag(Ffull.T.conj() @ Ffull)) / NRF)
    
    return Ffull, FRF, FBB