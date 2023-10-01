import torch
import numpy as np
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient
from pymanopt.manifolds import Stiefel

def sig_manif(Fopt, FRF, FBB):
    """
    Perform manifold optimization to update FRF (RF precoder matrix) and 
    compute the cost value y.

    Parameters:
    - Fopt (torch.Tensor): Optimal precoding matrix
    - FRF (torch.Tensor): Initial RF precoding matrix
    - FBB (torch.Tensor): Baseband precoding matrix

    Returns:
    - y (torch.Tensor): Updated RF precoding matrix
    - cost (float): Cost value
    """

    Nt, NRF = FRF.shape
    K = FBB.shape[2]
    Fopt = Fopt.cpu().numpy()
    FRF = FRF.cpu().numpy()
    FBB = FBB.cpu().numpy()

    # Compute various constants used in optimization
    C1, C2, C3, C4 = [], [], [], []
    for k in range(K):
        temp = Fopt[:,:,k]
        A = np.kron(FBB[:,:,k].T, np.eye(Nt))
        C1.append(np.matmul(temp.ravel().conj(), A))
        C2.append(np.matmul(A.T, temp.ravel()))
        C3.append(np.matmul(A.T, A))
        C4.append(np.linalg.norm(temp, 'fro') ** 2)

    B1 = np.sum(C1, axis=0)
    B2 = np.sum(C2, axis=0)
    B3 = np.sum(C3, axis=0)
    B4 = np.sum(C4)

    # Define the cost function for optimization
    def cost_func(X):
        x = X.ravel()
        return -np.matmul(B1, x) - np.matmul(x.conj(), B2) + np.trace(np.matmul(np.matmul(B3, x), x.conj())) + B4

    # Define the manifold and the problem
    manifold = Stiefel(Nt, NRF)
    problem = Problem(manifold=manifold, cost=cost_func)

    # Instantiate a Pymanopt solver
    solver = ConjugateGradient()

    # Let Pymanopt do the rest
    Xopt, optlog = solver.solve(problem, x=FRF)
    
    return torch.tensor(Xopt, dtype=torch.complex64), optlog['final_values']['f(x)']