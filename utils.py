import torch
import torch.nn.functional as F
import cmath
import math
from itertools import combinations
from typing import Union, Literal

def get_element_position(num_elements, spacing=0.5, axis='z'):
    """
    Calculate the element positions for a ULA.
    
    Parameters:
    - num_elements: Number of elements in the array
    - spacing: Distance between each element, default is 0.5
    - axis: Axis along which the elements are aligned ('x', 'y', or 'z')
    
    Returns:
    - positions: 3 x num_elements tensor of element positions

    Note:
    You can extend this function to support more complex array geometries as needed.
    """
    
    if axis not in ['x', 'y', 'z']:
        raise ValueError("Axis should be one of 'x', 'y', or 'z'")
        
    pos = torch.zeros(3, num_elements)
    
    # Calculate positions
    for i in range(num_elements):
        coordinate = -spacing * (num_elements - 1) / 2 + i * spacing
        if axis == 'x':
            pos[0, i] = coordinate
        elif axis == 'y':
            pos[1, i] = coordinate
        elif axis == 'z':
            pos[2, i] = coordinate
            
    return pos

def scatteringchanmtx(txarraypos, rxarraypos, txang, rxang, gains):
    """
    Parameters:
    - txarraypos: Tensor of shape (3, Nt) for the positions of elements in the transmitting array
    - rxarraypos: Tensor of shape (3, Nr) for the positions of elements in the receiving array
    - txang: Tensor of shape (2, Ns) for the azimuth and elevation angles for each path at the transmitter
    - rxang: Tensor of shape (2, Ns) for the azimuth and elevation angles for each path at the receiver
    - gains: Tensor of shape (Ns,) for the complex gain for each scatterer
    
    Returns:
    - chmat: Tensor of shape (Nt, Nr) representing the channel matrix

    Note: 
    This is a simplified function, and several assumptions have been made, such as free-space propagation and no path loss. 
    You may need to adapt this code to more closely match the actual conditions of your case.
    """
    Nt = txarraypos.shape[1]
    Nr = rxarraypos.shape[1]
    Ns = txang.shape[1]
    
    # Convert angles to radians
    txang_rad = torch.deg2rad(txang)
    rxang_rad = torch.deg2rad(rxang)
    
    # Compute steering vectors for transmitter and receiver
    steer_tx = torch.exp(1j * (torch.sin(txang_rad[0]) * torch.cos(txang_rad[1]) * txarraypos[0].reshape(-1, 1) +
                                torch.sin(txang_rad[0]) * torch.sin(txang_rad[1]) * txarraypos[1].reshape(-1, 1) +
                                torch.cos(txang_rad[0]) * txarraypos[2].reshape(-1, 1)))
    
    steer_rx = torch.exp(1j * (torch.sin(rxang_rad[0]) * torch.cos(rxang_rad[1]) * rxarraypos[0].reshape(-1, 1) +
                                torch.sin(rxang_rad[0]) * torch.sin(rxang_rad[1]) * rxarraypos[1].reshape(-1, 1) +
                                torch.cos(rxang_rad[0]) * rxarraypos[2].reshape(-1, 1)))
    
    # Compute channel matrix
    chmat = torch.zeros(Nt, Nr, dtype=torch.complex64)
    for s in range(Ns):
        chmat += gains[s] * torch.ger(steer_tx[:, s], steer_rx[:, s].conj())
        
    return chmat

def steervec(pos, ang, nqbits=0):
    """
    Parameters:
    - pos: Tensor of shape (D, N) where D is the dimensionality (1, 2, or 3) and N is the number of array elements
    - ang: Tensor of shape (2, M) where M is the number of incoming waves. Each column specifies the azimuth and elevation angles.
    - nqbits: Number of phase shifter quantization bits. Default is 0 (no quantization).
    
    Returns:
    - sv: Tensor of shape (N, M) representing the steering vector

    Note: 
    The phase is calculated based on the assumption that the wavelength of the incoming wave is 1. 
    If you have a different wavelength, you'd adjust the phase calculation accordingly.
    """
    N = pos.shape[1]
    M = ang.shape[1]
    
    # Convert angles to radians
    ang_rad = torch.deg2rad(ang)
    
    # Initialize steering vector
    sv = torch.zeros(N, M, dtype=torch.complex64)
    
    # Calculate steering vector
    for m in range(M):
        az, el = ang_rad[:, m]
        for n in range(N):
            phase = torch.cos(el) * (torch.cos(az) * pos[0, n] +
                                      torch.sin(az) * pos[1, n]) + \
                    torch.sin(el) * pos[2, n]
                    
            # Compute complex exponential
            sv[n, m] = cmath.exp(1j * 2 * cmath.pi * phase)
            
            # Quantize if needed
            if nqbits > 0:
                angle = cmath.phase(sv[n, m])
                quantized_angle = round(angle / (2 * cmath.pi / (2 ** nqbits))) * (2 * cmath.pi / (2 ** nqbits))
                sv[n, m] = cmath.rect(1, quantized_angle)
                
    return sv

def findFrfFbb(Hin, Ns, NtRF, At):
    """
    Computes the RF and baseband precoders (Frf and Fbb) for a given channel matrix Hin, 
    number of streams Ns, number of transmit RF chains NtRF, and transmit array response matrix At.
    
    Parameters:
    - Hin (torch.Tensor): Input channel matrix of shape (Nr, Nt)
    - Ns (int): Number of streams
    - NtRF (int): Number of transmit RF chains
    - At (torch.Tensor): Transmit array response matrix
    
    Returns:
    - Frf (torch.Tensor): RF precoder
    - Fbb (torch.Tensor): Baseband precoder
    """
    H = Hin
    _, _, v = torch.linalg.svd(H)
    Fopt = v[:, :Ns]
    Frf = At
    Fbb = torch.linalg.solve(Frf.T @ Frf, Frf.T @ Fopt)
    Fbb = torch.sqrt(torch.tensor([Ns])) * Fbb / torch.linalg.norm(Frf @ Fbb, 'fro')
    return Frf, Fbb

def findWrfWbb(Hin, Ns, NrRF, Ar, noisevar):
    """
    Computes the RF and baseband combiners (Wrf and Wbb) for a given channel matrix Hin, 
    number of streams Ns, number of receive RF chains NrRF, receive array response matrix Ar,
    and noise variance.
    
    Parameters:
    - Hin (torch.Tensor): Input channel matrix of shape (Nr, Nt)
    - Ns (int): Number of streams
    - NrRF (int): Number of receive RF chains
    - Ar (torch.Tensor): Receive array response matrix
    - noisevar (float): Noise variance
    
    Returns:
    - Wrf (torch.Tensor): RF combiner
    - Wbb (torch.Tensor): Baseband combiner
    """
    H = Hin
    _, _, v = torch.linalg.svd(H)
    Fopt = v[:, :Ns]
    Wmmse = torch.linalg.solve(Fopt.T @ (H.T @ H) @ Fopt + noisevar * Ns * torch.eye(Ns), Fopt.T @ H.T).T
    Ess = torch.eye(Ns) / Ns
    Eyy = H @ Fopt @ Ess @ Fopt.T @ H.T + noisevar * torch.eye(H.shape[0])
    Wrf = Ar
    Wbb = torch.linalg.solve(Wrf.T @ Eyy @ Wrf, Wrf.T @ Eyy @ Wmmse)
    Wrf = Wrf.conj()
    Wbb = Wbb.conj()
    return Wrf, Wbb

def helperComputeSpectralEfficiency(H, F, W, Ns, snr):
    """
    Compute the spectral efficiency given the channel matrix H, precoding matrix F, 
    combining matrix W, number of streams Ns, and signal-to-noise ratio snr.

    Parameters:
    - H (torch.Tensor): Channel matrix of shape (Nr, Nt)
    - F (torch.Tensor): Precoding matrix of shape (Nt, Ns)
    - W (torch.Tensor): Combiner matrix of shape (Nr, Ns)
    - Ns (int): Number of streams
    - snr (float): Signal-to-noise ratio

    Returns:
    - R (float): Spectral Efficiency
    """

    # Transpose channel and precoding matrices to match original MATLAB code
    H = H.t()
    F = F.t()

    # Compute the effective channel
    temp = F[0:Ns, :] @ H @ W[:, 0:Ns]

    # Compute the spectral efficiency
    R = torch.log2(torch.det(torch.eye(Ns, dtype=torch.complex64) + snr / Ns * 
                             (torch.linalg.solve(torch.mm(W[:, 0:Ns].conj().t(), W[:, 0:Ns]).real,
                                                 torch.mm(temp.conj().t(), temp).real))))
    return R.item()

def helperComputeSpectralEfficiencyAS(H, Ns, snr):
    """
    Compute spectral efficiency for a given channel matrix H, 
    number of streams Ns, and signal-to-noise ratio snr.
    
    Parameters:
    - H (torch.Tensor): Channel matrix of shape (Nt, Nr)
    - Ns (int): Number of streams
    - snr (float): Signal-to-noise ratio
    
    Returns:
    - R (float): Spectral Efficiency
    """
    F, W = helperOptimalHybridWeights(H, Ns, 1/snr)
    H = H.T
    F = F.T
    
    temp = torch.mm(torch.mm(F[0:Ns, :], H), W[:, 0:Ns])
    R = torch.log2(torch.det(torch.eye(Ns) + snr/Ns * (torch.mm(temp.T, temp).real / torch.mm(W[:, 0:Ns].T, W[:, 0:Ns]).real)))
    return R.item()

def helperOptimalHybridWeights(H, Ns, noisevar):
    """
    Compute optimal hybrid weights.
    
    Parameters:
    - H (torch.Tensor): Channel matrix
    - Ns (int): Number of streams
    - noisevar (float): Noise variance
    
    Returns:
    - Fopt (torch.Tensor): Optimal precoder
    - Wopt (torch.Tensor): Optimal combiner
    """
    _, _, v = torch.svd(H)
    Fopt = v[:, :Ns]
    
    term1 = torch.mm(torch.mm(Fopt.T, H.T), H)
    term2 = torch.mm(Fopt, term1)
    Wopt = torch.inverse(term2 + noisevar * Ns * torch.eye(Ns))
    Wopt = torch.mm(torch.mm(Fopt.T, H.T), Wopt.T).T
    
    Wopt = Wopt.conj()
    return Fopt, Wopt

def calculateNumberOfSubsets(M, K):
    """
    Calculates the number of subsets and the subsets themselves for given M and K.

    Parameters:
    - M (int): Total number of elements
    - K (int): Number of elements in each subset

    Returns:
    - Q (int): Number of subsets
    - subSet (list of lists): The subsets
    """
    # Calculate the number of subsets (combinations)
    Q = math.comb(M, K)
    
    # Generate all possible combinations (subsets)
    subSet = list(combinations(range(1, M + 1), K))
    
    return Q, subSet

def phase_steering(weights: torch.Tensor, angle: float, wavelength: float, array_type: Literal['tx', 'rx']) -> torch.Tensor:
    """
    Apply phase steering to the given antenna array weights.
    
    Parameters:
        weights (torch.Tensor): The weight tensor for the antenna array, with dimensions [N_RF, N_antennas].
            N_RF is the number of RF chains, and N_antennas is the number of antennas.
        
        angle (float): The steering angle in degrees. This angle is used to calculate the phase shifts.
        
        wavelength (float): The wavelength of the signal, used to calculate the wavenumber.
        
        array_type (str): The type of antenna array, either 'tx' for transmitter or 'rx' for receiver.
        
    Returns:
        torch.Tensor: The updated weight tensor after applying phase steering.
    """
    
    k = 2 * torch.pi / wavelength  # Wavenumber
    d = wavelength / 2  # Assuming half wavelength spacing
    phase_shifts = torch.exp(1j * k * d * torch.sin(angle * torch.pi / 180))
    
    updated_weights = weights.clone()
    
    if array_type == 'tx':
        for i in range(weights.shape[0]):
            updated_weights[i, :] *= phase_shifts
    elif array_type == 'rx':
        for i in range(weights.shape[0]):
            updated_weights[i, :] *= phase_shifts
            
    return updated_weights