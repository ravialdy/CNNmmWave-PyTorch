import torch

def add_awgn_noise(s, SNRdB, L=1):
    """
    Add AWGN noise to the input signal.
    
    Parameters:
    - s: torch.Tensor
        The input signal to which noise will be added.
    - SNRdB: float
        The desired Signal-to-Noise Ratio in dB.
    - L: int, optional
        The oversampling ratio. Default is 1.
        
    Returns:
    - r: torch.Tensor
        The received signal after adding noise.
    - n: torch.Tensor
        The noise vector that is added.
    - N0: float
        The noise spectral density.
    """
    
    gamma = 10 ** (SNRdB / 10)  # Convert SNR from dB to linear scale
    
    if len(s.shape) == 1:
        P = L * torch.sum(torch.abs(s)**2) / s.shape[0]  # Actual power in the signal
    else:
        P = L * torch.sum(torch.abs(s)**2) / torch.numel(s)  # For multi-dimensional arrays
    
    N0 = P / gamma  # Noise spectral density
    
    if torch.isreal(s):
        n = torch.sqrt(torch.tensor(N0 / 2)) * torch.randn(*s.shape)  # Real noise
    else:
        n = torch.sqrt(torch.tensor(N0 / 2)) * (torch.randn(*s.shape) + 1j * torch.randn(*s.shape))  # Complex noise
    
    r = s + n  # Received signal
    return r, n, N0