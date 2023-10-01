import torch
import numpy as np
from utils import get_element_position, scatteringchanmtx, steervec

class generate_H_Ar_At:
    """
    generate_H_Ar_At Class
    
    This class is responsible for generating the channel matrices H, Ar, and At,
    as well as the angles rxang and txang.
    
    Parameters:
    - Ncl (int): Number of clusters
    - Nscatter (int): Number of scatterers
    - Nray (int): Number of rays
    - angspread (float): Angular spread
    - lambda_ (float): Wavelength
    - txarray (array-like): Transmit array
    - rxarray (array-like): Receive array
    - opts (dict): Options dictionary
    
    Methods:
    - generate(): Generates the channel matrices and angles.
    
    Returns:
    - H, Ar, At, rxang, txang: Generated channel matrices and angles.
    """
    
    def __init__(self, Ncl, Nscatter, Nray, angspread, lambda_, txarray, rxarray, opts):
        self.Ncl = Ncl
        self.Nscatter = Nscatter
        self.Nray = Nray
        self.angspread = angspread
        self.lambda_ = lambda_
        self.txarray = txarray
        self.rxarray = rxarray
        self.opts = opts

    def generate(self):
        """
        Generate the channel matrices and angles based on the initialized parameters.
        
        This method uses the PyTorch library for tensor operations.
        
        Returns:
        - H, Ar, At, rxang, txang: Generated channel matrices and angles.
        """
        
        # Set random seed for generating tx and rx angles
        if self.opts.get('fixedUsers', 0) == 0:
            torch.manual_seed(int(torch.randint(1e6, (1,)).item()))
        else:
            torch.manual_seed(4096)
        
        # Generate tx and rx cluster angles
        txclang = torch.rand(1, self.Ncl) * 120 - 60
        rxclang = torch.rand(1, self.Ncl) * 120 - 60

        # Reset the random seed for other parts of the code
        torch.manual_seed(4096)

        txang = torch.zeros(2, self.Nscatter)
        rxang = torch.zeros(2, self.Nscatter)
        
        # Compute the rays within each cluster
        for m in range(self.Ncl):
            txang[:, (m) * self.Nray : (m + 1) * self.Nray] = torch.randn(2, self.Nray) * torch.sqrt(self.angspread) + txclang[:, m]
            rxang[:, (m) * self.Nray : (m + 1) * self.Nray] = torch.randn(2, self.Nray) * torch.sqrt(self.angspread) + rxclang[:, m]
        
        # Set random seed for channel gain
        if self.opts.get('fixedChannelGain', 0) == 0:
            torch.manual_seed(int(torch.randint(1e6, (1,)).item()))
        else:
            torch.manual_seed(4096)

        g = (torch.randn(1, self.Nscatter) + 1j * torch.randn(1, self.Nscatter)) / torch.sqrt(self.Nscatter)

        txpos = get_element_position(self.txarray) / self.lambda_
        rxpos = get_element_position(self.rxarray) / self.lambda_

        At = steervec(txpos, txang)
        Ar = steervec(rxpos, rxang)
        H = scatteringchanmtx(txpos, rxpos, txang, rxang, g)

        return H, Ar, At, rxang, txang