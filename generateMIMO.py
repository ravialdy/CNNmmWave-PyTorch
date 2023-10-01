import torch
import argparse
import numpy as np
from typing import List, Dict, Union, Tuple
from .utils import *
from .generate_H_Ar_At import generate_H_Ar_At
from .MO_AltMin_F import MO_AltMin_F
from .MO_AltMin_W import MO_AltMin_W
from .awgn_noise import add_awgn_noise

class MIMODataGenerator:
    def __init__(self, Nr: int = 16, Nrs: int = 16, NrRF: int = 4, NtRF: int = 4, opts: Dict = None) -> None:
        """
        Initializes the MIMODataGenerator class with given or default parameters.
        
        Parameters:
            Nr (int): Number of receive antennas.
            Nrs (int): Number of selected receive antennas.
            NrRF (int): Number of RF chains at the receiver.
            NtRF (int): Number of RF chains at the transmitter.
            opts (Dict): Additional options for MIMO system setup.
            
        Returns:
            None
        """
        self.Nr = Nr
        self.Nrs = Nrs
        self.NrRF = NrRF
        self.NtRF = NtRF
        self.opts = opts

        self.tx_weights = torch.ones((self.NtRF, self.Nr))
        self.rx_weights = torch.ones((self.NrRF, self.Nr))

        self.steering_angle_tx = 30  
        self.steering_angle_rx = 45  

        self.fc = 28e9  # Frequency
        speed_of_light = 3e8  # Speed of light
        self.lambda_ = speed_of_light / self.fc  # Wavelength
    
    def generateMIMO(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
                                    List[int], List[List[int]], List[Dict[str, Union[torch.Tensor, float, List[float]]]]]:
        """
        Generate MIMO channel data, performing antenna selection and RF chain selection to optimize the spectral efficiency.

        Returns:
            Tuple containing:
                - XAS (torch.Tensor): Channel state information across all antennas.
                - XRF (torch.Tensor): Channel state information across selected antennas.
                - Y (torch.Tensor): Labels for selected antenna subsets.
                - YFRFr (torch.Tensor): Phases or DOA for RF precoder.
                - YWRFr (torch.Tensor): Phases or DOD for RF combiner.
                - bestAntennas (List[int]): Indices of best antennas.
                - subSetA (List[List[int]]): All possible antenna subsets.
                - Z (List[Dict]): Additional info like H, At, Ar, spectral efficiencies, etc.
        """
        Q, subSetA = calculateNumberOfSubsets(self.Nr, self.Nrs)  # Assume this function is defined
        Nray = self.opts['Nray_param']
        Ncl = self.opts['Ncl_param']
        Nscatter = Nray * Ncl
        snr = 10 ** (self.opts['snr_param'] / 10)
        noisevar = 1 / snr
        Nreal = self.opts['Nreal']
        
        NNs = len(self.opts['Ns_param'])
        NNcl = len(self.opts['Ncl_param'])
        Nch = self.opts['Nchannels']
        
        Nb = NNcl * NNs * Nch
        N = Nreal * Nb

        XAS = torch.zeros((self.Nr, self.Nt, 3, N))
        XRF = torch.zeros((self.Nrs, self.Nt, 3, N))
        Y = torch.zeros((1, N))
        Yb = torch.zeros((1, N))

        if self.opts['selectOutputAsPhases'] == 1:
            YFRFr = torch.zeros((N, self.Nt * self.NtRF))
            YWRFr = torch.zeros((N, self.Nrs * self.NrRF))
        else:
            YFRFr = torch.zeros((N, 2 * self.NtRF))
            YWRFr = torch.zeros((N, 2 * self.NrRF))

        Z: List[Dict[str, Union[torch.Tensor, float, List[float]]]] = [{} for _ in range(N)]
        j = 0
        for nch in range(Nch):
            for ncl in range(NNcl):
                Ncl = self.opts['Ncl_param'][ncl]
                Nscatter = Nray * Ncl
                for ns in range(NNs):
                    Ns = self.opts['Ns_param'][ns]
                    for nr in range(Nreal):
                        if len(self.opts['Ns_param']) > 1:
                            nn = ns
                        elif len(self.opts['Ncl_param']) > 1:
                            nn = ncl
                        else:
                            nn = j

                        if nn < round(Nreal / 3):
                            snrH = self.opts['noiseLevelHdB'][0]
                        elif round(Nreal / 3) < nn < round(2 * Nreal / 3):
                            snrH = self.opts['noiseLevelHdB'][1]
                        else:
                            snrH = self.opts['noiseLevelHdB'][2]

                        if nr == 0:  # Zero-based indexing, so MATLAB's nr == 1 becomes nr == 0
                            jFirst = j
                            # Generate H, Ar, At, rxang, txang
                            generate_H_Ar_At(Ncl, Nscatter, Nray, self.opts['angspread'], self.lambda_, self.txarray, self.rxarray, self.opts)
                            H, Ar, At, rxang, txang = generate_H_Ar_At.generate()
                            H, _, _ = add_awgn_noise(H, snrH)
                            Z[nn] = {'H': H, 'At': At, 'Ar': Ar, 'rxang': rxang, 'txang': txang}
                         
                            # Antenna Selection
                            R: torch.Tensor = torch.zeros(Q)
                            for qA in range(Q):
                                subset = subSetA[qA]
                                R[qA] = helperComputeSpectralEfficiencyAS(H[subset, :], Ns, snr)
                            qAb = torch.argmax(R)
                            bestAntennas = subSetA[qAb]

                            # RF Chain Selection
                            FoptMO, WoptMO = helperOptimalHybridWeights(H, Ns, 1/snr)
                            _, FrfMO, _ = MO_AltMin_F(FoptMO, self.NtRF, 1e-3)
                            _, WrfMO, _ = MO_AltMin_W(WoptMO, FoptMO, H, self.NrRF, 1/snr, 1e-3)

                            Atb = torch.cat([At, FrfMO], dim=1)
                            Arb = torch.cat([Ar, WrfMO], dim=1)

                            QF, subSetF = calculateNumberOfSubsets(Atb.size(1), self.NtRF)
                            RhybF = torch.zeros(QF)
                            RhybW = torch.zeros(QF)
                            Frf = torch.zeros((self.Nt, self.NtRF, QF))
                            Fbb = torch.zeros((self.NtRF, Ns, QF))
                            Wrf = torch.zeros((self.Nrs, self.NrRF, QF))
                            Wbb = torch.zeros((self.NrRF, Ns, QF))
                            Z[nn].update({'H': H, 'At': At, 'Ar': Ar, 'rxang': rxang, 'txang': txang, 'Atb': Atb, 'Arb': Arb})

                            # RF precoder design
                            Fopt, Wopt = helperOptimalHybridWeights(H[bestAntennas, :], Ns, 1 / snr)
                            for qF in range(QF):
                                Frf[:, :, qF], Fbb[:, :, qF] = findFrfFbb(H[bestAntennas, :], Ns, self.NtRF, Atb[:, subSetF[qF, :]])
                                RhybF[qF] = helperComputeSpectralEfficiency(H[bestAntennas, :], Frf[:, :, qF] @ Fbb[:, :, qF], Wopt, Ns, snr)
                            
                            qFb = torch.argmax(RhybF)
                            
                            # RF combiner design
                            Fbest = Fopt  # or Fbest = Frf[:, :, qFb] @ Fbb[:, :, qFb] based on your requirements
                            for qW in range(QF):
                                Wrf[:, :, qW], Wbb[:, :, qW] = findWrfWbb(H[bestAntennas, :], Ns, self.NrRF, Arb[bestAntennas, subSetF[qW, :]], 1 / snr)
                                RhybW[qW] = helperComputeSpectralEfficiency(H[bestAntennas, :], Fbest, Wrf[:, :, qW] @ Wbb[:, :, qW], Ns, snr)
                            
                            qWb = torch.argmax(RhybW)

                            # Update tx_weights and rx_weights
                            self.tx_weights = phase_steering(Frf[:, :, qFb], self.steering_angle_tx, self.lambda_, 'tx')
                            self.rx_weights = phase_steering(Wrf[:, :, qWb], self.steering_angle_rx, self.lambda_, 'rx')
                            
                            # Select phases for output
                            if self.opts['selectOutputAsPhases'] == 1:
                                Z[nn].update({
                                    'FrfSelected': Frf[:, :, qFb],
                                    'WrfSelected': Wrf[:, :, qWb],
                                    'FbbSelected': Fbb[:, :, qFb],
                                    'WbbSelected': Wbb[:, :, qWb]
                                })
                            else:
                                DOASelected = txang[:, subSetF[qFb, :]]
                                DODSelected = rxang[:, subSetF[qWb, :]]
                                Z[nn].update({
                                    'rxang': rxang,
                                    'txang': txang,
                                    'DOASelected': DOASelected,
                                    'DODSelected': DODSelected
                                })                        

                        else:  # Other realizations
                            H, _, _ = add_awgn_noise(Z[jFirst]['H'], snrH)
                            Z[nn].update({
                                'H': H,
                                'At': Z[jFirst]['At'],
                                'Ar': Z[jFirst]['Ar'],
                                'Atb': Z[jFirst]['Atb'],
                                'Arb': Z[jFirst]['Arb']
                            })
                            
                            if self.opts['selectOutputAsPhases'] == 1:
                                Z[nn].update({
                                    'FrfSelected': Z[jFirst]['FrfSelected'],
                                    'WrfSelected': Z[jFirst]['WrfSelected'],
                                    'FbbSelected': Z[jFirst]['FbbSelected'],
                                    'WbbSelected': Z[jFirst]['WbbSelected']
                                })
                            else:
                                DOASelected = Z[jFirst]['DOASelected']
                                DODSelected = Z[jFirst]['DODSelected']
                                Z[nn].update({
                                    'rxang': Z[jFirst]['rxang'],
                                    'txang': Z[jFirst]['txang'],
                                    'DOASelected': DOASelected,
                                    'DODSelected': DODSelected
                                })

                        # Output of the network. Classification
                        Y[0, j] = qAb
                        Yb[0, j] = Y[0, jFirst]
                        Z[nn]['Y'] = Yb[0, j]

                        if self.opts['selectOutputAsPhases'] == 1:
                            YFRFr[j, :] = torch.angle(Z[nn]['FrfSelected'].view(-1))
                            YWRFr[j, :] = torch.angle(Z[nn]['WrfSelected'].view(-1))
                        else:
                            YFRFr[j, :] = (np.pi / 180) * DOASelected.view(-1)
                            YWRFr[j, :] = (np.pi / 180) * DODSelected.view(-1)

                        XAS[:, :, 0, j] = torch.abs(H)
                        XAS[:, :, 1, j] = torch.real(H)
                        XAS[:, :, 2, j] = torch.imag(H)

                        XRF[:, :, 0, j] = torch.abs(H[bestAntennas, :])
                        XRF[:, :, 1, j] = torch.real(H[bestAntennas, :])
                        XRF[:, :, 2, j] = torch.imag(H[bestAntennas, :])

                        j += 1

        Y = Yb
        return XAS, XRF, Y, YFRFr, YWRFr, bestAntennas, subSetA, Z
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MIMO dataset.')
    parser.add_argument('--Nt_param', type=int, default=16, help='Number of Tx antennas')
    parser.add_argument('--Nray_param', type=int, default=4, help='Number of rays for each user')
    parser.add_argument('--Ncl_param', type=int, default=4, help='Number of clusters')
    parser.add_argument('--angspread', type=int, default=5, help='Angle spread of users')
    parser.add_argument('--selectOutputAsPhases', type=int, default=1, help='Output phases or not')
    parser.add_argument('--snr_param', type=int, default=0, help='SNR in dB')
    parser.add_argument('--Ns_param', type=int, nargs='+', default=[2], help='Number of data streams')
    parser.add_argument('--Nreal', type=int, default=100, help='Number of realizations')
    parser.add_argument('--Nchannels', type=int, default=10, help='Number of channel matrices for the input data')
    parser.add_argument('--noiseLevelHdB', type=int, nargs='+', default=[15, 20, 25], help='Noise level in dB')
    args = parser.parse_args()

    opts = vars(args) # Convert args to dictionary

    # Initialize the MIMODataGenerator
    mimo_gen = MIMODataGenerator(Nr=16, Nrs=16, NrRF=4, NtRF=4, opts=opts)

    # Generate the MIMO data
    XAS, XRF, Y, YFRFr, YWRFr, bestAntennas, subSetA, Z = mimo_gen.generateMIMO()

    # Save the data
    torch.save({
        'XAS': XAS,
        'XRF': XRF,
        'Y': Y,
        'YFRFr': YFRFr,
        'YWRFr': YWRFr,
        'bestAntennas': bestAntennas,
        'subSetA': subSetA,
        'Z': Z,
        'opts': opts
    }, 'mimo_data.pth')