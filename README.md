# Reproducing "CNN-Based Precoder and Combiner Design in mmWave MIMO Systems", but written in PyTorch!!
The first unofficial PyTorch implementation of a paper with the titled "CNN-Based Precoder and Combiner Design in mmWave MIMO Systems". In this repository, I write pytorch codes on how to generate MIMO dataset as introduced by Elbin (2019). This is very challenging since Matlab and PyTorch has many differences and utilities and I never learn Matlab before (so I need to learn that first). 

**Paper Reference**: A. M. Elbir, "CNN-Based Precoder and Combiner Design in mmWave MIMO Systems," in IEEE Communications Letters, vol. 23, no. 7, pp. 1240-1243, July 2019, [doi: 10.1109/LCOMM.2019.2915977](https://ieeexplore.ieee.org/document/8710287).

## Installation

Clone the repository and install the required packages.

```bash
git clone https://github.com/yourusername/CNNmmWave-PyTorch.git
cd CNNmmWave-PyTorch
pip install -r requirements.txt
```

## Usage

### Generating MIMO Data

You can generate MIMO data using the `generateMIMO.py` script. The script accepts various command-line arguments to customize the dataset. Here's an example:

```bash
python generateMIMO.py --Nt_param 16 --Nray_param 4 --Ncl_param 4 --angspread 5 --selectOutputAsPhases 1 --snr_param 0 --Ns_param 2 --Nreal 100 --Nchannels 10 --noiseLevelHdB 15 20 25
```

### Arguments

- `--Nt_param`: Number of Tx antennas. Default is 16.
- `--Nray_param`: Number of rays for each user. Default is 4.
- `--Ncl_param`: Number of clusters. Default is 4.
- `--angspread`: Angle spread of users. Default is 5.
- `--selectOutputAsPhases`: Output phases or not. Default is 1.
- `--snr_param`: SNR in dB. Default is 0.
- `--Ns_param`: Number of data streams. Default is [2].
- `--Nreal`: Number of realizations. Default is 100.
- `--Nchannels`: Number of channel matrices for the input data. Default is 10.
- `--noiseLevelHdB`: Noise level in dB. Default is [15, 20, 25].

## File Structure

- `MO_AltMin_F.py`: Computes the full precoder Ffull, RF precoder FRF, and baseband precoder FBB.
- `MO_AltMin_W.py`: Computes the full combiner Wfull, RF combiner WRF, and baseband combiner WBB.
- `awgn_noise.py`: Adds AWGN noise to the input signal.
- `complex_circle.py`: Provides utilities to mimic a Riemannian manifold for unit-modulus complex numbers.
- `generateMIMO.py`: Generates MIMO data based on Elbir (2019).
- `generate_H_Ar_At.py`: Generates the channel matrices and angles.
- `manopt.py`: Performs manifold optimization.
- `utils.py`: Contains all helper functions.

## Contributing

Feel free to open issues or PRs if you find any problems or have suggestions for improvements.

## License

This project is under MIT License - see the [LICENSE.md](LICENSE) file for details.