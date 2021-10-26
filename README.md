# pghipy: Phase Gradient Heap Integration in Python

A Python implementation of STFT/ISTFT transforms and phase recovery using [Phase Gradient Heap Integration](https://arxiv.org/abs/1609.00291). Based on code from [phase-reconstruction](https://github.com/rrlyman/phase-reconstruction) and [tifgan/phase_recovery](https://github.com/tifgan/phase-recovery). The package does not require installation of ltfatpy and works in Windows/MacOS/Linux.

## Installation

```bash
pip install pghipy
```
## Usage

```python
import librosa
import numpy as np
from pghipy import get_default_window, calculate_synthesis_window
from pghipy import stft, pghi, istft

NFFT = 1024
HOP = NFFT//8   # Increasing overlap improves phase recovery

# Create Gaussian windows
winpghi, gamma = get_default_window(NFFT)
winsynth = calculate_synthesis_window(NFFT, HOP, winpghi)

# Magnitude spectrogram
y, sr = librosa.load(librosa.example('trumpet'))
S = np.abs(stft(y,win_length=NFFT,hop_length=HOP,window=winpghi))

# Estimate phase
phase = pghi(S,win_length=NFFT,hop_length=HOP,gamma=gamma)

# Invert
S = S*np.exp(1.0j*phase)
y_inv = istft(S,win_length=NFFT,hop_length=HOP,synthesis_window=winsynth)

```
Note: Uses [numba](https://numba.pydata.org/) JIT compiler to obtain a significant speed-up in phase recovery. Compilation is deferred until the first execution of the function pghi (i.e., lazy compilation).

## Dependencies
* numpy
* scipy
* numba    

## Thanks
Richard Lyman [rrlyman](https://github.com/rrlyman)

Andr&eacute;s Marafioti [andimarafioti](https://github.com/andimarafioti)

## License
[The MIT License (MIT)](https://choosealicense.com/licenses/mit/)
