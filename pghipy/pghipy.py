import numpy as np
import heapq
import scipy.signal as signal
from numba import jit

def get_default_window(n_fft):
    lambda_ = (-n_fft**2/(8*np.log(0.01)))**.5
    lambdasqr = lambda_**2
    gamma = 2*np.pi*lambdasqr
    window = np.array(signal.windows.gaussian(2*n_fft+1, lambda_*2, sym=False))[1:2*n_fft+1:2]

    return window, gamma

def calculate_synthesis_window(win_length=2048, hop_length=512, window=None):
    if window is None:
        window, _ = get_default_window(win_length)
    
    gsynth = np.zeros_like(window)
    for l in range(int(win_length)):
        denom = 0                
        for n in range(-win_length//hop_length, win_length//hop_length+1):
            dl = l-n*hop_length
            if dl >=0 and dl < win_length:
                denom += window[dl]**2
        gsynth[l] = window[l]/denom

    return gsynth

def stft(x,win_length=2048,hop_length=512,window=None):
    x = np.append(np.zeros(win_length//2),x)
    x = np.append(x,np.zeros(win_length))
    L = x.shape[0] - win_length
    if window is None:
        window, _ = get_default_window(win_length)
        
    return np.stack([np.fft.rfft(window*x[ix:ix + win_length]) for ix in range(0, L, hop_length)])

def istft(X, win_length=2048,hop_length=512,window=None,synthesis_window=None,length=None):
    N = X.shape[0]
    vr=np.fft.irfft(X)
    sig = np.zeros((N*hop_length+win_length))

    if window is None:
        window, _ = get_default_window(win_length)
    if synthesis_window is None:
        synthesis_window = calculate_synthesis_window(win_length, hop_length, win_length, window)

    for n in range(N):
        vs = vr[n]*synthesis_window
        sig[n*hop_length: n*hop_length+win_length] += vs
    
    sig = sig[win_length//2:-win_length]
    
    if length is not None: 
        if len(sig) >= length:
            sig = sig[0:length]
        else:
            sig = np.pad(sig,(0,length-len(sig)), 'constant')
    
    return sig

@jit(nopython=True)
def pghi(x,win_length=2048, hop_length=512, gamma=None, tol=1e-6):
    if gamma is None: gamma = 2*np.pi*((-win_length**2/(8*np.log(0.01)))**.5)**2
   
    spectrogram = x.copy()
    abstol = np.array([1e-10], dtype=spectrogram.dtype)[0]  # if abstol is not the same type as spectrogram then casting occurs
    phase = np.zeros_like(spectrogram)
    max_val = np.amax(spectrogram)  # Find maximum value to start integration
    max_x, max_y = np.where(spectrogram == max_val)
    max_pos = max_x[0], max_y[0]

    if max_val <= abstol:  # Avoid integrating the phase for the spectrogram of a silent signal
        return phase

    M2 = spectrogram.shape[0]
    N = spectrogram.shape[1]
       
    fmul = gamma/(hop_length * win_length)

    y = np.empty((spectrogram.shape[0]+2,spectrogram.shape[1]+2),dtype=spectrogram.dtype)
    y[1:-1,1:-1] = np.log(spectrogram + 1e-50)
    y[0,:] = y[1,:]
    y[-1,:] = y[-2,:]
    y[:,0] = y[:,1]
    y[:,-1] = y[:,-2]
    dxdw = (y[1:-1,2:]-y[1:-1,:-2])/2
    dxdt = (y[2:,1:-1]-y[:-2,1:-1])/2
    
    fgradw = dxdw/fmul + (2*np.pi*hop_length/win_length)*np.arange(int(win_length/2)+1)
    tgradw = -fmul*dxdt + np.pi
    
    magnitude_heap = [(-max_val, max_pos)] # Numba requires heap to be initialized with content
    spectrogram[max_pos] = abstol

    small_x, small_y = np.where(spectrogram < max_val*tol)
    for x, y in zip(small_x, small_y):
        spectrogram[x, y] = abstol # Do not integrate over silence

    while max_val > abstol:
        while len(magnitude_heap) > 0: # Integrate around maximum value until reaching silence
            max_val, max_pos = heapq.heappop(magnitude_heap)

            col = max_pos[0]
            row = max_pos[1]

            #Spread to 4 direct neighbors
            N_pos = col+1, row
            S_pos = col-1, row
            E_pos = col, row+1
            W_pos = col, row-1

            if max_pos[0] < M2-1 and spectrogram[N_pos] > abstol:
                phase[N_pos] = phase[max_pos] + (fgradw[max_pos] + fgradw[N_pos])/2
                heapq.heappush(magnitude_heap, (-spectrogram[N_pos], N_pos))
                spectrogram[N_pos] = abstol

            if max_pos[0] > 0 and spectrogram[S_pos] > abstol:
                phase[S_pos] = phase[max_pos] - (fgradw[max_pos] + fgradw[S_pos])/2
                heapq.heappush(magnitude_heap, (-spectrogram[S_pos], S_pos))
                spectrogram[S_pos] = abstol

            if max_pos[1] < N-1 and spectrogram[E_pos] > abstol:
                phase[E_pos] = phase[max_pos] + (tgradw[max_pos] + tgradw[E_pos])/2
                heapq.heappush(magnitude_heap, (-spectrogram[E_pos], E_pos))
                spectrogram[E_pos] = abstol

            if max_pos[1] > 0 and spectrogram[W_pos] > abstol:
                phase[W_pos] = phase[max_pos] - (tgradw[max_pos] + tgradw[W_pos])/2
                heapq.heappush(magnitude_heap, (-spectrogram[W_pos], W_pos))
                spectrogram[W_pos] = abstol

        max_val = np.amax(spectrogram) # Find new maximum value to start integration
        max_x, max_y = np.where(spectrogram==max_val)
        max_pos = max_x[0], max_y[0]
        heapq.heappush(magnitude_heap, (-max_val, max_pos))
        spectrogram[max_pos] = abstol
        
    return phase

def griffin_lim(X, win_length=2048,hop_length=512,window=None,synthesis_window=None, n_iters=100):
    if window is None:
        window, _ = get_default_window(win_length)
    if synthesis_window is None:
        synthesis_window = calculate_synthesis_window(win_length, hop_length, window)
    
    mag_X = np.abs(X)
    for i in range(n_iters):
        x = istft(X, win_length, hop_length, window, synthesis_window)
        X = stft(x,win_length,hop_length,window)
        X = mag_X*np.exp(1.0j*np.angle(X))
        
    return x
