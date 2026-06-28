# Defines hann_window, hamming_window, blackman_window

import numpy as np

def hann_window(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("Window length must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float32)
    
    k = np.arange(n)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * k / (n-1)))
    
    return window.astype(np.float32)

def hamming_window(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("Window length must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float32)
    
    k = np.arange(n)
    window = 0.54 - 0.46 * np.cos(2.0 * np.pi * k / (n-1))
    return window.astype(np.float32)

def blackman_window(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("Window length must be positive")
    if n == 1:
        return np.ones(1, dtype=np.float32)
    
    k = np.arange(n)
    window = (0.42 
              - 0.5 * np.cos(2.0 * np.pi * k / (n-1)) 
              + 0.08 * np.cos(4.0 * np.pi * k / (n-1))
             )
    return window.astype(np.float32)

'''
Notes:

1. n is the size of the window --> if n = 1, then there is only one frame so there is nothing to taper
                               --> so our window (which must be size 1) can only be [1.0] conceptually
2. k is an index for each frame in the window --> if n = 8, k = [1,2,3,4,5,6,7,8]
3. We return window.astype(float32) because numpy returns float64 but almost all audio libraries use float32
'''