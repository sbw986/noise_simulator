#Noise Simulator
import numpy as np
import matplotlib.pyplot as plt

def noise(num_samples = 10000, alpha = None, noise_type = 'pink', to_plot = 'False'):
    """
    :type num_samples: int
    :type alpha: float
    :type noise_type: str
    :rtype: List[float]
    """
    
    if alpha is None:
        if noise_type == 'white':
            alpha = 0
        elif noise_type == 'pink':
            alpha = 1
        elif noise_type == 'brown':
            alpha = 2

    samps = np.random.normal(0, 1, num_samples)
    samps_fft = np.fft.fft(samps)

    if len(samps_fft) % 2 == 0:
        den1 = np.arange(1, len(samps_fft)//2 + 2)
        den2 = np.arange(len(samps_fft)//2, 1, -1)
    else:
        den1 = np.arange(1, len(samps_fft)//2 + 2)
        den2 = np.arange(len(samps_fft)//2 + 1, 1, -1)
    dens = np.concatenate([den1, den2])

    new_samps = samps_fft / (np.sqrt(dens) ** alpha)

    new_samps_ifft = np.fft.ifft(new_samps)

    if to_plot == True:
        plt.plot(new_samps_ifft)
        plt.show()

    return new_samps_ifft

if __name__ == '__main__':
    data = noise(10000, alpha = 1, to_plot = True)
    #plt.plot(data)
    #plt.show()
