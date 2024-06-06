import numpy as np

def subhist(image_pdf, minimum, maximum, normalize):
    """
    Compute the subhistogram between [minimum, maximum] of a given histogram image_pdf
    :param image_pdf: numpy.array
    :param minimum: int
    :param maximum: int
    :param normalize: boolean
    :return: numpy.array
    """
    hi = image_pdf[minimum:maximum+1].copy()
    total = hi.sum()
    if normalize and total > 0:
        hi /= total
    return np.pad(hi, (minimum, 255-maximum), 'constant')

def CDF(hist):
    """
    Compute the CDF of the input histogram
    :param hist: numpy.array
    :return: numpy.array
    """
    return np.cumsum(hist)

def BEASF(image, gamma):
    """
    Compute the Bi-Histogram Equalization with Adaptive Sigmoid Functions algorithm (BEASF)
    A python implementation of the original MATLAB code:
    https://mathworks.com/matlabcentral/fileexchange/47517-beasf-image-enhancer-for-gray-scale-images
    The algorithm is introduced by E. F. Arriaga-Garcia et al., in the research paper:
    https://ieeexplore.ieee.org/document/6808563
    :param image: numpy.ndarray
    :param gamma: float [0, 1]
    :return: numpy.ndarray
    """
    m = int(np.mean(image))
    h = np.histogram(image, bins=256, range=(0, 256))[0] / image.size

    h_lower = subhist(h, 0, m, normalize=True)
    h_upper = subhist(h, m, 255, normalize=True)

    cdf_lower = CDF(h_lower)
    cdf_upper = CDF(h_upper)

    half_low = np.searchsorted(cdf_lower[:m+1], 0.5)
    half_up = np.searchsorted(cdf_upper[m:], 0.5) + m

    tones_low = np.linspace(0, m, m+1)
    tones_up = np.linspace(m, 255, 256-m)

    x_low = 5.0 * (tones_low - half_low) / m
    x_up = 5.0 * (tones_up - half_up) / (255 - m)

    s_low = 1 / (1 + np.exp(-gamma * x_low))
    s_up = 1 / (1 + np.exp(-gamma * x_up))

    mapping_vector = np.zeros(256, dtype=np.int32)
    mapping_vector[:m+1] = (m * s_low).astype(np.int32)
    min_low, max_low = mapping_vector[0], mapping_vector[m]
    mapping_vector[:m+1] = ((m / (max_low - min_low)) * (mapping_vector[:m+1] - min_low)).astype(np.int32)

    mapping_vector[m+1:] = (m + (255 - m) * s_up).astype(np.int32)
    min_up, max_up = mapping_vector[m+1], mapping_vector[255]
    mapping_vector[m+1:] = ((255 - m) * (mapping_vector[m+1:] - min_up) / (max_up - min_up) + m).astype(np.int32)

    res = mapping_vector[image]
    return res
