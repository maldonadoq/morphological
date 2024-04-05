import cv2 as cv
import numpy as np

kernel3 = cv.getGaussianKernel(g, sigma=0)


def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


# gaussian = conv2d(Z, kernel3)       # Horizontal
# gaussian = conv2d(Z, kernel3.T)     # Vertical
