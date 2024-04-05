import cv2 as cv
import numpy as np


def getDarkChannel(image, w=15):
    M, N, _ = image.shape
    padded = np.pad(
        image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    darkChannel = np.zeros((M, N))
    for i, j in np.ndindex(darkChannel.shape):
        darkChannel[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkChannel


def getAtmosphereLight(image, darkChannel, p=0.0001):
    # image = image.transpose(1, 2, 0)
    M, N = darkChannel.shape
    flatI = image.reshape(M * N, 3)
    flatDark = darkChannel.ravel()
    # find top M * N * p indexes
    searchidx = (-flatDark).argsort()[:int(M * N * p)]
    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)


def getTransmissionMap(image, atmosphereLight, omega=0.95, w=15):
    M, N, _ = image.shape
    padded = np.pad(
        image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    transmission = np.zeros((M, N))
    for i, j in np.ndindex(transmission.shape):
        pixel = (padded[i:i + w, j:j + w, :] / atmosphereLight).min()
        transmission[i, j] = 1 - (omega*pixel)

    return transmission


def getAtmosphericScatteringModel(I, t, A, tx=0.1):
    temporal = I - A
    t = cv.max(t, tx)
    J = np.empty(I.shape, I.dtype)
    for i in range(3):
        J[:, :, i] = (temporal[:, :, i] / t) + A[i]

    return J
