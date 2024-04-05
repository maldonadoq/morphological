import cv2 as cv
import numpy as np


def getMinimalChannel(image):
    minimal = np.min(image, axis=2)
    return minimal


def getMorphologicalTransmissionMap(minimalChannel, s1, s2, g):
    kernel1 = np.ones((s1, s1), np.uint8)
    kernel2 = np.ones((s2, s2), np.uint8)

    opening = cv.morphologyEx(minimalChannel, cv.MORPH_OPEN, kernel1)
    gaussian = cv.GaussianBlur(opening, (g, g), 0)
    dilation = cv.morphologyEx(gaussian, cv.MORPH_DILATE, kernel2)
    tHat = 1 - np.minimum(minimalChannel, dilation)

    return tHat
