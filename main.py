# from functions import *
import cv2 as cv
import numpy as np

# blocks = Blocks()
# transformations = Transformations()

image = cv.imread("assets/sombre.jpg", cv.IMREAD_GRAYSCALE)
downsize = (640, 640)
image = cv.resize(image, downsize, interpolation= cv.INTER_LINEAR)
# Parameters

L = 41 # Training Window Size
K = 5 # Central Block Size
sigma = 10 # (Gaussian White) Noise Variance
T = 25 # Threshold

# Useful constants

sh = image.shape
w, h = sh
m = K**2
l = L//2
max_Ei = T + 2*sigma**2

noisy_image =  np.uint8(image.copy() + sigma*np.random.randn(w,h))

cv.imshow('test', noisy_image)
cv.waitKey(0)
cv.destroyAllWindows()