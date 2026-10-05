from functions import *
import cv2 as cv
import numpy as np

blocks = Blocks()
transformations = Transformations()

image = cv.imread("assets/sombre.jpg", cv.IMREAD_GRAYSCALE)
downsize = (640, 640)
image = cv.resize(image, downsize, interpolation= cv.INTER_LINEAR)
# Parameters

L = 41 # Training Window Size
K = 5 # Central Block Size
# L-K should be even
sigma = 10 # (Gaussian White) Noise Variance
T = 25 # Threshold

# Useful constants

sh = image.shape
w, h = sh
m = K**2
l = L//2
max_Ei = T + 2*sigma**2

noisy_image =  np.uint8(image.copy() + sigma*np.random.randn(w,h))

"""
Base idea :
create as many L_blocks and do the PCA on each of them
extract pixels and compute I_hat

Issue :
A lot of pixels on the border are left out...
see later
"""
# L_corners is the corners of all possible L_blocks in image
L_corners = []
for x in range(w-L+1):
    for y in range(h-L+1):
        L_corners.append((x,y))

for L_corner in L_corners:
    L_block = blocks.square_block(noisy_image, L_corner, L)
    K_corners = blocks.possible_corners(K,L)
    x_i_v_hat = []
    for K_corner in K_corners:
        x_i_v_hat.append(blocks.square_block(L_block, K_corner, K))
    x_0_v_hat = x_i_v_hat.pop(len(x_i_v_hat)//2)


cv.imshow('test', noisy_image)
cv.waitKey(0)
cv.destroyAllWindows()