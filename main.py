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
c = 8

noisy_image = np.uint8(image.copy() + sigma*np.random.randn(w,h))
I_hat = noisy_image.copy()

"""
Base idea :
create as many L_blocks and do the PCA on each of them
extract pixels and compute I_hat

State :
LPG-PCA 1st stage (out of 2) is implemented (correctly ?)
Second step requires much less work

Issue :
A lot of pixels on the border are left out...
see later
"""
# L_corners is the set of corners of all possible L_blocks in image
L_corners = []
for x in range(w-L+1):
    for y in range(h-L+1):
        L_corners.append((x,y))

# L_corners = L_corners[:len(L_corners)//2] # debug

for L_corner in L_corners:
    print(L_corner)
    # Building the L and K block structures

    L_block = blocks.square_block(noisy_image, L_corner, L)
    K_corners = blocks.possible_corners(K,L)
    x_i_v_hat = []

    for K_corner in K_corners:
        x_i_v_hat.append(blocks.vectorize(blocks.square_block(L_block, K_corner, K)))
    x_0_v_hat = x_i_v_hat.pop(len(x_i_v_hat)//2)

    # At that point, x_0_v_hat should be the central K block vector
    # and x_i_v_hat the matrix of all potential sample vectors for x_v

    X_v = [x_0_v_hat]

    for x_test_v_hat in x_i_v_hat:
        if (np.mean(x_0_v_hat - x_test_v_hat**2) < max_Ei):
            X_v.append(x_test_v_hat)
    # To do : add something so that x_v contains at least the c*m most decent vectors
    
    X_v = np.array(X_v)
    X_v_hat, mu = transformations.centralize_sample_matrix(X_v)

    omega_X_v_hat = transformations.covariance_sample_matrix(X_v_hat)
    cap_Lambda_X_hat, cap_Phi_X_hat = transformations.LambdaPhi(omega_X_v_hat) # Actually, cap_Lamba_X_hat doesn't need to be a matrix
    P_X_hat = np.transpose(cap_Phi_X_hat)
    Y_v_hat = np.dot(P_X_hat,X_v_hat)

    # Notation hell...

    w_Y,h_Y = Y_v_hat.shape
    clean_Y = np.zeros((w_Y,h_Y))

    # LMMSE Technique = Shrinking the noise in the PCA domain
    for k in range (h_Y):
        w_k = cap_Lambda_X_hat[k,k]/(cap_Lambda_X_hat[k,k] + sigma**2) # Unsure about sigma in this one
        clean_Y[k] = Y_v_hat[k]*w_k
    
    # Back to normal = Inverse PCA transformation and adding back means
    clean_X = np.dot(cap_Phi_X_hat,clean_Y)
    for k in range(h_Y):
        clean_X[k] = clean_X[k] + mu[k]

    # Putting the clean pixels back in place !
    I_hat[L_corner[0],L_corner[1]] = clean_X[K//2,K//2]

cv.imshow('test', I_hat)
cv.waitKey(0)
cv.destroyAllWindows()