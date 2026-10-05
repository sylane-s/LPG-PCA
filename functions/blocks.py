import cv2 as cv
import numpy as np

class Blocks:
    def __init__(self):
        pass    

    # Flattens matrix (M_n,k(R) -> R^(n*k))
    def vectorize(self, array):
        return array.flatten()

    # List of corners for K blocks in L block
    def possible_corners(self,small,big):
        corners = []
        for x in range(big-small+1):
            for y in range(big-small+1):
                corners.append((x,y))
        return corners

    # Square block of size "size" with top left corner "corner" in image "img"
    # Within L block, 
    # K_blocks = [blocks.square_block(L_block,corners[j],K) for j in range(len(corners))] 
    def square_block(self, img, corner, size):
        return img[corner[0]:corner[0]+size,corner[1]:corner[1]+size]

    """
    # Block of size "size"
    # "corner" is the top left corner
    # May not be square if "corner" is too close to an edge
    # To be improved
    # Reason : blocks shouldn't be defined by top left corner (as it removes a lot of pixels for the bottom and/or right pixels of the img)
    def block(self, img, corner, size):
        return img[max(0,corner[0]):min(img.shape[0],corner[0]+size),max(0,corner[1]):min(img.shape[1],corner[1]+size)]
    """

    # Block of size "size" in the middle of "img"
    def middle_block(self, img, size):
        middle_pixel = self.middle_pixel(img)
        s = size//2
        corner_pixel = (middle_pixel[0]-s,middle_pixel[1]-s)
        return self.square_block(img, corner_pixel, size)
        
    # Middle pixel of an image
    def middle_pixel(self, img):
        return ((img.shape[0]//2,img.shape[1]//2))
    
    # Mean Sqaured Error (MSE) between two blocks
    # To be improved
    # Reason : check typing
    def block_matching(self, vector1, vector2, SNR):
        MSE = 0
        assert(vector1.shape == vector2.shape)
        vector1 = vector1.astype(np.float64) 
        vector2 = vector2.astype(np.float64)
        for i in range (len(vector1)):
            MSE += (vector1[i]-vector2[i])**2
        MSE = MSE/(len(vector1)**2)
        if SNR:
            return 10*np.log10(255*255/MSE)
        return MSE
    
        