# don't mind this it's just testing
import numpy as np

L = 10
K = 2

print(np.array([[1,0],[0,1]]).flatten())

def possible_corners(small,big):
    corners = []
    for x in range(big-small+1):
        for y in range(big-small+1):
            corners.append((x,y))
    return corners

print(len(possible_corners(K,L))-(L-K+1)*(L-K+1))