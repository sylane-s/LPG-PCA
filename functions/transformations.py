import cv2 as cv
import numpy as np

class Transformations:
    def __init__(self):
        pass  

    # Centralizes sample matrix (duh)
    def centralize_sample_matrix(self, sample_matrix):
        for i in range (len(sample_matrix)):
            mu_i = sample_matrix[i].mean()
            sample_matrix[i] = sample_matrix[i] - mu_i
        return sample_matrix

    # Covariance matrix of the centralized sample matrix
    def covariance_sample_matrix(self, centralized_sample_matrix):
        n,m = centralized_sample_matrix.shape
        return (1/n*np.dot(centralized_sample_matrix,np.transpose(centralized_sample_matrix)))

    # Returns cap_lambda = diag(lambda_1,...,labmda_m), cap_phi, s.t. covariance_sample_matrix = cap_phi*cap_lambda*cap_phi^T
    def LambdaPhi(self, block):
        eigvals, eigvecs = np.linalg.eigh(block)
        return (self.matricize(eigvals), eigvecs)

    # Diagonal matrix of covariance_sample_matrix's eigvals
    def matricize(self, eigvals):
        cap_lambda = np.diag(eigvals)
        return cap_lambda