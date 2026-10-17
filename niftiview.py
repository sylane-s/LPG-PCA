import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import numpy as np

nifti_img = nib.load("niftis/0aeoorbt.nii.gz")
print(nifti_img.shape) # prints the shape of the image data
print(nifti_img.get_data_dtype()) # prints the data type of the image data
print(nifti_img.affine)
print(type(nifti_img.get_fdata()[:, :, 50]))
images = [nifti_img.get_fdata()[:, :, k] for k in range(256)]
max_img = max(images, key=np.std)
plt.imshow(max_img, cmap='gray')
plt.show()