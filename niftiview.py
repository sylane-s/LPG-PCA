import glob, random
from pathlib import Path
import cv2 as cv
import numpy as np
from main2 import denoise_nifti

files = glob.glob("niftis/*.nii*")
for f in random.sample(files, min(2, len(files))):
    noisy, clean, very_clean = denoise_nifti(f)
    z = noisy.shape[2] // 2
    cv.imshow(Path(f).name, np.hstack([a[:, :, z] for a in (noisy, clean, very_clean)]).astype(np.uint8))
    cv.waitKey(1)

cv.waitKey(0)
cv.destroyAllWindows()