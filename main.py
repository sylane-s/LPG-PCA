import cv2 as cv
import numpy as np
import torch

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device (cuda or cpu) :", dev)


def lpg_pca(noisy_image, L, K, sigma, T, c):

    # Useful constants

    H, W = noisy_image.shape
    m = K * K
    l = L // 2
    nw = L - K + 1                 
    N = nw * nw # Number of K-Blocks within L-Window
    max_Ei = T + 2 * sigma**2 # Threshold (see paper)
    pad = cv.copyMakeBorder(noisy_image, l, l, l, l, cv.BORDER_REFLECT)
    # pad is useful for edge of image 
    # so that each pixel is the center of an appropriate L-Window
    # maybe try other border types ? BORDER_REFLECT does work though
    img = torch.from_numpy(pad).float().to(dev)

    # Computing all K-Blocks
    # patches is the set of values taken by sliding window of size "K" on image "img" along both axes (=all K-Blocks)
    Hp, Wp = H + 2 * l - K + 1, W + 2 * l - K + 1
    patches = img.unfold(0, K, 1).unfold(1, K, 1).reshape(Hp, Wp, m)

    out = torch.empty(H, W, device=dev)

    for x in range(H):
        
        # PCA is computed for each row of the image (=solving W PCA problems at once)
        rows = patches[x:x + nw]                                   # (nw, Wp, m)
        win = rows.unfold(1, nw, 1)                                # (nw, W, m, nw)
        X = win.permute(1, 0, 3, 2).reshape(W, N, m)               # (W, N, m)

        x0 = X[:, N // 2, :] # Middle element of the sample vector (see paper) 
        # a.k.a K-Block centered around the pixel to be retrieved

        d = ((X - x0[:, None, :] )** 2).mean(-1)                   # (W, N)
        # Tensor containing MSEs between X and x0
        # Reminder : Samples are added to cov if MSE < max_Ei i.e. if they are not "too noisy"

        kth = d.kthvalue(c * m, dim=1, keepdim=True).values        # (W, 1)
        # 
        mask = ((d < max_Ei) | (d <= kth)).float().unsqueeze(-1)   # (W, N, 1)
        # mask*X = valid samples for X in regard to x0
        cnt = mask.sum(1)                                          # (W, 1)
        # Number of valid samples

        mu = (X * mask).sum(1) / cnt                               # (W, m)
        # Average of valid samples
        Xc = (X - mu[:, None, :]) * mask                           # (W, N, m)
        # Xc (Xcentered) = samples with mean 0
        cov = Xc.transpose(1, 2) @ Xc / cnt[:, :, None]            # (W, m, m)
        # cov = 1/m X^T X
        lam, Phi = torch.linalg.eigh(cov)                          # (W, m), (W, m, m)

        y0 = torch.einsum("wik,wi->wk", Phi, x0 - mu)              # Phi^T (x0 - mu)
        wk = (lam - sigma**2).clamp(min=0) / lam.clamp(min=1e-10)
        out[x] = mu[:, m // 2] + (Phi[:, m // 2, :] * (wk * y0)).sum(-1)
        # out = mean + P^-1 * (LMMSEd y0)

    return out.clamp(0, 255).cpu().numpy()


if __name__ == "__main__":
    image = cv.imread("assets/lena.tif", cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (640, 640), interpolation=cv.INTER_LINEAR)

    sigma_noise = 20.0 # (Gaussian White) Noise Variance
    c_s = 0.35 # Experimental Parameter for "Noise Conservation" (see paper)
    noisy = image.astype(np.float32) + (sigma_noise * np.random.randn(*image.shape)).astype(np.float32) 
    # Overall is better as float32
    # Having uint8 made pixel values go above 255 with noise so clipping is only done at the end

    import time

    t0 = time.time()

    # First step
    clean = lpg_pca(noisy, L=41, K=5, sigma=sigma_noise, T=25.0, c=8)
    print(f"Runtime (Step 1) :",time.time()-t0,"s")
    
    # Updating noise level
    sigma_s = c_s * np.sqrt(max(0,sigma_noise**2 - np.mean((noisy-clean)**2)))
    print(f"sigma_s :",sigma_s)

    # Second step
    very_clean = lpg_pca(clean, L=41, K=5, sigma=sigma_s, T=25.0, c=8)
    print(f"Runtime (Step 1+2) :",time.time()-t0,"s")

    noisy_u8 = np.clip(noisy, 0, 255).astype(np.uint8)
    clean_u8 = clean.astype(np.uint8)
    very_clean_u8 = very_clean.astype(np.uint8)

    # Metrics
    PSNR_1 = cv.PSNR(image, noisy_u8)
    PSNR_2 = cv.PSNR(image, clean_u8)
    PSNR_3 = cv.PSNR(image, very_clean_u8)
    print("PSNR noisy :", PSNR_1,"dB")
    print("PSNR denoised 1 time (=stage 1 of LPG PCA):", PSNR_2,"dB")
    print("PSNR denoised 2 times (=stage 2 of LPG PCA)", PSNR_3,"dB")
    print("Gain from step 1 -> 2 :", PSNR_2-PSNR_1,"dB")
    print("Gain from step 2 -> 3 :", PSNR_3-PSNR_2,"dB")

    cv.imshow("noisy | clean | very clean", np.hstack([noisy_u8, clean_u8, very_clean_u8]))
    cv.waitKey(0)
    cv.destroyAllWindows()