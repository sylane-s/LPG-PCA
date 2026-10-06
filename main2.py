import cv2 as cv
import numpy as np
import torch

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device (cuda or cpu) :", dev)


def lpg_pca(noisy_image, L=41, K=5, sigma=10, T=25, c=8):

    # Useful constants

    H, W = noisy_image.shape
    m = K * K
    l = L // 2
    nw = L - K + 1                 
    N = nw * nw # Number of K-Blocks within L-Window
    max_Ei = T + 2 * sigma**2 # Threshold (see paper)

    pad = cv.copyMakeBorder(noisy, l, l, l, l, cv.BORDER_REFLECT)
    # pad is useful for edge of image 
    # so that each pixel is the center of an appropriate L-Window
    # maybe try other border types ? BORDER_REFLECT does work though
    img = torch.from_numpy(pad).float().to(dev)

    # Computing all K-Blocks
    # patches is the set of values taken by sliding window of size "K" on image "img" along both axes (=all K-Blocks)
    Hp, Wp = H + 2 * l - K + 1, W + 2 * l - K + 1
    patches = img.unfold(0, K, 1).unfold(1, K, 1).reshape(Hp, Wp, m)

    out = torch.empty(H, W, device=dev)
    # empty tensor

    for x in range(H):
        # PCA is computed by each row of the image (=solving W PCA problems at once)
        rows = patches[x:x + nw]                                   # (nw, Wp, m)
        win = rows.unfold(1, nw, 1)                                # (nw, W, m, nw)
        X = win.permute(1, 0, 3, 2).reshape(W, N, m)               # (W, N, m)

        x0 = X[:, N // 2, :]

        d = (X - x0[:, None, :] ** 2).mean(-1)                   # (W, N)

        kth = d.kthvalue(c * m, dim=1, keepdim=True).values        # (W, 1)
        mask = ((d < max_Ei) | (d <= kth)).float().unsqueeze(-1)   # (W, N, 1)
        cnt = mask.sum(1)                                          # (W, 1)

        mu = (X * mask).sum(1) / cnt                               # (W, m)
        Xc = (X - mu[:, None, :]) * mask                           # (W, N, m)
        cov = Xc.transpose(1, 2) @ Xc / cnt[:, :, None]            # (W, m, m)

        lam, Phi = torch.linalg.eigh(cov)                          # (W, m), (W, m, m)

        y0 = torch.einsum("wik,wi->wk", Phi, x0 - mu)              # Phi^T (x0 - mu)
        wk = (lam - sigma**2).clamp(min=0) / lam.clamp(min=1e-8)
        out[x] = mu[:, m // 2] + (Phi[:, m // 2, :] * (wk * y0)).sum(-1)

    return out.clamp(0, 255).cpu().numpy()


if __name__ == "__main__":
    image = cv.imread("assets/sombre.jpg", cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (640, 640), interpolation=cv.INTER_LINEAR)

    sigma = 10.0 # (Gaussian White) Noise Variance
    c_s = 0.35 # Experimental Parameter for "Noise Conservation" (see paper)
    noisy = image.astype(np.float32) + sigma * np.random.randn(*image.shape).astype(np.float32) 
    # Overall is better as float32
    # Having uint8 made pixel values go above 255 with noise so clipping is only done at the end

    import time
    t0 = time.time()
    clean = lpg_pca(noisy, L=41, K=5, sigma=sigma, T=25.0, c=8)
    t1 = time.time()
    print(f"temps :",t1-t0,"s")
    # Takes about 1m10s on my laptop with only CPU-bound computations
    """
    I_tilde = noisy - clean
    E_Itilde2 = np.mean(I_tilde**2)
    sigma_s = c_s * np.sqrt(sigma**2 - E_Itilde2)
    very_clean = lpg_pca(clean, L=41, K=5, sigma=sigma_s, T=25.0, c=8)
    print(f"temps :",time.time()-t1,"s")
    """

    noisy_u8 = np.clip(noisy, 0, 255).astype(np.uint8)
    clean_u8 = clean.astype(np.uint8)
    # very_clean_u8 = very_clean.astype(np.uint8)
    print("PSNR bruitée :", cv.PSNR(image, noisy_u8),"dB")
    print("PSNR débruitée 1 fois:", cv.PSNR(image, clean_u8),"dB")
    # print("PSNR débruitée 2 fois:", cv.PSNR(image, very_clean_u8),"dB")
    cv.imshow("noisy | clean ", np.hstack([noisy_u8, clean_u8]))
    # cv.imshow("noisy | clean | very clean", np.hstack([noisy_u8, clean_u8, very_clean_u8]))
    cv.waitKey(0)
    cv.destroyAllWindows()