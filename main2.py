import itertools
import time
from pathlib import Path

import cv2 as cv
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device (cuda or cpu) :", dev)

L, K = 11, 3             # Trainin cube size and sample cube size
T = 25.0                 # Threshold (number of samples deemed correct if lower)
c = 8                    # min number of samples = c * K^3
c_s = 0.35               # "Noise Conservation" between two steps (see paper)
batch = 2048             # Number of voxels in a batch
aggregate = True         # True : Avergae all voxel estimations (paper, +0.3 dB)
                         # False : Use only the central K-Block

noise_mode = "gaussian"  # "none" : no added nosie | "gaussian" / "rician" : added g/r noise
sigma_noise = 20.0       # Noise parameter (std)
sigma_known = None       # If None, code estimates noise using 

fg_thresh = 5.0          # Pixels > fg_thresh are subjected to denoising
dilate = 2               # Mask dilatation (to include borders)
show = False             # Show central layer
out_dir = "output"       # Output file (for saving niftis)


def lpg_pca_3d(vol, L, K, sigma, T, c, mask=None, batch=2048, aggregate=True):

    # Useful constants

    assert L % 2 == 1 and K % 2 == 1 and L > K, "L et K doivent être impairs, L > K"
    H, W, D = vol.shape
    m = K ** 3
    l = L // 2
    r = K // 2
    nw = L - K + 1
    N = nw ** 3                    # Number of K-Blocks within L-Window
    k_min = min(c * m, N)          # Min amount of samples
    max_Ei = T + 2 * sigma ** 2    # Threshold (see paper)

    orig = torch.from_numpy(np.ascontiguousarray(vol)).float().to(dev)
    # Added padding such that evey pixel can be the center of a L-Block
    img = F.pad(orig[None, None], (l,) * 6, mode="reflect")[0, 0]     # (H+2l, W+2l, D+2l)

    fg = (torch.ones(H, W, D, dtype=torch.bool, device=dev) if mask is None
          else torch.from_numpy(np.ascontiguousarray(mask)).bool().to(dev))

    if aggregate:
        acc = torch.zeros_like(img)
        wgt = torch.zeros_like(img)
    else:
        out = orig.clone()

    offsets = list(itertools.product(range(K), repeat=3))   # même ordre que le flatten (K,K,K)
    cy = max(1, batch // D)                                  # nb de lignes y par lot

    for x in range(H):
        if not fg[x].any():
            continue
        slab = img[x:x + L]                                  # (L, Wp, Dp)

        for y0 in range(0, W, cy):
            y1 = min(y0 + cy, W)
            sel = fg[x, y0:y1]                               # (cy', D)
            if not sel.any():
                continue

            # L-cubes centrés sur chaque voxel, puis K-cubes dedans (vues, sans copie)
            sub = slab[:, y0:y1 + L - 1, :]
            v = (sub.unfold(0, L, 1).unfold(1, L, 1).unfold(2, L, 1)      # (1,cy',D,L,L,L)
                    .unfold(3, K, 1).unfold(4, K, 1).unfold(5, K, 1))     # (1,cy',D,nw,nw,nw,K,K,K)
            X = v[0][sel].reshape(-1, N, m)                  # (n, N, m) : useful voxels

            x0 = X[:, N // 2, :]  # K-cube to be denoised

            d = ((X - x0[:, None, :]) ** 2).mean(-1)                   # (n, N)
            kth = d.kthvalue(k_min, dim=1, keepdim=True).values        # (n, 1)
            msk = ((d < max_Ei) | (d <= kth)).float().unsqueeze(-1)    # (n, N, 1)
            cnt = msk.sum(1)                                           # (n, 1)

            mu = (X * msk).sum(1) / cnt                                # (n, m)
            Xc = (X - mu[:, None, :]) * msk
            cov = Xc.transpose(1, 2) @ Xc / cnt[:, :, None]            # (n, m, m)
            lam, Phi = torch.linalg.eigh(cov)                          # (n, m), (n, m, m)

            y0_ = torch.einsum("nik,ni->nk", Phi, x0 - mu)             # Phi^T (x0 - mu)
            wk = (lam - sigma ** 2).clamp(min=0) / lam.clamp(min=1e-10)
            est = mu + torch.einsum("nik,nk->ni", Phi, wk * y0_)       # (n, m) : Denoised K-Block !

            if aggregate:
                E = torch.zeros(y1 - y0, D, m, device=dev)
                E[sel] = est
                Wt = sel.float()
                for o, (a, b, cc) in enumerate(offsets):
                    xs, ys, zs = x + l - r + a, y0 + l - r + b, l - r + cc
                    acc[xs, ys:ys + (y1 - y0), zs:zs + D] += E[:, :, o]
                    wgt[xs, ys:ys + (y1 - y0), zs:zs + D] += Wt
            else:
                tmp = out[x, y0:y1]
                tmp[sel] = est[:, m // 2]
                out[x, y0:y1] = tmp

    if aggregate:
        a_ = acc[l:l + H, l:l + W, l:l + D]
        w_ = wgt[l:l + H, l:l + W, l:l + D]
        out = torch.where(w_ > 0, a_ / w_.clamp(min=1), orig)
    return out.cpu().numpy()


def make_mask(vol, thr, dilate):
    """Masque avant-plan : vol > thr, dilaté de `dilate` voxels (pour inclure les bords)."""
    m = torch.from_numpy((vol > thr).astype(np.float32))[None, None]
    if dilate > 0:
        m = F.max_pool3d(m, 2 * dilate + 1, stride=1, padding=dilate)
    return m[0, 0].numpy() > 0


def estimate_sigma(v, mask=None):
    """Estimation robuste de sigma (type Immerkaer 3D) : Laplacien discret -> variance du bruit x 42, MAD."""
    ctr = v[1:-1, 1:-1, 1:-1]
    lap = 6 * ctr - (v[:-2, 1:-1, 1:-1] + v[2:, 1:-1, 1:-1] +
                     v[1:-1, :-2, 1:-1] + v[1:-1, 2:, 1:-1] +
                     v[1:-1, 1:-1, :-2] + v[1:-1, 1:-1, 2:])
    if mask is not None:
        lap = lap[mask[1:-1, 1:-1, 1:-1]]
    lap = lap.ravel()
    return 1.4826 * np.median(np.abs(lap - np.median(lap))) / np.sqrt(42.0)


def psnr(ref, est, mask=None, peak=255.0):
    d = (ref.astype(np.float64) - est.astype(np.float64)) ** 2
    mse = d[mask].mean() if mask is not None else d.mean()
    return 10 * np.log10(peak ** 2 / max(mse, 1e-12))


def denoise_nifti(path):
    # Denoise nifti files
    nii = nib.load(path)
    data = np.squeeze(nii.get_fdata(dtype=np.float32))
    assert data.ndim == 3, f"{path} : volume 3D attendu, reçu {data.shape}"
    name = Path(path).name.replace(".nii.gz", "").replace(".nii", "")
    print(f"\n=== {name} {data.shape} ===")

    # Normalizing
    hi = float(np.percentile(data, 99.9))
    scale = 255.0 / hi
    image = data * scale          # "image" = reference (either raw data or itself based on noise)

    mask = make_mask(image, fg_thresh, dilate) if fg_thresh is not None else None
    if mask is not None:
        print(f"masque : {mask.mean() * 100:.1f} % des voxels")

    rng = np.random.default_rng(0)
    if noise_mode == "gaussian":
        noisy = image + sigma_noise * rng.standard_normal(image.shape).astype(np.float32)
        sigma = sigma_noise
    elif noise_mode == "rician":
        n1 = rng.standard_normal(image.shape).astype(np.float32)
        n2 = rng.standard_normal(image.shape).astype(np.float32)
        noisy = np.sqrt((image + sigma_noise * n1) ** 2 + (sigma_noise * n2) ** 2)
        sigma = sigma_noise
    else:
        noisy = image
        sigma = sigma_known if sigma_known is not None else estimate_sigma(noisy, mask)
        print(f"sigma ({'connu' if sigma_known is not None else 'estimé'}) : {sigma:.3f}")
    noisy = noisy.astype(np.float32)

    t0 = time.time()

    # First step
    clean = lpg_pca_3d(noisy, L=L, K=K, sigma=sigma, T=T, c=c, mask=mask, batch=batch, aggregate=aggregate)
    print(f"Runtime (Step 1) :", time.time() - t0, "s")

    # Noise level update
    resid = noisy - clean
    mse_res = np.mean(resid[mask] ** 2) if mask is not None else np.mean(resid ** 2)
    sigma_s = c_s * np.sqrt(max(0, sigma ** 2 - mse_res))
    print(f"sigma_s :", sigma_s)

    # Second step
    very_clean = lpg_pca_3d(clean, L=L, K=K, sigma=sigma_s, T=T, c=c, mask=mask, batch=batch, aggregate=aggregate)
    print(f"Runtime (Step 1+2) :", time.time() - t0, "s")

    # Clipping (noise can make the pixel values go above 255)
    noisy_c, clean_c, very_clean_c = (np.clip(a, 0, 255) for a in (noisy, clean, very_clean))

    # Metrics
    if noise_mode != "none":
        PSNR_1, PSNR_2, PSNR_3 = (psnr(image, a, mask) for a in (noisy_c, clean_c, very_clean_c))
        print("PSNR noisy :", PSNR_1, "dB")
        print("PSNR denoised 1 time (=stage 1 of LPG PCA):", PSNR_2, "dB")
        print("PSNR denoised 2 times (=stage 2 of LPG PCA)", PSNR_3, "dB")
        print("Gain from step 1 -> 2 :", PSNR_2 - PSNR_1, "dB")
        print("Gain from step 2 -> 3 :", PSNR_3 - PSNR_2, "dB")

    # Saving
    Path(out_dir).mkdir(exist_ok=True)
    hdr = nii.header.copy()
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image((very_clean_c / scale).astype(np.float32), nii.affine, hdr),
             f"{out_dir}/{name}_lpgpca.nii.gz")

    z = image.shape[2] // 2
    tiles = [noisy_c[:, :, z], clean_c[:, :, z], very_clean_c[:, :, z]]
    preview = np.hstack([t.astype(np.uint8) for t in tiles])
    cv.imwrite(f"{out_dir}/{name}_preview.png", preview)
    if show:
        cv.imshow("noisy | clean | very clean", preview)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return noisy_c, clean_c, very_clean_c


if __name__ == "__main__":
    denoise_nifti("niftis/0aeoorbt.nii.gz")
