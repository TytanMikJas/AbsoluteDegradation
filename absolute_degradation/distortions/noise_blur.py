import albumentations as A
import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal as mvn


def gm_blur_kernel(mean, cov, size=15):
    # ensure covariance is symmetric positive definite
    cov = (cov + cov.T) / 2
    cov += np.eye(2) * 1e-6  # numerical stability

    # create a grid of (x, y) coordinates
    center = size / 2.0 + 0.5
    x = np.arange(size) - center + 1
    y = np.arange(size) - center + 1
    xv, yv = np.meshgrid(x, y)

    # stack into (N, 2) points
    points = np.stack([xv, yv], axis=-1)  # shape: (size, size, 2)

    # evaluate multivariate normal PDF at all points
    k = mvn.pdf(points, mean=mean, cov=cov)

    # normalize
    return k / np.sum(k)


def anisotropic_Gaussian(ksize=15, theta=0.0, l1=6.0, l2=6.0):
    # rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # diagonal covariance
    D = np.array([[l1**2, 0], [0, l2**2]])

    covariance_matrix = R @ D @ R.T
    return gm_blur_kernel(mean=[0, 0], cov=covariance_matrix, size=ksize)


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    arg = -((x * x + y * y) / (2 * sigma * sigma))
    h = np.exp(arg)
    h[h < np.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def color_jitter(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Applies Albumentations ColorJitter. And simulates discrete pixels.
    Args:
        image: np.ndarray (H, W, C), float32 [0-1]
    Returns:
        np.ndarray (H, W, C), float32 [0-1]
    """
    transform = A.ColorJitter(
        brightness=(0.8, 1.2),
        contrast=(0.9, 1.0),
        saturation=(1.0, 1.0),
        hue=0.0,
        p=0.5,
    )
    jittered = transform(image=image)["image"]

    # simulate discrete pixels
    jittered = jittered * 255
    jittered = np.clip(jittered, 0, 255)
    jittered = jittered.astype(np.float32) / 255.0
    return jittered
