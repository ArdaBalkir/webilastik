"""
Feature extraction for pixel classification.

Primary backend: fastfilters (fast Gaussian-family convolutions, C extension).
Fallback: scipy.ndimage (pure Python/C, slightly slower but always available).

All public functions work on (H, W, C) uint8 or float32 numpy arrays and
return (H*W, num_features) float32.
"""

from __future__ import annotations

from typing import List

import numpy as np

try:
    import fastfilters as _ff  # type: ignore

    _FASTFILTERS = True
except ImportError:
    _ff = None
    _FASTFILTERS = False

import scipy.ndimage as _ndi  # always available


FILTER_NAMES = [
    "gaussianSmoothing",
    "laplacianOfGaussian",
    "gaussianGradientMagnitude",
    "differenceOfGaussians",
    "structureTensorEigenvalues",
    "hessianOfGaussianEigenvalues",
]


def extract_features(
    tile: np.ndarray,
    filters: List[str],
    scales: List[float],
) -> np.ndarray:
    """
    Parameters
    ----------
    tile:    (H, W) or (H, W, C) uint8 / float32 image
    filters: list of filter names (subset of FILTER_NAMES)
    scales:  list of sigma values, e.g. [0.3, 0.7, 1.0, 1.6, 3.5, 5.0]

    Returns
    -------
    (H*W, num_features) float32 array
    """
    if tile.ndim == 2:
        tile = tile[:, :, np.newaxis]
    h, w, c = tile.shape
    img = tile.astype(np.float32)

    channels: List[np.ndarray] = []
    for ch in range(c):
        plane = img[:, :, ch]  # (H, W)
        for scale in scales:
            for filt in filters:
                result = _apply(plane, filt, scale)
                if result.ndim == 2:
                    channels.append(result)
                else:
                    # multi-channel output (e.g. eigenvalues → 2 maps)
                    for i in range(result.shape[-1]):
                        channels.append(result[:, :, i])

    if not channels:
        raise ValueError("No feature channels produced — check filters list")

    stack = np.stack(channels, axis=-1)  # (H, W, F)
    return stack.reshape(h * w, -1).astype(np.float32)


def _apply(plane: np.ndarray, filter_name: str, scale: float) -> np.ndarray:
    """Dispatch to fastfilters or scipy."""
    if _FASTFILTERS:
        return _ff_apply(plane, filter_name, scale)
    return _scipy_apply(plane, filter_name, scale)


# ── fastfilters backend ───────────────────────────────────────────────────────


def _ff_apply(plane: np.ndarray, name: str, scale: float) -> np.ndarray:
    ff = _ff
    if name == "gaussianSmoothing":
        return ff.gaussianSmoothing(plane, scale)
    if name == "laplacianOfGaussian":
        return ff.laplacianOfGaussian(plane, scale)
    if name == "gaussianGradientMagnitude":
        return ff.gaussianGradientMagnitude(plane, scale)
    if name == "differenceOfGaussians":
        s1 = ff.gaussianSmoothing(plane, scale)
        s2 = ff.gaussianSmoothing(plane, scale * 0.66)
        return (s1 - s2).astype(np.float32)
    if name == "structureTensorEigenvalues":
        return ff.structureTensorEigenvalues(plane, scale, scale * 0.5)
    if name == "hessianOfGaussianEigenvalues":
        return ff.hessianOfGaussianEigenvalues(plane, scale)
    raise ValueError(f"Unknown filter: {name!r}")


# ── scipy fallback ────────────────────────────────────────────────────────────


def _scipy_apply(plane: np.ndarray, name: str, scale: float) -> np.ndarray:
    ndi = _ndi
    if name == "gaussianSmoothing":
        return ndi.gaussian_filter(plane, scale).astype(np.float32)

    if name == "laplacianOfGaussian":
        sm = ndi.gaussian_filter(plane, scale)
        return ndi.laplace(sm).astype(np.float32)

    if name == "gaussianGradientMagnitude":
        gy = ndi.gaussian_filter(plane, scale, order=[1, 0])
        gx = ndi.gaussian_filter(plane, scale, order=[0, 1])
        return np.sqrt(gx**2 + gy**2).astype(np.float32)

    if name == "differenceOfGaussians":
        s1 = ndi.gaussian_filter(plane, scale)
        s2 = ndi.gaussian_filter(plane, scale * 0.66)
        return (s1 - s2).astype(np.float32)

    if name == "structureTensorEigenvalues":
        gy = ndi.gaussian_filter(plane, scale, order=[1, 0])
        gx = ndi.gaussian_filter(plane, scale, order=[0, 1])
        inner = scale * 0.5
        Ixx = ndi.gaussian_filter(gx * gx, inner)
        Ixy = ndi.gaussian_filter(gx * gy, inner)
        Iyy = ndi.gaussian_filter(gy * gy, inner)
        disc = np.sqrt(np.maximum(0.0, (Ixx - Iyy) ** 2 / 4 + Ixy**2))
        mean = (Ixx + Iyy) / 2
        lam1 = (mean + disc).astype(np.float32)
        lam2 = (mean - disc).astype(np.float32)
        return np.stack([lam1, lam2], axis=-1)

    if name == "hessianOfGaussianEigenvalues":
        Hxx = ndi.gaussian_filter(plane, scale, order=[2, 0])
        Hxy = ndi.gaussian_filter(plane, scale, order=[1, 1])
        Hyy = ndi.gaussian_filter(plane, scale, order=[0, 2])
        disc = np.sqrt(np.maximum(0.0, (Hxx - Hyy) ** 2 / 4 + Hxy**2))
        mean = (Hxx + Hyy) / 2
        lam1 = (mean + disc).astype(np.float32)
        lam2 = (mean - disc).astype(np.float32)
        return np.stack([lam1, lam2], axis=-1)

    raise ValueError(f"Unknown filter: {name!r}")
