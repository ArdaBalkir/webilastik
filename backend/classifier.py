"""
GPU-accelerated Random Forest classifier.

Uses RAPIDS cuML when available (requires CUDA GPU + cuML install).
Falls back transparently to scikit-learn on CPU.

cuML RF is API-compatible with sklearn RF but trains and predicts on GPU,
giving 10–100× speedups for the tile prediction workload.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Backend detection ─────────────────────────────────────────────────────────

_GPU_AVAILABLE = False
try:
    from cuml.ensemble import RandomForestClassifier as _CuRF  # type: ignore
    import cudf  # type: ignore

    _GPU_AVAILABLE = True
    logger.info("cuML RandomForest available — GPU acceleration enabled")
except ImportError:
    _CuRF = None  # type: ignore
    cudf = None  # type: ignore
    logger.info("cuML not found — using scikit-learn CPU backend")

from sklearn.ensemble import RandomForestClassifier as _SkRF


class Classifier:
    """
    Thin wrapper around a Random Forest that prefers GPU (cuML) and falls
    back to CPU (scikit-learn).  Interface mirrors sklearn's RF.

    Usage::

        clf = Classifier()
        clf.fit(X, y)                     # X: (N, F) float32, y: (N,) int32
        proba = clf.predict_proba(X_new)  # returns (N, n_classes) float32
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        force_cpu: bool = False,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.use_gpu = _GPU_AVAILABLE and not force_cpu
        self.n_classes: int = 0
        self.classes_: Optional[np.ndarray] = None
        self._clf: object = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier.

        Parameters
        ----------
        X : (N, F) float32
        y : (N,)  integer labels (0-based or 1-based — kept as-is)
        """
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        if self.use_gpu:
            self._clf = _CuRF(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_streams=4,  # parallel tree building on GPU
            )
            X_gpu = cudf.DataFrame(X)
            y_gpu = cudf.Series(y)
            self._clf.fit(X_gpu, y_gpu)
            logger.debug(
                "GPU RF trained: %d trees, %d samples, %d features",
                self.n_estimators,
                X.shape[0],
                X.shape[1],
            )
        else:
            self._clf = _SkRF(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=2,  # cap per-job parallelism so concurrent trains don't starve each other
            )
            self._clf.fit(X, y)
            logger.debug(
                "CPU RF trained: %d trees, %d samples, %d features",
                self.n_estimators,
                X.shape[0],
                X.shape[1],
            )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability matrix.

        Parameters
        ----------
        X : (N, F) float32

        Returns
        -------
        (N, n_classes) float32
        """
        if self._clf is None:
            raise RuntimeError("Classifier has not been trained yet")
        X = X.astype(np.float32)

        if self.use_gpu:
            X_gpu = cudf.DataFrame(X)
            proba = self._clf.predict_proba(X_gpu)
            return np.asarray(proba, dtype=np.float32)
        else:
            return self._clf.predict_proba(X).astype(np.float32)

    @property
    def is_trained(self) -> bool:
        return self._clf is not None
