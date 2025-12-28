"""
Numpy-compatible wrapper for OneEuroFilter.

Provides an adaptive One Euro Filter that automatically handles numpy arrays
by maintaining separate filter instances for each element.
"""

import numpy as np
from OneEuroFilter import OneEuroFilter
from typing import Optional
import time

class OneEuroFilterNumpy:
    """
    Auto-initializing One Euro Filter for numpy arrays.
    
    Automatically determines shape from first input and maintains independent
    OneEuroFilter instances for each element in the array.
    
    Example:
        >>> filter = AdaptiveOneEuroFilterNumpy(freq=30, mincutoff=1.0, beta=0.01)
        >>> noisy_data = np.array([1.0, 2.0, 3.0])
        >>> filtered = filter(noisy_data)
        >>> filtered = filter(noisy_data + np.random.randn(3) * 0.1)
    """
    
    def __init__(
        self,
        freq: float,
        mincutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0
    ) -> None:
        """
        Initialize an adaptive One Euro Filter.
        
        Args:
            freq: An estimate of the frequency in Hz of the signal (> 0)
            mincutoff: Min cutoff frequency in Hz (> 0). Lower values remove more jitter
            beta: Parameter to reduce latency (>= 0)
            dcutoff: Cutoff frequency for derivatives (> 0)
        """
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._filters = None
        self._shape = None
    
    def __call__(
        self,
        x: np.ndarray,
        timestamp: Optional[float] = None
    ) -> np.ndarray:
        """
        Filter a noisy numpy array.
        
        On first call, initializes filters based on input shape.
        
        Args:
            x: Noisy array to filter
            timestamp: Timestamp in seconds (optional)
        
        Returns:
            Filtered array with same shape as input
        """
        # Initialize filters on first call
        if self._filters is None:
            self._shape = x.shape
            size = np.prod(self._shape).astype(int)
            self._filters = [
                OneEuroFilter(
                    freq=self.freq,
                    mincutoff=self.mincutoff,
                    beta=self.beta,
                    dcutoff=self.dcutoff
                )
                for _ in range(size)
            ]
        
        # Validate shape
        if x.shape != self._shape:
            raise ValueError(
                f"Input shape {x.shape} doesn't match initialized shape {self._shape}"
            )
        
        if timestamp is None:
            timestamp = time.perf_counter()
        
        # Flatten, filter, reshape
        x_flat = x.flatten()
        filtered_flat = np.array([
            self._filters[i](float(val), timestamp)
            for i, val in enumerate(x_flat)
        ])
        
        return filtered_flat.reshape(self._shape)
    
    def filter(
        self,
        x: np.ndarray,
        timestamp: Optional[float] = None
    ) -> np.ndarray:
        """Filter a noisy numpy array (alias for __call__)."""
        return self.__call__(x, timestamp)
    
    def reset(self) -> None:
        """Reset all internal filter states."""
        if self._filters is not None:
            for f in self._filters:
                f.reset()
    
    def setFrequency(self, freq: float) -> None:
        """Set the frequency for all filters."""
        self.freq = freq
        if self._filters is not None:
            for f in self._filters:
                f.setFrequency(freq)
    
    def setMinCutoff(self, mincutoff: float) -> None:
        """Set the min cutoff frequency for all filters."""
        self.mincutoff = mincutoff
        if self._filters is not None:
            for f in self._filters:
                f.setMinCutoff(mincutoff)
    
    def setBeta(self, beta: float) -> None:
        """Set the beta parameter for all filters."""
        self.beta = beta
        if self._filters is not None:
            for f in self._filters:
                f.setBeta(beta)
    
    def setDerivateCutoff(self, dcutoff: float) -> None:
        """Set the derivative cutoff frequency for all filters."""
        self.dcutoff = dcutoff
        if self._filters is not None:
            for f in self._filters:
                f.setDerivateCutoff(dcutoff)
    
    def setParameters(
        self,
        freq: float,
        mincutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0
    ) -> None:
        """Set all parameters for all filters."""
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        if self._filters is not None:
            for f in self._filters:
                f.setParameters(freq, mincutoff, beta, dcutoff)