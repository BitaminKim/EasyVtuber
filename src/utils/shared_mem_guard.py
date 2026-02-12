# windows_only_shm_lock.py
from __future__ import annotations
import ctypes
import ctypes.wintypes as wt
import hashlib
import time
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Optional, Union

# ---- WinAPI for Named Mutex ----
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# HANDLE CreateMutexW(LPSECURITY_ATTRIBUTES, BOOL, LPCWSTR);
_CreateMutexW = kernel32.CreateMutexW
_CreateMutexW.argtypes = [wt.LPVOID, wt.BOOL, wt.LPCWSTR]
_CreateMutexW.restype = wt.HANDLE

# HANDLE OpenMutexW(DWORD, BOOL, LPCWSTR);
_OpenMutexW = kernel32.OpenMutexW
_OpenMutexW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
_OpenMutexW.restype = wt.HANDLE

# DWORD WaitForSingleObject(HANDLE, DWORD);
_WaitForSingleObject = kernel32.WaitForSingleObject
_WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
_WaitForSingleObject.restype = wt.DWORD

# BOOL ReleaseMutex(HANDLE);
_ReleaseMutex = kernel32.ReleaseMutex
_ReleaseMutex.argtypes = [wt.HANDLE]
_ReleaseMutex.restype = wt.BOOL

# BOOL CloseHandle(HANDLE);
_CloseHandle = kernel32.CloseHandle
_CloseHandle.argtypes = [wt.HANDLE]
_CloseHandle.restype = wt.BOOL

# Constants
INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
WAIT_ABANDONED = 0x00000080
WAIT_FAILED = 0xFFFFFFFF
MUTEX_ALL_ACCESS = 0x1F0001
ERROR_ALREADY_EXISTS = 183


class SharedMemoryGuard:
    """
    Two-process exclusive access wrapper using Windows Named Mutex:
    - payload SharedMemory is provided externally
    - Uses Windows Named Mutex for cross-process synchronization
    - More direct and simpler than manual InterlockedCompareExchange implementation
    """

    def __init__(
        self,
        payload: shared_memory.SharedMemory,
        ctrl_name: str
    ):
        # 1) attach payload (external)
        self.payload_shm = payload

        # 2) create/open named mutex for synchronization
        self.mutex_name = f"Global\\{ctrl_name}_mutex"
        self._mutex_handle = self._create_or_open_mutex(self.mutex_name)
        if self._mutex_handle is None or self._mutex_handle == 0:
            error = ctypes.get_last_error()
            raise OSError(f"Failed to create/open mutex {self.mutex_name}, error: {error}")

    @staticmethod
    def _create_or_open_mutex(name: str) -> wt.HANDLE:
        """Create or open a named mutex."""
        # Try to create the mutex
        handle = _CreateMutexW(None, False, name)
        if handle is None or handle == 0:
            # If creation failed, try to open existing mutex
            handle = _OpenMutexW(MUTEX_ALL_ACCESS, False, name)
        return handle

    # ---- payload view: zero-copy for numpy ----
    def payload_view(self) -> memoryview:
        return self.payload_shm.buf  # user can slice if needed

    # ---- lock API ----
    def acquire(self, timeout_ms: Optional[int] = None) -> bool:
        """
        Acquire exclusive access using Windows Mutex.
        - timeout_ms=None => infinite wait
        - spin parameter is kept for API compatibility but ignored (Mutex handles waiting efficiently)
        
        Returns True if lock acquired, False if timeout.
        """
        if timeout_ms is None:
            wait_time = INFINITE
        else:
            wait_time = max(0, timeout_ms)
        
        result = _WaitForSingleObject(self._mutex_handle, wait_time)
        
        if result == WAIT_OBJECT_0:
            # Successfully acquired the mutex
            return True
        elif result == WAIT_ABANDONED:
            # Previous owner died without releasing, but we now own it
            # This is still considered a successful acquisition
            return True
        elif result == WAIT_TIMEOUT:
            # Timeout occurred
            return False
        else:  # WAIT_FAILED or other error
            error = ctypes.get_last_error()
            raise OSError(f"WaitForSingleObject failed with error: {error}")

    def release(self) -> None:
        """Release the mutex."""
        if not _ReleaseMutex(self._mutex_handle):
            error = ctypes.get_last_error()
            raise OSError(f"ReleaseMutex failed with error: {error}")

    @contextmanager
    def lock(self, timeout_ms: Optional[int] = None):
        """
        Context manager for acquiring and releasing the lock.
        - timeout_ms: timeout in milliseconds (None for infinite)
        - spin: kept for API compatibility but ignored
        """
        if not self.acquire(timeout_ms=timeout_ms):
            raise TimeoutError("acquire() timeout")
        try:
            yield self
        finally:
            self.release()

    # ---- lifecycle ----
    def close(self) -> None:
        """Close handles and shared memory."""
        if self._mutex_handle and self._mutex_handle != 0:
            _CloseHandle(self._mutex_handle)
            self._mutex_handle = None
        self.payload_shm.close()

    def unlink_control_if_owner(self) -> None:
        """
        Named mutex is automatically cleaned up by Windows when all handles are closed.
        This method is kept for API compatibility but does nothing.
        """
        pass
