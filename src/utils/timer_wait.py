import ctypes
import time

ntdll = ctypes.WinDLL("ntdll")

desired = ctypes.c_ulong(5000)  # 0.5 ms = 5000 * 100ns
current = ctypes.c_ulong()

ntdll.NtSetTimerResolution(desired, True, ctypes.byref(current)) # Set timer resolution to 0.5ms

# -------------------------------
# 高精度等待函数
# -------------------------------
def wait_until(
    target_time: float,
    spin_threshold: float = 0.0005,
    sleep_min: float = 0.001,
):
    """
    等待直到 perf_counter() >= target_time

    :param target_time: 目标时间（time.perf_counter() 的绝对时间，秒）
    :param spin_threshold: 最后 busy-wait 窗口（秒，默认 0.5ms）
    :param sleep_min: 允许 sleep 的最小时间（避免 0 sleep）
    """

    # 防止负时间 / NaN
    if not isinstance(target_time, (int, float)):
        return

    while True:
        now = time.perf_counter()
        remaining = target_time - now

        # 已经过点
        if remaining <= 0:
            return

        # 进入 busy-wait 区间
        if remaining <= spin_threshold:
            break

        # 可 sleep 区间
        sleep_time = remaining - spin_threshold

        # 防止 sleep(0) 或极小值
        if sleep_time >= sleep_min:
            time.sleep(sleep_time)
        else:
            # 太小了，直接进入自旋
            break

    # -------- busy-wait 精对齐 --------
    # 使用局部变量减少属性查找开销
    perf = time.perf_counter
    while perf() < target_time:
        pass
