from multiprocessing import Process, shared_memory, Value
import numpy as np
import time
import math
from .utils.timer_wait import wait_until
from .utils.shared_mem_guard import SharedMemoryGuard
from .args import args

class DebugInputClientProcess(Process):
    def __init__(self, pose_position_shm: shared_memory.SharedMemory):
        super().__init__()
        self.pose_position_shm = pose_position_shm
        self.fps = Value('f', 60.0)

    def run(self):
        last_time : float = time.perf_counter()
        interval : float = 1.0 / 60 # 60 FPS
        pose_position_shm_guard = SharedMemoryGuard(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[:45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4:45 * 4 + 4 * 4])
        # 呼吸循环参数
        breath_start_time = time.perf_counter()
        while True:
            eyebrow_vector = [0.0] * 12
            mouth_eye_vector = [0.0] * 27
            pose_vector = [0.0] * 6
            position_vector = [0, 0, 0, 1]

            # 计算呼吸效果（使用 sin 函数，在 breath_cycle 时间内从 0 到 1 再到 0）
            breath_elapsed = (time.perf_counter() - breath_start_time) % args.breath_cycle
            # 使用 sin 函数，让值在一个周期内从 0 -> 1 -> 0
            # sin 在 0 到 π 之间从 0 到 1 到 0
            breath_value = np.sin(breath_elapsed / args.breath_cycle * np.pi)

            mouth_eye_vector[2] = math.sin(time.perf_counter() * 3)
            mouth_eye_vector[3] = math.sin(time.perf_counter() * 3)

            mouth_eye_vector[14] = 0

            mouth_eye_vector[25] = math.sin(time.perf_counter() * 2.2) * 0.2
            mouth_eye_vector[26] = math.sin(time.perf_counter() * 3.5) * 0.8

            pose_vector[0] = math.sin(time.perf_counter() * 1.1)
            pose_vector[1] = math.sin(time.perf_counter() * 1.2)
            pose_vector[2] = math.sin(time.perf_counter() * 1.5)
            pose_vector[3] = pose_vector[1]
            pose_vector[4] = pose_vector[2]
            pose_vector[5] = breath_value
            eyebrow_vector[6] = math.sin(time.perf_counter() * 1.1)
            eyebrow_vector[7] = math.sin(time.perf_counter() * 1.1)

            model_input_arr = eyebrow_vector
            model_input_arr.extend(mouth_eye_vector)
            model_input_arr.extend(pose_vector)

            position_vector[0] = math.sin(time.perf_counter() * 0.5) * 0.1
            position_vector[1] = math.sin(time.perf_counter() * 0.6) * 0.1
            position_vector[2] = math.sin(time.perf_counter() * 0.7) * 0.1
            position_vector[3] = 1

            with pose_position_shm_guard.lock():
                np_pose_shm[:] = np.array(model_input_arr, dtype=np.float32)
                np_position_shm[:] = np.array(position_vector, dtype=np.float32)
            wait_until(last_time + interval)
            last_time += interval