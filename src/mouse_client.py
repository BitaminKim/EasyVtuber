from multiprocessing import Process, shared_memory, Value
from pynput.mouse import Controller
import numpy as np
import time
import sounddevice as sd
from .args import args
from .utils.shared_mem_guard import SharedMemoryGuard
from .utils.timer_wait import wait_until
from OneEuroFilter import OneEuroFilter
from .utils.fps import FPS
from .utils.timer_wait import wait_until

class MouseClientProcess(Process):
    def __init__(self, pose_position_shm: shared_memory.SharedMemory):
        super().__init__()
        self.pose_position_shm = pose_position_shm
        self.fps = Value('f', 60.0)
        self.audio_volume = Value('f', 0.0)
        self.audio_running = False
        self.audio_callback_fps = FPS(60)

    def audio_callback(self, indata, frames, time_info, status):
        """音频回调函数，计算音量"""
        if status:
            print(f"Audio status: {status}")
        
        self.audio_callback_fps()
        # 计算 RMS (均方根) 音量
        volume_norm = np.linalg.norm(indata) * 10

        # 应用阈值和灵敏度
        if volume_norm < args.audio_threshold:
            volume_norm = 0
        else:
            volume_norm = (volume_norm - args.audio_threshold) * args.audio_sensitivity
        # 限制在 0-1 范围
        volume_norm = np.clip(volume_norm, 0, 1)
        self.audio_volume.value = volume_norm

    def run(self):
        mouse = Controller()
        pose_position_shm_guard = SharedMemoryGuard(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[:45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4:45 * 4 + 4 * 4])
        
        # posLimit = [0, 0, 1920, 1080]
        posLimit = [int(x) for x in args.mouse_input.split(',')]
        
        if args.mouse_audio_input:
            # 启动音频流 (使用 WASAPI)
            try:
                # 获取默认输入设备（麦克风或系统音频）
                # 如果要捕获系统音频输出，需要使用 loopback 设备
                audio_stream = sd.InputStream(
                    callback=self.audio_callback,
                    channels=1,
                    samplerate=16000,
                    blocksize=1024,
                )
                audio_stream.start()
                self.audio_running = True
                wait_until(time.perf_counter() + 1.0)  # 等待一秒以稳定音频输入
                print("Audio capture started (WASAPI)")
            except Exception as e:
                print(f"Failed to start audio capture: {e}")
                print("Continuing without audio input...")
                self.audio_running = False
            audio_filter = OneEuroFilter(freq=self.audio_callback_fps.view(), mincutoff=10.0, beta=0.0)
        
        prev = {
            'eye_l_h_temp': 0,
            'eye_r_h_temp': 0,
            'mouth_ratio': 0,
            'eye_y_ratio': 0,
            'eye_x_ratio': 0,
            'x_angle': 0,
            'y_angle': 0,
            'z_angle': 0,
        }
        last_time : float = time.perf_counter()
        interval : float = 1.0 / 60 # 60 FPS

        print("Mouse Input Running at 60 FPS")
        position_vector = np.array([0, 0, 0, 1], dtype=np.float32)
        while True:
            pos = mouse.position
            # print(pos)
            eye_limit = [0.8, 0.5]
            head_eye_reduce = 0.6
            head_slowness = 0.2
            
            # 获取音频音量并映射到 mouth_ratio
            current_mouth_ratio = audio_filter(self.audio_volume.value, time.perf_counter()) if self.audio_running else 0
            mouse_data = {
                'eye_l_h_temp': 0,
                'eye_r_h_temp': 0,
                'mouth_ratio': current_mouth_ratio,
                'eye_y_ratio': np.interp(pos[1], [posLimit[1], posLimit[3]], [1, -1]) * eye_limit[1],
                'eye_x_ratio': np.interp(pos[0], [posLimit[0], posLimit[2]], [1, -1]) * eye_limit[0],
                'x_angle': np.interp(pos[1], [posLimit[1], posLimit[3]], [1, -1]),
                'y_angle': np.interp(pos[0], [posLimit[0], posLimit[2]], [1, -1]),
                'z_angle': 0,
            }
            mouse_data['x_angle'] = np.interp(head_slowness, [0, 1], [prev['x_angle'], mouse_data['x_angle']])
            mouse_data['y_angle'] = np.interp(head_slowness, [0, 1], [prev['y_angle'], mouse_data['y_angle']])
            mouse_data['eye_y_ratio'] -= mouse_data['x_angle'] * eye_limit[1] * head_eye_reduce
            mouse_data['eye_x_ratio'] -= mouse_data['y_angle'] * eye_limit[0] * head_eye_reduce
            if args.bongo:
                mouse_data['y_angle'] += 0.05
                mouse_data['x_angle'] += 0.05
            prev = mouse_data

            eye_l_h_temp = mouse_data['eye_l_h_temp']
            eye_r_h_temp = mouse_data['eye_r_h_temp']
            mouth_ratio = mouse_data['mouth_ratio']
            eye_y_ratio = mouse_data['eye_y_ratio']
            eye_x_ratio = mouse_data['eye_x_ratio']
            x_angle = mouse_data['x_angle']
            y_angle = mouse_data['y_angle']
            z_angle = mouse_data['z_angle']

            eyebrow_vector = [0.0] * 12
            mouth_eye_vector = [0.0] * 27
            pose_vector = [0.0] * 6

            mouth_eye_vector[2] = eye_l_h_temp
            mouth_eye_vector[3] = eye_r_h_temp

            mouth_eye_vector[14] = mouth_ratio * 1.5

            mouth_eye_vector[25] = eye_y_ratio
            mouth_eye_vector[26] = eye_x_ratio

            pose_vector[0] = x_angle
            pose_vector[1] = y_angle
            pose_vector[2] = z_angle
            pose_vector[3] = pose_vector[1]
            pose_vector[4] = pose_vector[2]

            model_input_arr = eyebrow_vector
            model_input_arr.extend(mouth_eye_vector)
            model_input_arr.extend(pose_vector)

            with pose_position_shm_guard.lock():
                np_pose_shm[:] = np.array(model_input_arr, dtype=np.float32)
                np_position_shm[:] = position_vector

            wait_until(last_time + interval)
            last_time += interval