from multiprocessing import Process, shared_memory, Value
from pynput.mouse import Controller
import numpy as np
import time
from .args import args
from .utils.filter import OneEuroFilterNumpy
from .utils.channel_shared_mem import SharedMemoryExclusiveChannel
from .utils.timer_wait import wait_until

class MouseClientProcess(Process):
    def __init__(self, pose_position_shm: shared_memory.SharedMemory):
        super().__init__()
        self.pose_position_shm = pose_position_shm
        self.fps = Value('f', 60.0)

    def run(self):
        mouse = Controller()
        pose_position_shm_channel = SharedMemoryExclusiveChannel(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[:45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4:45 * 4 + 4 * 4])
        
        # posLimit = [0, 0, 1920, 1080]
        posLimit = [int(x) for x in args.mouse_input.split(',')]
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
        pose_filter = OneEuroFilterNumpy(freq=60, mincutoff=args.filter_min_cutoff, beta=args.filter_beta)
        position_vector = np.array([0, 0, 0, 1], dtype=np.float32)
        while True:
            pos = mouse.position
            # print(pos)
            eye_limit = [0.8, 0.5]
            head_eye_reduce = 0.6
            head_slowness = 0.2
            mouse_data = {
                'eye_l_h_temp': 0,
                'eye_r_h_temp': 0,
                'mouth_ratio': 0,
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

            with pose_position_shm_channel.lock():
                np_pose_shm[:] = pose_filter(np.array(model_input_arr, dtype=np.float32))
                np_position_shm[:] = position_vector

            wait_until(last_time + interval)
            last_time += interval