from multiprocessing import Process, shared_memory, Value
from .args import args
from .utils.timer_wait import wait_until
from .utils.shared_mem_guard import SharedMemoryGuard
import socket
import errno
import numpy as np
import time
import tha2.poser.modes.mode_20_wx
from tha2.mocap.ifacialmocap_constants import *
from .utils.fps import FPS
from .utils.filter import OneEuroFilterNumpy
ifm_converter = tha2.poser.modes.mode_20_wx.IFacialMocapPoseConverter20()

class IFMClientProcess(Process):
    def __init__(self, pose_position_shm: shared_memory.SharedMemory):
        super().__init__()
        self.pose_position_shm = pose_position_shm
        self.address = args.ifm_input.split(':')[0]
        self.port = int(args.ifm_input.split(':')[1])
        self.fps = Value('f', 0.0)
    def run(self):
        pose_position_shm_guard = SharedMemoryGuard(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[:45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4:45 * 4 + 4 * 4])
        
        udpClntSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        data = "iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719"

        data = data.encode('utf-8')

        udpClntSock.sendto(data, (self.address, self.port))

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("", self.port))
        self.socket.settimeout(5.0) # give ifm app some time to respond

        print("Warming up iFacialMocap connection...")

        frame_count = 0
        input_fps = FPS(60)
        while frame_count < 60:
            socket_bytes = self.socket.recv(8192)
            if not socket_bytes:
                raise Exception("Can't receive iFacialMocap data (stream end?).")
            input_fps()
            frame_count += 1

        self.socket.settimeout(0.5) # 500 ms

        # pose_filter = OneEuroFilterNumpy(freq=input_fps.view(), mincutoff=args.filter_min_cutoff, beta=args.filter_beta)
        # position_filter = OneEuroFilterNumpy(freq=input_fps.view(), mincutoff=args.filter_min_cutoff, beta=args.filter_beta)
        # 呼吸循环参数
        breath_start_time = time.perf_counter()
        print("iFacialMocap Input Running at {:.2f} FPS".format(input_fps.view()))
        while True:
            socket_bytes = self.socket.recv(8192)

            if not socket_bytes:
                raise Exception("Can't receive iFacialMocap data (stream end?).")
            
            self.fps.value = input_fps()

            socket_string = socket_bytes.decode("utf-8")

            try:
                data = self.convert_from_blender_data(socket_string)
            except Exception:
                print("iFacialMocap data parse error:", socket_string)
                continue

            # 计算呼吸效果（使用 sin 函数，在 breath_cycle 时间内从 0 到 1 再到 0）
            breath_elapsed = (time.perf_counter() - breath_start_time) % args.breath_cycle
            # 使用 sin 函数，让值在一个周期内从 0 -> 1 -> 0
            # sin 在 0 到 π 之间从 0 到 1 到 0
            breath_value = np.sin(breath_elapsed / args.breath_cycle * np.pi)

            ifacialmocap_pose_converted = ifm_converter.convert(data)
            position_vector = data[HEAD_BONE_QUAT]

            eyebrow_vector = [0.0] * 12
            mouth_eye_vector = [0.0] * 27
            pose_vector = [0.0] * 6
            for i in range(0, 12):
                eyebrow_vector[i] = ifacialmocap_pose_converted[i]
            for i in range(12, 39):
                mouth_eye_vector[i - 12] = ifacialmocap_pose_converted[i]
            for i in range(39, 42):
                pose_vector[i - 39] = ifacialmocap_pose_converted[i]
            pose_vector[3] = pose_vector[1]
            pose_vector[4] = pose_vector[2]
            pose_vector[5] = breath_value

            mouth_eye_vector[14] = mouth_eye_vector[14] * 1.5

            model_input_arr = eyebrow_vector
            model_input_arr.extend(mouth_eye_vector)
            model_input_arr.extend(pose_vector)

            with pose_position_shm_guard.lock():
                # np_pose_shm[:] = pose_filter(np.array(model_input_arr, dtype=np.float32))
                # np_position_shm[:] = position_filter(np.array(position_vector, dtype=np.float32))
                np_pose_shm[:] = np.array(model_input_arr, dtype=np.float32)
                np_position_shm[:] = np.array(position_vector, dtype=np.float32)

    @staticmethod
    def convert_from_blender_data(blender_data):
        data = {}

        for item in blender_data.split('|'):
            if item.find('#') != -1:
                k, arr = item.split('#')
                arr = [float(n) for n in arr.split(',')]
                data[k.replace("_L", "Left").replace("_R", "Right")] = arr
            elif item.find('-') != -1:
                k, v = item.split("-")
                data[k.replace("_L", "Left").replace("_R", "Right")] = float(v) / 100

        to_rad = 57.3
        data[HEAD_BONE_X] = data["=head"][0] / to_rad
        data[HEAD_BONE_Y] = data["=head"][1] / to_rad
        data[HEAD_BONE_Z] = data["=head"][2] / to_rad
        data[HEAD_BONE_QUAT] = [data["=head"][3], data["=head"][4], data["=head"][5], 1]
        # print(data[HEAD_BONE_QUAT][2],min(data[EYE_BLINK_LEFT],data[EYE_BLINK_RIGHT]))
        data[RIGHT_EYE_BONE_X] = data["rightEye"][0] / to_rad
        data[RIGHT_EYE_BONE_Y] = data["rightEye"][1] / to_rad
        data[RIGHT_EYE_BONE_Z] = data["rightEye"][2] / to_rad
        data[LEFT_EYE_BONE_X] = data["leftEye"][0] / to_rad
        data[LEFT_EYE_BONE_Y] = data["leftEye"][1] / to_rad
        data[LEFT_EYE_BONE_Z] = data["leftEye"][2] / to_rad

        return data