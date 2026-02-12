from multiprocessing import Process, Value, shared_memory
from .args import args
import socket
import struct
import numpy as np
from .utils.shared_mem_guard import SharedMemoryGuard
from .utils.fps import FPS
from .utils.filter import OneEuroFilterNumpy
from OneEuroFilter import OneEuroFilter
import time

class OSFClientProcess(Process):
    def __init__(self, pose_position_shm: shared_memory.SharedMemory):
        super().__init__()
        self.pose_position_shm = pose_position_shm
        self.address = args.osf_input.split(':')[0]
        self.port = int(args.osf_input.split(':')[1])
        self.fps = Value('f', 0.0)

    def run(self):
        pose_position_shm_guard = SharedMemoryGuard(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[:45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4:45 * 4 + 4 * 4])
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("", self.port))
        self.socket.settimeout(None) # wait indefinitely

        print("Warming up OpenSeeFace connection...")

        frame_count = 0
        input_fps = FPS(60)
        while frame_count < 5:
            socket_bytes = self.socket.recv(8192)
            if not socket_bytes:
                raise Exception("Can't receive OpenSeeFace data (stream end?).")
            input_fps()
            frame_count += 1

        position_vector_0 = None
        position_vector = [0, 0, 0, 1]
        # 呼吸循环参数
        breath_start_time = time.perf_counter()
        pose_filter = OneEuroFilterNumpy(freq=input_fps.view(), mincutoff=args.filter_min_cutoff, beta=args.filter_beta)
        position_filter = OneEuroFilterNumpy(freq=input_fps.view(), mincutoff=args.filter_min_cutoff, beta=args.filter_beta)
        iris_x_filter = OneEuroFilter(freq=input_fps.view(), mincutoff=0.001, beta=1.0) # extra filter for iris movement
        iris_y_filter = OneEuroFilter(freq=input_fps.view(), mincutoff=0.001, beta=1.0) # extra filter for iris movement
        rotation_offset = None
        print("OpenSeeFace Input Running at {:.2f} FPS".format(input_fps.view()))
        while True:
            socket_bytes = self.socket.recv(8192)

            if not socket_bytes:
                raise Exception("Can't receive OpenSeeFace data (stream end?).")
            
            self.fps.value = input_fps()

            try:
                osf_raw = (struct.unpack('=di2f2fB1f4f3f3f68f136f210f14f', socket_bytes))
                # print(osf_raw[432:])
                data = {}
                OpenSeeDataIndex = [
                    'time',
                    'id',
                    'cameraResolutionW',
                    'cameraResolutionH',
                    'rightEyeOpen',
                    'leftEyeOpen',
                    'got3DPoints',
                    'fit3DError',
                    'rawQuaternionX',
                    'rawQuaternionY',
                    'rawQuaternionZ',
                    'rawQuaternionW',
                    'rawEulerX',
                    'rawEulerY',
                    'rawEulerZ',
                    'translationY',
                    'translationX',
                    'translationZ',
                ]
                for i in range(len(OpenSeeDataIndex)):
                    data[OpenSeeDataIndex[i]] = osf_raw[i]
                data['translationY'] *= -1
                data['translationZ'] *= -1
                data['rotationY'] = data['rawEulerY'] - 10
                data['rotationX'] = (-data['rawEulerX'] + 360) % 360 - 180
                data['rotationZ'] = (data['rawEulerZ'] - 90)
                OpenSeeFeatureIndex = [
                    'EyeLeft',
                    'EyeRight',
                    'EyebrowSteepnessLeft',
                    'EyebrowUpDownLeft',
                    'EyebrowQuirkLeft',
                    'EyebrowSteepnessRight',
                    'EyebrowUpDownRight',
                    'EyebrowQuirkRight',
                    'MouthCornerUpDownLeft',
                    'MouthCornerInOutLeft',
                    'MouthCornerUpDownRight',
                    'MouthCornerInOutRight',
                    'MouthOpen',
                    'MouthWide'
                ]
            except Exception:
                print("OpenSeeFace data parse error:", socket_bytes)
                continue

            for i in range(68):
                data['confidence' + str(i)] = osf_raw[i + 18]
            for i in range(68):
                data['pointsX' + str(i)] = osf_raw[i * 2 + 18 + 68]
                data['pointsY' + str(i)] = osf_raw[i * 2 + 18 + 68 + 1]
            for i in range(70):
                data['points3DX' + str(i)] = osf_raw[i * 3 + 18 + 68 + 68 * 2]
                data['points3DY' + str(i)] = osf_raw[i * 3 + 18 + 68 + 68 * 2 + 1]
                data['points3DZ' + str(i)] = osf_raw[i * 3 + 18 + 68 + 68 * 2 + 2]

            for i in range(len(OpenSeeFeatureIndex)):
                data[OpenSeeFeatureIndex[i]] = osf_raw[i + 432]
            # print(data['rotationX'],data['rotationY'],data['rotationZ'])

            # 计算呼吸效果（使用 sin 函数，在 breath_cycle 时间内从 0 到 1 再到 0）
            breath_elapsed = (time.perf_counter() - breath_start_time) % args.breath_cycle
            # 使用 sin 函数，让值在一个周期内从 0 -> 1 -> 0
            # sin 在 0 到 π 之间从 0 到 1 到 0
            breath_value = np.sin(breath_elapsed / args.breath_cycle * np.pi)

            a = np.array([
                data['points3DX66'] - data['points3DX68'] + data['points3DX67'] - data['points3DX69'],
                data['points3DY66'] - data['points3DY68'] + data['points3DY67'] - data['points3DY69'],
                data['points3DZ66'] - data['points3DZ68'] + data['points3DZ67'] - data['points3DZ69']
            ])
            a = (a / np.linalg.norm(a))
            data['eyeRotationX'] = a[0]
            data['eyeRotationY'] = a[1]

            eyebrow_vector = [0.0] * 12
            mouth_eye_vector = [0.0] * 27
            pose_vector = [0.0] * 6

            mouth_eye_vector[2] = 1 - data['leftEyeOpen']
            mouth_eye_vector[3] = 1 - data['rightEyeOpen']

            mouth_eye_vector[14] = max(data['MouthOpen'], 0) * 2 # Open larger mouth
            # print(mouth_eye_vector[14])

            mouth_eye_vector[25] = iris_x_filter(-data['eyeRotationY'] * 3 - (data['rotationX']) / 57.3 * 1.5, timestamp=time.perf_counter())
            mouth_eye_vector[26] = iris_y_filter(data['eyeRotationX'] * 3 + (data['rotationY']) / 57.3, timestamp=time.perf_counter())
            # print(mouth_eye_vector[25:27])
            eyebrow_vector[6] = data['EyebrowUpDownLeft']
            eyebrow_vector[7] = data['EyebrowUpDownRight']
            # print(data['EyebrowUpDownLeft'],data['EyebrowUpDownRight'])
            if rotation_offset is None:
                rotation_offset = [data['rotationX'], data['rotationY'], data['rotationZ']]
            pose_vector[0] = (data['rotationX'] - rotation_offset[0]) / 57.3 * 3
            pose_vector[1] = -(data['rotationY'] - rotation_offset[1]) / 57.3 * 3
            pose_vector[2] = (data['rotationZ'] - rotation_offset[2]) / 57.3 * 2
            pose_vector[3] = pose_vector[1]
            pose_vector[4] = pose_vector[2]
            pose_vector[5] = breath_value

            if position_vector_0 == None: #Provide an initial reference point
                position_vector_0 = [0, 0, 0, 1]
                position_vector_0[0] = data['translationX']
                position_vector_0[1] = data['translationY']
                position_vector_0[2] = data['translationZ']
            #Compute relative translation
            position_vector[0] = -(data['translationX'] - position_vector_0[0]) * 0.1
            position_vector[1] = -(data['translationY'] - position_vector_0[1]) * 0.1
            position_vector[2] = -(data['translationZ'] - position_vector_0[2]) * 0.1

            model_input_arr = eyebrow_vector
            model_input_arr.extend(mouth_eye_vector)
            model_input_arr.extend(pose_vector)

            with pose_position_shm_guard.lock():
                np_pose_shm[:] = pose_filter(np.array(model_input_arr, dtype=np.float32))
                np_position_shm[:] = position_filter(np.array(position_vector, dtype=np.float32))
                # np_pose_shm[:] = np.array(model_input_arr, dtype=np.float32)
                # np_position_shm[:] = np.array(position_vector, dtype=np.float32)
