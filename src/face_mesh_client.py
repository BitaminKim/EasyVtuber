import mediapipe as mp
from multiprocessing import Process, shared_memory, Value
import cv2
import numpy as np
from .args import args
from .utils.shared_mem_guard import SharedMemoryGuard
from .utils.pose import get_pose
from .utils.fps import FPS
from .utils.filter import OneEuroFilterNumpy
from OneEuroFilter import OneEuroFilter
import time


class FaceMeshClientProcess(Process):
    def __init__(self, pose_position_shm: shared_memory.SharedMemory):
        super().__init__()
        self.pose_position_shm = pose_position_shm
        self.fps = Value('f', 0.0)

    def run(self):
        facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        pose_position_shm_guard = SharedMemoryGuard(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[:45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4:45 * 4 + 4 * 4])
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise Exception("Can't open webcam")
        
        print("Warming up webcam...")

        frame_count = 0
        input_fps = FPS(60)
        while frame_count < 5:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Can't receive frame (stream end?).")
            input_fps()
            frame_count += 1

        # 呼吸循环参数
        breath_start_time = time.perf_counter()
        pose_filter = OneEuroFilterNumpy(freq=input_fps.view(), mincutoff=args.filter_min_cutoff, beta=args.filter_beta)
        position_offset = None
        print("Webcam Input Running at {:.2f} FPS".format(input_fps.view()))
        position_vector = np.array([0, 0, 0, 1], dtype=np.float32)
        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Can't receive frame (stream end?).")
            self.fps.value = input_fps()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = facemesh.process(rgb_frame)
            if results.multi_face_landmarks is None:
                continue

            # 计算呼吸效果（使用 sin 函数，在 breath_cycle 时间内从 0 到 1 再到 0）
            breath_elapsed = (time.perf_counter() - breath_start_time) % args.breath_cycle
            # 使用 sin 函数，让值在一个周期内从 0 -> 1 -> 0
            # sin 在 0 到 π 之间从 0 到 1 到 0
            breath_value = np.sin(breath_elapsed / args.breath_cycle * np.pi)

            facial_landmarks = results.multi_face_landmarks[0].landmark
            pose = get_pose(facial_landmarks)
            eye_l_h_temp = pose[0]
            eye_r_h_temp = pose[1]
            mouth_ratio = pose[2]
            eye_y_ratio = pose[3]
            eye_x_ratio = pose[4]
            x_angle = pose[5]
            y_angle = pose[6]
            z_angle = pose[7]

            eyebrow_vector = [0.0] * 12
            mouth_eye_vector = [0.0] * 27
            pose_vector = [0.0] * 6

            mouth_eye_vector[2] = max(eye_l_h_temp, eye_r_h_temp)
            mouth_eye_vector[3] = max(eye_l_h_temp, eye_r_h_temp)

            mouth_eye_vector[14] = mouth_ratio * 2.0

            mouth_eye_vector[25] = 0.0 # keep iris stable for user demo
            mouth_eye_vector[26] = 0.0
            if position_offset is None:
                position_offset = [(x_angle - 1.5) * 1.6, y_angle * 2.0 , (z_angle + 1.5) * 2]
            pose_vector[0] = (x_angle - 1.5) * 1.6 - position_offset[0]
            pose_vector[1] = y_angle * 2.0 - position_offset[1]  # temp weight
            pose_vector[2] = (z_angle + 1.5) * 2 - position_offset[2]  # temp weight
            pose_vector[3] = pose_vector[1]
            pose_vector[4] = pose_vector[2]
            pose_vector[5] = breath_value

            model_input_arr = eyebrow_vector
            model_input_arr.extend(mouth_eye_vector)
            model_input_arr.extend(pose_vector)

            with pose_position_shm_guard.lock():
                np_pose_shm[:] = pose_filter(np.array(model_input_arr, dtype=np.float32))
                np_position_shm[:] = position_vector