import math
from multiprocessing import Process, Value, shared_memory, Event
import cv2
import numpy as np
import time
from .ezvtb_rt_interface import get_core
from .args import args
from .utils.shared_mem_guard import SharedMemoryGuard
from .utils.pose_simplify import pose_simplify
from .utils.fps import FPS, Interval
from typing import List

class ModelClientProcess(Process):
    def __init__(self, input_image, pose_position_shm: shared_memory.SharedMemory , input_fps):
        super().__init__()
        self.input_image = input_image
        self.pose_position_shm = pose_position_shm  # 45 floats for pose, 4 floats for position
        
        self.alpha_width_scale = 2 if args.alpha_split else 1
        self.ret_channels = 3 if args.output_virtual_cam or args.output_debug else 4
        self.ret_shape = (args.interpolation_scale, args.model_output_size, self.alpha_width_scale * args.model_output_size, self.ret_channels)
        self.ret_nbytes = self.alpha_width_scale * args.interpolation_scale * args.model_output_size * args.model_output_size * self.ret_channels # RGBA
        self.ret_shared_mem = shared_memory.SharedMemory(create=True, size=self.ret_nbytes)
        
        self.last_model_interval = Value('f', 0.0)
        self.average_model_interval = Value('f', 0.0)
        self.cache_hit_ratio = Value('f', 0.0)
        self.gpu_cache_hit_ratio = Value('f', 0.0)
        self.pipeline_fps_number = Value('f', 0.0)
        self.input_fps = input_fps

        self.finish_event = Event()

    def run(self):
        pose_position_shm_guard = SharedMemoryGuard(self.pose_position_shm, ctrl_name="pose_position_shm_ctrl")
        np_pose_shm = np.ndarray((45,), dtype=np.float32, buffer=self.pose_position_shm.buf[:45 * 4])
        np_position_shm = np.ndarray((4,), dtype=np.float32, buffer=self.pose_position_shm.buf[45 * 4:45 * 4 + 4 * 4])
        
        ret_batch_shm_guard = [
            SharedMemoryGuard(self.ret_shared_mem, ctrl_name=f"ret_shm_ctrl_batch_{i}")
            for i in range(args.interpolation_scale)
        ]
        np_ret_shms = [
            np.ndarray((args.model_output_size, self.alpha_width_scale * args.model_output_size, self.ret_channels), dtype=np.uint8,
                        buffer=self.ret_shared_mem.buf[i * self.alpha_width_scale * args.model_output_size * args.model_output_size * self.ret_channels:
                                                       (i + 1) * self.alpha_width_scale * args.model_output_size * args.model_output_size * self.ret_channels])
            for i in range(args.interpolation_scale)
        ]

        model_infer_average_interval: Interval = Interval()
        pipeline_fps = FPS()

        # Use unified ezvtb_rt interface for both THA3 and THA4
        model = get_core(use_tensorrt=args.use_tensorrt,
                            model_version=args.model_version,
                            model_name=args.model_name,

                            model_seperable = args.model_seperable,
                            model_half=args.model_half, 
                            model_cache_size=args.max_gpu_cache_len, 
                            model_use_eyebrow=args.eyebrow,

                            use_interpolation=args.use_interpolation,
                            interpolation_scale=args.interpolation_scale,
                            interpolation_half=args.interpolation_half,

                            cacher_ram_size=args.max_ram_cache_len,

                            use_sr=args.use_sr,
                            sr_half=args.sr_half,
                            sr_x4=args.sr_x4,
                            sr_a4k=args.sr_a4k,
                            )
        model.setImage(self.input_image)
        model_infer_average_interval.start()
        model.inference([np.zeros((1, 45), dtype=np.float32)])  # Warm up
        model_infer_average_interval.stop()
        self.last_model_interval.value = model_infer_average_interval.last()

        last_pose = np.zeros((45,), dtype=np.float32)

        print("Model Inference Ready")
        while True:
            with pose_position_shm_guard.lock():
                np_pose = np_pose_shm.copy()
                np_position = np_position_shm.copy()

            input_poses = []
            increment = (np_pose - last_pose) / args.interpolation_scale
            for i in range(args.interpolation_scale):
                input_poses.append(pose_simplify(last_pose + increment * (i + 1)))
            last_pose = np_pose

            model_infer_average_interval.start()
            output_images = model.inference(input_poses)

            if args.max_ram_cache_len > 0:
                hits = model.cacher.hits
                miss = model.cacher.miss
                if args.use_sr:
                    hits += model.sr_cacher.hits
                    miss += model.sr_cacher.miss
                total = hits + miss
                self.cache_hit_ratio.value = (hits / total) if total > 0 else 0.0

            if args.use_tensorrt and args.max_gpu_cache_len > 0:
                hits = model.tha.cacher.hits
                miss = model.tha.cacher.miss
                total = hits + miss
                self.gpu_cache_hit_ratio.value = (hits / total) if total > 0 else 0.0

            output_images = self.post_process_ret(np_position, output_images)

            self.average_model_interval.value = model_infer_average_interval.stop()
            self.last_model_interval.value = model_infer_average_interval.last()

            self.pipeline_fps_number.value = pipeline_fps()
            for i in range(args.interpolation_scale):
                with ret_batch_shm_guard[i].lock(): # get pressure from main process if ret not consumed
                    np_ret_shms[i][:, :, :] = output_images[i]

            self.finish_event.set() # Back pressure main process loop if infer slow

    def post_process_ret(self, np_position: np.ndarray, output_images: np.ndarray) -> List[np.ndarray]:
        k_scale = 1
        rotate_angle = 0
        dx = 0
        dy = 0
        if args.extend_movement:
            k_scale = np_position[2] * math.sqrt(args.extend_movement) + 1
            rotate_angle = -np_position[0] * 10 * args.extend_movement
            dx = np_position[0] * 400 * k_scale * args.extend_movement
            dy = -np_position[1] * 600 * k_scale * args.extend_movement
        if args.bongo:
            rotate_angle -= 5

        rm = cv2.getRotationMatrix2D((output_images[0].shape[1] / 2, output_images[0].shape[0] / 2), rotate_angle, k_scale)
        rm[0, 2] += dx + output_images[0].shape[1] / 2 - output_images[0].shape[1] / 2
        rm[1, 2] += dy + output_images[0].shape[0] / 2 - output_images[0].shape[0] / 2

        ret = []
        for i in range(output_images.shape[0]):
            bgra_image = output_images[i]
            bgra_image = cv2.warpAffine(
                bgra_image,
                rm,
                (bgra_image.shape[1], bgra_image.shape[0]))

            if args.output_debug:
                cv2.putText(bgra_image, str('OUT_FPS:%.1f' % (self.pipeline_fps_number.value * args.interpolation_scale)), (0, 16), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 1)
                cv2.putText(bgra_image, str(
                    'GPU_FPS:%.1f' % (
                        1.0 / (self.average_model_interval.value + 1e-6) if not args.use_interpolation else 1.0 / (self.average_model_interval.value + 1e-6) * args.interpolation_scale)),
                            (0, 32),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                cv2.putText(bgra_image, str('INPUT_FPS:%.1f' % self.input_fps.value), (0, 48),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                if args.max_ram_cache_len > 0:
                    cv2.putText(bgra_image, str('MEMCACHED:%.1f%%' % (self.cache_hit_ratio.value * 100)),
                                (0, 64),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                if args.max_gpu_cache_len > 0.0:
                    cv2.putText(bgra_image, str('GPUCACHED:%.1f%%' % (self.gpu_cache_hit_ratio.value * 100)),
                                (0, 80),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

            if args.alpha_split:
                rgba_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA)
                alpha_channel = rgba_image[:, :, 3]
                rgb_channels = rgba_image[:,:,:3]
                alpha_image = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2RGB)
                rgb_channels = cv2.hconcat([rgb_channels, alpha_image])

            if args.output_debug:
                if args.alpha_split:
                    bgr_channels = cv2.cvtColor(rgb_channels, cv2.COLOR_RGB2BGR)
                else:
                    bgr_channels = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR)
                ret.append(bgr_channels)
            elif args.output_virtual_cam:
                if not args.alpha_split:
                    rgb_channels = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGB)
                ret.append(rgb_channels)
            else:
                rgba_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA)
                ret.append(rgba_image)
        return ret
    
if __name__ == "__main__":
    pass