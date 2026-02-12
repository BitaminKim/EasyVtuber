"""
THA4 Core - PyTorch THA4 model with optional TensorRT RIFE/SR acceleration
Provides interface compatible with CoreTRT/CoreORT for THA4 models
"""
import torch
import numpy as np
import cv2
from tha4_adapter import THA4Wrapper, convert_tha3_pose_to_tha4

# Import color space conversion from THA4
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tha4', 'src'))
from tha4.image_util import convert_linear_to_srgb

# Add ezvtuber-rt path for TensorRT modules
dir_path = os.path.dirname(os.path.realpath(__file__))
ezvtb_path = os.path.join(dir_path, 'ezvtuber-rt')
if os.path.exists(ezvtb_path) and ezvtb_path not in sys.path:
    sys.path.append(ezvtb_path)


class THA4Core:
    """
    THA4 Core class - Pure PyTorch implementation (No TensorRT)
    
    This class only uses PyTorch for THA4 model inference.
    It does NOT support real frame interpolation or super resolution.
    If interpolation_scale > 1, frames will be simply duplicated.
    
    For real RIFE interpolation and SR with TensorRT, use THA4CoreWithAccel.
    
    Uses CharacterModel to load YAML + PNG + PT files properly.
    The PNG image from YAML is used as the source image.
    
    Expected interface:
    - setImage(img: np.ndarray) - NOT USED, image comes from YAML
    - inference(pose: np.ndarray) -> List[np.ndarray] - run inference
    """
    
    def __init__(self, device_id=0, use_eyebrow=True, yaml_path=None,
                 interpolation_scale=1):
        """
        Initialize THA4 Core
        
        Args:
            device_id: GPU device ID
            use_eyebrow: whether to use eyebrow parameters
            yaml_path: path to character_model.yaml
            interpolation_scale: number of output frames to generate
                      (THA4 doesn't support real interpolation,
                       will duplicate frames)
        """
        device_str = f'cuda:{device_id}' if torch.cuda.is_available() \
            else 'cpu'
        self.device = torch.device(device_str)
        self.use_eyebrow = use_eyebrow
        self.interpolation_scale = interpolation_scale
        
        # Create THA4 wrapper (loads YAML + PNG + PT)
        self.wrapper = THA4Wrapper(device=self.device, yaml_path=yaml_path)
        
        # Use character image from YAML (already loaded by wrapper)
        self.input_image_tensor = self.wrapper.character_image
        
        # For compatibility (no caching in THA4 core)
        self.cacher = None
        
        print(f"THA4 Core initialized on {self.device}")
        print("Using character image from model (ignoring setImage calls)")
        if interpolation_scale > 1:
            print(f"Note: Pure PyTorch mode doesn't support real "
                  f"interpolation, will duplicate frames {interpolation_scale}x")
            print("      Use THA4CoreWithAccel with TensorRT for "
                  "real RIFE interpolation")
    
    def setImage(self, img: np.ndarray):
        """
        Set input character image - IGNORED for THA4
        
        THA4 uses the character image from the YAML/PNG files loaded
        during initialization. This method exists only for interface
        compatibility with CoreTRT/CoreORT.
        
        Args:
            img: numpy array in BGRA format (ignored)
        """
        # THA4 uses character image from YAML, ignore external image
        print("THA4: setImage() called but ignored (using image from YAML)")
    
    def inference(self, pose: np.ndarray) -> list:
        """
        Run inference
        
        Args:
            pose: numpy array of shape (1, 45)
                  [0:12] - eyebrow parameters
                  [12:39] - face parameters
                  [39:45] - head pose parameters
                  
        Returns:
            List of output images as numpy arrays [H, W, 4]
            in BGRA format, uint8
        """
        if self.input_image_tensor is None:
            raise RuntimeError("Image not set. Call setImage() first.")
        
        # Extract pose components
        if self.use_eyebrow:
            eyebrow_pose = pose[:, :12]  # [1, 12]
        else:
            eyebrow_pose = np.zeros((1, 12), dtype=np.float32)
            
        face_pose = pose[:, 12:39]  # [1, 27]
        head_pose = pose[:, 39:45]  # [1, 6]
        
        # Convert to torch tensors
        eyebrow_tensor = torch.from_numpy(eyebrow_pose).to(self.device)
        face_tensor = torch.from_numpy(face_pose).to(self.device)
        head_tensor = torch.from_numpy(head_pose).to(self.device)
        
        # Call wrapper (it handles the pose conversion internally)
        # We need to create dummy compressed versions for
        # interface compatibility
        eyebrow_c = eyebrow_pose[0].tolist()
        face_c = face_pose[0].tolist()
        
        with torch.no_grad():
            output_tensor = self.wrapper.forward(
                self.input_image_tensor,
                face_tensor,
                head_tensor,
                eyebrow_tensor,
                face_c,
                eyebrow_c,
                ratio=None
            )
        
        # THA4 outputs in linear color space, convert to sRGB
        # Output is [batch, 4, H, W] in range [-1, 1]
        output_image = output_tensor[0].float()  # [4, 512, 512]
        
        # Clip to [-1, 1] and convert to [0, 1]
        output_image = torch.clamp((output_image + 1.0) / 2.0, 0.0, 1.0)
        
        # Convert linear RGB to sRGB (keep alpha as-is)
        output_image = convert_linear_to_srgb(output_image)
        
        # Convert to [H, W, C] numpy
        output_np = output_image.permute(1, 2, 0).cpu().numpy()
        
        # Convert to uint8
        output_np = (output_np * 255.0).astype(np.uint8)
        
        # Convert RGBA to BGRA for output
        output_bgra = cv2.cvtColor(output_np, cv2.COLOR_RGBA2BGRA)
        
        # Return as list with proper number of frames
        # Pure PyTorch mode: duplicate frames (no real interpolation)
        # Use THA4CoreWithAccel with TensorRT for real RIFE interpolation
        return [output_bgra] * self.interpolation_scale


class THA4CoreWithAccel:
    """
    THA4 Core with TensorRT acceleration for RIFE and SR
    
    Architecture:
    - THA4 main model: PyTorch (PT files) - always PyTorch
    - RIFE interpolation: TensorRT (optional) - REAL frame interpolation
    - SR super-resolution: TensorRT (optional) - REAL upscaling
    - Cacher: RAM cache (optional)
    
    This class provides REAL interpolation and super resolution using
    TensorRT, unlike THA4Core which only duplicates frames.
    
    Follows the same design as CoreTRT for THA3 models.
    """
    
    def __init__(self, device_id=0, use_eyebrow=True, yaml_path=None,
                 use_tensorrt=False, use_interpolation=False,
                 interpolation_scale=2, interpolation_half=True,
                 use_sr=False, sr_x4=True, sr_half=True, sr_noise=1,
                 cacher_quality=85, cacher_ram_size=2.0):
        """
        Initialize THA4 Core with optional TensorRT acceleration
        
        Args:
            device_id: GPU device ID
            use_eyebrow: whether to use eyebrow parameters
            yaml_path: path to character_model.yaml
            use_tensorrt: enable TensorRT for RIFE/SR
            use_interpolation: enable RIFE frame interpolation
            interpolation_scale: interpolation scale (2/3/4)
            interpolation_half: use FP16 for RIFE
            use_sr: enable super resolution
            sr_x4: use 4x SR (else 2x)
            sr_half: use FP16 for SR
            sr_noise: noise level for waifu2x (1/2/3)
            cacher_quality: cache compression quality
            cacher_ram_size: RAM cache size in GB
        """
        self.device_id = device_id
        self.use_eyebrow = use_eyebrow
        self.use_tensorrt = use_tensorrt
        self.use_interpolation = use_interpolation
        self.use_sr = use_sr
        
        # Initialize PyTorch device
        device_str = f'cuda:{device_id}' if torch.cuda.is_available() \
            else 'cpu'
        self.device = torch.device(device_str)
        
        # Initialize THA4 wrapper (PyTorch)
        self.wrapper = THA4Wrapper(device=self.device, yaml_path=yaml_path)
        self.input_image_tensor = self.wrapper.character_image
        
        # Initialize TensorRT components if requested
        self.rife = None
        self.sr = None
        self.cacher = None
        self.scale = 1
        self.instream = None
        self.output_memory = None
        
        if use_tensorrt and (use_interpolation or use_sr):
            # Initialize CUDA context for TensorRT
            self._init_tensorrt(device_id)
            
            # Build model paths
            rife_model_path = ''
            if use_interpolation:
                rife_model_path = os.path.join(
                    '.', 'data', 'models', 'rife_512',
                    f'x{interpolation_scale}',
                    'fp16' if interpolation_half else 'fp32')
            
            sr_model_path = ''
            if use_sr:
                if sr_x4:
                    base_path = os.path.join('.', 'data', 'models',
                                            'Real-ESRGAN')
                    sr_model_path = os.path.join(
                        base_path,
                        'exported_256_fp16' if sr_half else 'exported_256')
                else:  # x2
                    base_path = os.path.join('.', 'data', 'models',
                                            'waifu2x_upconv',
                                            'fp16' if sr_half else 'fp32',
                                            'upconv_7', 'art')
                    sr_model_path = os.path.join(
                        base_path, f'noise{sr_noise}_scale2x')
            
            # Initialize TensorRT modules
            self._init_trt_modules(rife_model_path, sr_model_path)
        
        # Initialize cacher
        if cacher_ram_size > 0.0:
            from ezvtb_rt.cache import Cacher
            self.cacher = Cacher(cacher_ram_size, cacher_quality)
        
        print("THA4 Core initialized:")
        print(f"  Device: {self.device}")
        print(f"  TensorRT: {use_tensorrt}")
        interp_info = f"scale: {interpolation_scale}" \
            if use_interpolation else "N/A"
        print(f"  Interpolation: {use_interpolation} ({interp_info})")
        if use_interpolation:
            print("      (Real RIFE interpolation via TensorRT)")
        print(f"  Super Resolution: {use_sr}")
    
    def _init_tensorrt(self, device_id):
        """Initialize TensorRT and CUDA context"""
        try:
            from ezvtb_rt.trt_utils import cudaSetDevice
            cudaSetDevice(device_id)
            os.environ['CUDA_DEVICE'] = str(device_id)
            import pycuda.autoinit
            device_name = pycuda.autoinit.device.name()
            print(f'TensorRT initialized on device: {device_name}')
        except Exception as e:
            print(f'Warning: Failed to initialize TensorRT: {e}')
            self.use_tensorrt = False
    
    def _init_trt_modules(self, rife_model_path, sr_model_path):
        """Initialize RIFE and SR TensorRT modules"""
        try:
            import pycuda.driver as cuda
            
            # Create CUDA stream
            self.instream = cuda.Stream()
            
            # Create output memory compatible with RIFE input
            # [1, 512, 512, 4]
            output_shape = (1, 512, 512, 4)
            self.output_memory = self._create_host_device_mem(
                output_shape, np.uint8)
            
            # Initialize RIFE
            if self.use_interpolation and rife_model_path:
                from ezvtb_rt.rife import RIFE
                
                self.rife = RIFE(rife_model_path, self.instream,
                                self.output_memory)
                self.scale = self.rife.scale
                print(f"RIFE TensorRT module loaded: {rife_model_path}")
            
            # Initialize SR
            if self.use_sr and sr_model_path:
                from ezvtb_rt.sr import SR
                
                if self.rife is not None:
                    instream = self.rife.instream
                    mems = [self.rife.memories['framegen_'+str(i)]
                            for i in range(self.rife.scale)]
                else:
                    instream = self.instream
                    mems = [self.output_memory]
                
                self.sr = SR(sr_model_path, instream, mems)
                print(f"SR TensorRT module loaded: {sr_model_path}")
                
        except Exception as e:
            print(f'Warning: Failed to initialize TensorRT modules: {e}')
            self.rife = None
            self.sr = None
    
    def _create_host_device_mem(self, shape, dtype):
        """Create HostDeviceMem for TensorRT modules"""
        import pycuda.driver as cuda
        from ezvtb_rt.trt_utils import HostDeviceMem
        
        host_mem = cuda.pagelocked_empty(shape, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        return HostDeviceMem(host_mem, device_mem)
    
    def setImage(self, img: np.ndarray):
        """Set input character image - IGNORED for THA4"""
        # THA4 uses character image from YAML, ignore external image
        pass
    
    def inference(self, pose: np.ndarray) -> list:
        """
        Run full inference pipeline: THA4 -> RIFE (TRT) -> SR (TRT)
        
        Args:
            pose: numpy array of shape (1, 45)
                  
        Returns:
            List of output images [H, W, 4] in BGRA format, uint8
        """
        if self.input_image_tensor is None:
            raise RuntimeError("Image not set.")
        
        pose = pose.astype(np.float32)
        
        # Cache management
        need_cache_write = 0
        res_carrier = None
        
        # Step 1: THA4 inference (PyTorch) with optional caching
        if self.cacher is None:
            output_bgra = self._run_tha4_inference(pose)
            res_carrier = [output_bgra]
        else:
            hs = hash(str(pose))
            cached = self.cacher.read(hs)
            
            if cached is not None:  # Cache hit
                output_bgra = cached
                res_carrier = [cached]
                
                # If using TensorRT, upload to GPU
                if self.use_tensorrt and \
                   (self.rife is not None or self.sr is not None):
                    np.copyto(self.output_memory.host,
                             output_bgra.reshape(1, 512, 512, 4))
                    self.output_memory.htod(self.instream)
            else:  # Cache miss
                output_bgra = self._run_tha4_inference(pose)
                need_cache_write = hs
                res_carrier = [output_bgra]
        
        # Step 2: RIFE interpolation (TensorRT)
        if self.rife is not None:
            self.rife.inference()
            res_carrier = self.rife
        
        # Step 3: SR super resolution (TensorRT)
        if self.sr is not None:
            self.sr.inference()
            res_carrier = self.sr
        
        # Step 4: Fetch results
        if need_cache_write != 0:
            self.cacher.write(need_cache_write, output_bgra)
        
        # Get final results
        if type(res_carrier) is not list:
            res_carrier = res_carrier.fetchRes()
        
        return res_carrier
    
    def _run_tha4_inference(self, pose: np.ndarray) -> np.ndarray:
        """Run THA4 PyTorch inference and return BGRA numpy array"""
        # Extract pose components
        if self.use_eyebrow:
            eyebrow_pose = pose[:, :12]
        else:
            eyebrow_pose = np.zeros((1, 12), dtype=np.float32)
        
        face_pose = pose[:, 12:39]
        head_pose = pose[:, 39:45]
        
        # Convert to torch tensors
        eyebrow_tensor = torch.from_numpy(eyebrow_pose).to(self.device)
        face_tensor = torch.from_numpy(face_pose).to(self.device)
        head_tensor = torch.from_numpy(head_pose).to(self.device)
        
        eyebrow_c = eyebrow_pose[0].tolist()
        face_c = face_pose[0].tolist()
        
        # Run THA4 inference
        with torch.no_grad():
            output_tensor = self.wrapper.forward(
                self.input_image_tensor,
                face_tensor,
                head_tensor,
                eyebrow_tensor,
                face_c,
                eyebrow_c,
                ratio=None
            )
        
        # Post-process output
        output_image = output_tensor[0].float()
        output_image = torch.clamp((output_image + 1.0) / 2.0, 0.0, 1.0)
        output_image = convert_linear_to_srgb(output_image)
        output_np = output_image.permute(1, 2, 0).cpu().numpy()
        output_np = (output_np * 255.0).astype(np.uint8)
        output_bgra = cv2.cvtColor(output_np, cv2.COLOR_RGBA2BGRA)
        
        # Upload to GPU memory if TensorRT is active
        if self.use_tensorrt and \
           (self.rife is not None or self.sr is not None):
            np.copyto(self.output_memory.host,
                     output_bgra.reshape(1, 512, 512, 4))
            self.output_memory.htod(self.instream)
        
        return output_bgra


def get_tha4_core(device_id=0, use_eyebrow=True, yaml_path=None,
                  interpolation_scale=1, use_tensorrt=False,
                  use_interpolation=False, interpolation_half=True,
                  use_sr=False, sr_x4=True, sr_half=True, sr_noise=1,
                  cacher_quality=85, cacher_ram_size=2.0):
    """
    Factory function to create THA4 core
    
    Args:
        device_id: GPU device ID
        use_eyebrow: whether to use eyebrow control
        yaml_path: path to character_model.yaml
        interpolation_scale: number of frames for interpolation
        use_tensorrt: enable TensorRT acceleration for RIFE/SR
        use_interpolation: enable RIFE frame interpolation
        interpolation_half: use FP16 for RIFE
        use_sr: enable super resolution
        sr_x4: use 4x SR (else 2x)
        sr_half: use FP16 for SR
        sr_noise: noise level for waifu2x
        cacher_quality: cache compression quality
        cacher_ram_size: RAM cache size in GB
        
    Returns:
        THA4Core or THA4CoreWithAccel instance
    """
    # Use accelerated version if TensorRT is requested
    if use_tensorrt and (use_interpolation or use_sr):
        return THA4CoreWithAccel(
            device_id=device_id,
            use_eyebrow=use_eyebrow,
            yaml_path=yaml_path,
            use_tensorrt=use_tensorrt,
            use_interpolation=use_interpolation,
            interpolation_scale=interpolation_scale,
            interpolation_half=interpolation_half,
            use_sr=use_sr,
            sr_x4=sr_x4,
            sr_half=sr_half,
            sr_noise=sr_noise,
            cacher_quality=cacher_quality,
            cacher_ram_size=cacher_ram_size
        )
    else:
        # Use simple PyTorch-only version
        return THA4Core(
            device_id=device_id,
            use_eyebrow=use_eyebrow,
            yaml_path=yaml_path,
            interpolation_scale=interpolation_scale if use_interpolation
            else 1
        )
