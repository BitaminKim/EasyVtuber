"""
THA4 Core - Pure PyTorch implementation
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


class THA4Core:
    """
    THA4 Core class with interface compatible with CoreTRT/CoreORT
    
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
                      Default: 'data/models/tha4/character_model.yaml'
            interpolation_scale: number of output frames to generate
                      (THA4 doesn't support real interpolation, will duplicate frames)
        """
        device_str = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
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
        print(f"Using character image from model (ignoring setImage calls)")
        if interpolation_scale > 1:
            print(f"Note: THA4 doesn't support interpolation, "
                  f"will duplicate frames {interpolation_scale}x")
    
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
            List of output images as numpy arrays [H, W, 4] in BGRA format, uint8
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
        # We need to create dummy compressed versions for interface compatibility
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
        # THA4 doesn't support real interpolation, so duplicate frames if needed
        return [output_bgra] * self.interpolation_scale


def get_tha4_core(device_id=0, use_eyebrow=True, yaml_path=None,
                  interpolation_scale=1):
    """
    Factory function to create THA4 core
    
    Args:
        device_id: GPU device ID
        use_eyebrow: whether to use eyebrow control
        yaml_path: path to character_model.yaml
                  Default: 'data/models/tha4/character_model.yaml'
        interpolation_scale: number of frames (for compatibility with interpolation)
        
    Returns:
        THA4Core instance
    """
    return THA4Core(device_id=device_id,
                    use_eyebrow=use_eyebrow,
                    yaml_path=yaml_path,
                    interpolation_scale=interpolation_scale)
