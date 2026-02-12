"""
THA4 Adapter Module - Simplified Version
PyTorch only, no TensorRT/ONNX support
Adapts THA4 models to existing EasyVtuber architecture with THA3-compatible interface
Uses CharacterModel to properly load YAML + PNG + PT files
"""
import sys
import os
import torch

# Add tha4 source code path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tha4', 'src'))

from tha4.charmodel.character_model import CharacterModel


def convert_tha3_pose_to_tha4(eyebrow_vector, mouth_eye_vector, pose_vector):
    """
    Convert THA3 pose format to THA4 format
    
    THA3 inputs:
        - eyebrow_vector: [12] eyebrow parameters
        - mouth_eye_vector: [27] facial expression parameters
        - pose_vector: [6] head pose parameters
        
    THA4 input:
        - pose: [45] unified pose vector
          - [0:39] face pose (12 eyebrow + 27 face)
          - [39:45] body pose (6 head rotation)
    
    Returns:
        torch.Tensor [batch, 45]
    """
    # Concatenate: [eyebrow(12), mouth_eye(27), pose(6)] = [45]
    face_pose = torch.cat([eyebrow_vector, mouth_eye_vector], dim=1)
    tha4_pose = torch.cat([face_pose, pose_vector], dim=1)
    return tha4_pose


class THA4Wrapper:
    """
    THA4 Poser wrapper providing THA3 TalkingAnime3-compatible interface
    Uses CharacterModel to load YAML config + PNG image + PT models
    """
    
    def __init__(self, device: torch.device, yaml_path: str = None):
        """
        Initialize THA4 wrapper
        
        Args:
            device: torch device (cuda/cpu)
            yaml_path: path to character model YAML (required)
        """
        self.device = device
        
        # YAML path is required
        if yaml_path is None:
            raise ValueError(
                "THA4 requires a YAML config path. "
                "Please select a .yaml file in the Character selection."
            )
        
        # Check if YAML exists
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"THA4 character model YAML not found: {yaml_path}\n"
                f"Expected files in data/images/:\n"
                f"  - <character>.yaml (config with relative paths)\n"
                f"  - <character>.png (source image)\n"
                f"  - <character>_face_morpher.pt (face model)\n"
                f"  - <character>_body_morpher.pt (body model)"
            )
        
        # Load CharacterModel from YAML
        print(f"Loading THA4 CharacterModel from: {yaml_path}")
        self.character_model = CharacterModel.load(yaml_path)
        
        # Get poser (lazy loads .pt files)
        self.poser = self.character_model.get_poser(device)
        
        # Get character image tensor (loads .png file)
        char_img = self.character_model.get_character_image(device)
        
        # Add batch dimension if needed: [4, 512, 512] -> [1, 4, 512, 512]
        if len(char_img.shape) == 3:
            self.character_image = char_img.unsqueeze(0)
        else:
            self.character_image = char_img
        
        print(f"THA4 model loaded successfully:")
        print(f"  Device: {device}")
        print(f"  Image: {self.character_model.character_image_file_name}")
        print(f"  Face: {self.character_model.face_morpher_file_name}")
        print(f"  Body: {self.character_model.body_morpher_file_name}")
        
    def forward(self, image, mouth_eye_vector, pose_vector, eyebrow_vector,
                mouth_eye_vector_c, eyebrow_vector_c, ratio=None):
        """
        Forward inference, compatible with THA3 TalkingAnime3.forward()
        
        Args:
            image: [batch, 4, 512, 512]
            mouth_eye_vector: [batch, 27]
            pose_vector: [batch, 6]
            eyebrow_vector: [batch, 12]
            mouth_eye_vector_c: compressed version (not used by THA4)
            eyebrow_vector_c: compressed version (not used by THA4)
            ratio: GPU cache hit ratio (not used by THA4)
            
        Returns:
            output image [batch, 4, 512, 512]
        """
        # Convert pose format
        tha4_pose = convert_tha3_pose_to_tha4(
            eyebrow_vector, mouth_eye_vector, pose_vector
        )
        
        # Call THA4 poser
        with torch.no_grad():
            output = self.poser.pose(image, tha4_pose, output_index=0)
        
        return output
    
    def to(self, device: torch.device):
        """Move model to specified device"""
        self.device = device
        self.poser.to(device)
        return self
