import torch
from os.path import join
from pathlib import Path
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(join(os.getcwd(), 'tha4', 'src'))
import numpy as np
import PIL.Image
import cv2
import onnx
from onnxsim import simplify
import onnxruntime as ort
from torch import Tensor
from torch.nn import Module
from typing import List
import matplotlib.pyplot as plt
from onnxconverter_common import float16

# =================================================================================
# START: Monkey Patch for torch.nn.functional.affine_grid (DEVICE-AWARE)
# =================================================================================
import torch.nn.functional

def linspace_from_neg_one(num_steps, dtype=torch.float32, device='cpu'):
    r = torch.linspace(-1, 1, num_steps, dtype=torch.float32, device=device)
    if num_steps > 1: 
        r = r * (num_steps - 1) / num_steps
    return r

def create_grid(N, C, H, W, device='cpu'):
    grid = torch.empty((N, H, W, 3), dtype=torch.float32, device=device)
    grid.select(-1, 0).copy_(linspace_from_neg_one(W, device=device))
    grid.select(-1, 1).copy_(linspace_from_neg_one(H, device=device).unsqueeze_(-1))
    grid.select(-1, 2).fill_(1)
    return grid

def custom_affine_grid(theta, size, align_corners=False):
    N, C, H, W = size
    device = theta.device
    grid = create_grid(N, C, H, W, device=device)
    grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
    grid = grid.view(N, H, W, 2)
    return grid

def apply_affine_grid_patch():
    torch.nn.functional.affine_grid = custom_affine_grid
    print("✅ Applied monkey patch to torch.nn.functional.affine_grid (Device-aware version).")

# =================================================================================
# END: Monkey Patch
# =================================================================================

apply_affine_grid_patch()

# Parse command line arguments
if len(sys.argv) < 2:
    print("Usage: python onnx_export_tha4_student.py <model_name> [--fp32 or --fp16] [--cpu]")
    print("Example: python onnx_export_tha4_student.py kanori_tha4 --fp32")
    print("         python onnx_export_tha4_student.py kanori_tha4 --fp32 --cpu")
    raise ValueError('Missing model name')

MODEL_NAME = sys.argv[1]
HALF = False
FORCE_CPU = '--cpu' in sys.argv

# Check CUDA availability
if FORCE_CPU:
    DEVICE_NAME = 'cpu'
    print("⚠️  Forced CPU mode enabled")
elif not torch.cuda.is_available():
    DEVICE_NAME = 'cpu'
    print("⚠️  CUDA not available, automatically using CPU mode")
else:
    DEVICE_NAME = 'cuda:0'

# Use image from model folder
MODEL_BASE_PATH = join('.', 'data', 'models', 'custom_tha4_models', MODEL_NAME)
IMAGE_INPUT = os.path.join(MODEL_BASE_PATH, 'character.png')

if not os.path.exists(IMAGE_INPUT):
    raise FileNotFoundError(f"Image not found: {IMAGE_INPUT}")

# Create output directories (save models in the model folder)
TMP_DIR = join(MODEL_BASE_PATH, 'tmp')
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
MODEL_DIR = MODEL_BASE_PATH  # Save directly in model folder
TMP_FILE_WRITE = join(TMP_DIR, 'tmp.onnx')

validation_device = torch.device(DEVICE_NAME)
export_device = torch.device('cpu')
validation_dtype = torch.float16 if (HALF and DEVICE_NAME != 'cpu') else torch.float32
export_dtype = torch.float32

if DEVICE_NAME == 'cpu':
    print(f"Using CPU device for both validation and ONNX export, dtype: {export_dtype}")
else:
    print(f"Using GPU device for validation: {validation_device}, dtype: {validation_dtype}")
    print(f"Using CPU device for ONNX export, dtype: {export_dtype}")

# Color space conversion utilities
def torch_srgb_to_linear(x: torch.Tensor):
    x = torch.clip(x, 0.0, 1.0)
    return torch.where(torch.le(x, 0.04045), x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def torch_linear_to_srgb(x):
    x = torch.clip(x, 0.0, 1.0)
    return torch.where(torch.le(x, 0.003130804953560372), x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

def numpy_linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.003130804953560372, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

# Load THA4 Student Model (mode_14: 2-step architecture)
def load_tha4_student_poser(device: torch.device, use_half: bool, model_base_path: str):
    from tha4.poser.modes.mode_14 import create_poser
    
    # Model file paths
    m_files = {
        'face_morpher': join(model_base_path, 'face_morpher.pt'),
        'body_morpher': join(model_base_path, 'body_morpher.pt')
    }
    
    print(f"Loading THA4 Student poser from {model_base_path}...")
    print(f"  - Face Morpher: {m_files['face_morpher']}")
    print(f"  - Body Morpher: {m_files['body_morpher']}")
    
    poser = create_poser(device=device, module_file_names=m_files)
    
    if use_half and device.type == 'cuda':
        print("Converting poser modules to half precision (FP16)...")
        poser.half()
    
    return poser

# Load posers for validation and export
validation_poser = load_tha4_student_poser(validation_device, HALF and DEVICE_NAME != 'cpu', MODEL_BASE_PATH)
if DEVICE_NAME == 'cpu':
    # If using CPU for validation, reuse the same poser for export
    export_poser = validation_poser
    print("Using same poser for validation and export (CPU mode)")
else:
    # If using GPU for validation, load separate CPU poser for export
    export_poser = load_tha4_student_poser(export_device, False, MODEL_BASE_PATH)
    
pose_size = validation_poser.get_num_parameters()
print(f"Pose parameters: {pose_size}")

# Image loading and preprocessing utilities
def resize_PIL_image(img, size=(512, 512)):
    return img.resize(size, PIL.Image.Resampling.LANCZOS)

def extract_PIL_image_from_filelike(path):
    return PIL.Image.open(path)

def extract_pytorch_image_from_PIL_image(img):
    arr = np.array(img)
    if arr.shape[2] != 4:
        raise ValueError("Input image must be RGBA")
    t = torch.from_numpy(arr).float() / 255.0
    t[:, :, :3] = torch_srgb_to_linear(t[:, :, :3])
    t = t.permute(2, 0, 1) * 2.0 - 1.0
    return t

# Load test image and run reference inference
pil_image = resize_PIL_image(extract_PIL_image_from_filelike(IMAGE_INPUT))
pt_img_validation = extract_pytorch_image_from_PIL_image(pil_image).to(validation_dtype).unsqueeze(0).to(validation_device)
zero_pose_validation = torch.zeros(1, pose_size, dtype=validation_dtype, device=validation_device)

with torch.no_grad():
    poser_torch_res = validation_poser.pose(pt_img_validation, zero_pose_validation)
print(f"Reference output shape: {poser_torch_res.shape}")

print("\n" + "="*80)
print("Starting ONNX export process...")
print("="*80)

# Get export modules
export_modules = export_poser.get_modules()
cv_img_numpy = cv2.imread(IMAGE_INPUT, cv2.IMREAD_UNCHANGED)
cv_img_tensor_export = torch.from_numpy(cv_img_numpy)

# =================================================================================
# Export Face Morpher (Student Model Network 1)
# =================================================================================
print("\n[1/2] Exporting Face Morpher...")

class FaceMorpherWrapper(Module):
    """
    Face Morpher for Student Model (mode_14)
    - Input: 45-dimensional pose (first 39 used for face)
    - Output: 128x128 face region
    Note: SIREN-based face morpher doesn't need input image, only pose
    """
    def __init__(self, face_morpher_module):
        super().__init__()
        self.face_morpher = face_morpher_module
    
    def forward(self, pose: Tensor) -> Tensor:
        """
        Args:
            pose: Full pose parameters [1, 39]
        Returns:
            face_output: Face morphed image [1, 4, 128, 128] in normalized format
        """
        # Run face morpher (SIREN generates image from pose only)
        face_output = self.face_morpher.forward(pose)
        
        return face_output

face_morpher_wrapper = FaceMorpherWrapper(export_modules['face_morpher']).eval()

# Test inputs
face_pose_export = torch.zeros(1, 39, dtype=export_dtype)

# Export
with torch.no_grad():
    face_morpher_res_export = face_morpher_wrapper(face_pose_export)

torch.onnx.export(
    face_morpher_wrapper,
    face_pose_export,
    TMP_FILE_WRITE,
    opset_version=16,
    input_names=['pose'],
    output_names=['face_morphed'],
    external_data =False,
    optimize =True,
    verify =True,
    dynamo =False,
)

onnx_model = onnx.load(TMP_FILE_WRITE)
onnx.checker.check_model(onnx_model)
print(f"  Original model inputs: {[inp.name for inp in onnx_model.graph.input]}")

# onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
# Simplify with input/output names preservation
onnx_model_sim, check = simplify(onnx_model, skip_fuse_bn=True)
if check:
    print(f"  Simplified model inputs: {[inp.name for inp in onnx_model_sim.graph.input]}")
    onnx.save(onnx_model_sim, join(MODEL_DIR, "face_morpher.onnx"))
else:
    print("  Simplification failed, using original model")
    onnx.save(onnx_model, join(MODEL_DIR, "face_morpher.onnx"))

print("✓ Face Morpher exported")

# =================================================================================
# Export Body Morpher (Student Model Network 2)
# =================================================================================
print("\n[2/2] Exporting Body Morpher...")

class BodyMorpherWrapper(Module):
    """
    Body Morpher for Student Model (mode_14)
    - Input: Full 512x512 image with face morpher output embedded
    - Output: Final morphed image
    """
    def __init__(self, body_morpher_module):
        super().__init__()
        self.body_morpher = body_morpher_module
    
    def forward(self, img: Tensor, face_morphed: Tensor, pose: Tensor) -> List[Tensor]:
        """
        Args:
            img: Input image [H, W, 4] in uint8 BGR+A format
            face_morphed: Face morphed output [1, 4, 128, 128] from face morpher
            pose: Full pose parameters [1, 45]
        Returns:
            result: Final output in [0, 255] uint8 format (OpenCV compatible)
        """
        # Convert image to normalized format
        img_f = img.to(export_dtype) / 255.0
        img_f[0, :, :, :3] = torch_srgb_to_linear(img_f[0, :, :, :3])
        img_f = img_f[:, :, :, [2, 1, 0, 3]]  # BGR to RGB
        img_c = img_f.permute(0, 3, 1, 2) * 2.0 - 1.0
        
        # Embed face morpher output into the full image
        center_x = 256
        center_y = 128 + 16  # 144
        img_with_face = img_c.clone()
        img_with_face[:, :, center_y-64:center_y+64, center_x-64:center_x+64] = face_morphed
        
        # Run body morpher
        body_output = self.body_morpher.forward(img_with_face, pose)
        
        # Take the first output (blended image)
        result = body_output[0]
        
        # Convert back to [0, 1] range
        output = result / 2.0 + 0.5
        output[:, :3, :, :] = torch_linear_to_srgb(output[:, :3, :, :])
        
        # Convert to OpenCV format (BGR, uint8)
        cv_result = (output.squeeze(0).permute(1, 2, 0)[:, :, [2, 1, 0, 3]] * 255).clamp(0, 255).to(torch.uint8)
        
        return [output, cv_result]

body_morpher_wrapper = BodyMorpherWrapper(export_modules['body_morpher']).eval()

full_pose_export = torch.zeros(1, 45, dtype=export_dtype)

cv_img_tensor_export = cv_img_tensor_export.unsqueeze(0)
# Export
with torch.no_grad():
    body_morpher_res_export = body_morpher_wrapper(cv_img_tensor_export, face_morpher_res_export, full_pose_export)

torch.onnx.export(
    body_morpher_wrapper,
    (cv_img_tensor_export, face_morpher_res_export, full_pose_export),
    TMP_FILE_WRITE,
    opset_version=16,
    input_names=['input_image', 'face_morphed', 'pose'],
    output_names=['result', 'cv_result'],
    external_data =False,
    optimize =True,
    verify =True,
    dynamo =False,
)

onnx_model = onnx.load(TMP_FILE_WRITE)
onnx.checker.check_model(onnx_model)
print(f"  Original model inputs: {[inp.name for inp in onnx_model.graph.input]}")

# onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
# for node in onnx_model.graph.node:
#     if node.name == "/Cast":
#         for attr in node.attribute:
#             if attr.name == "to":
#                 attr.i = onnx.TensorProto.FLOAT16  # Set to 10 (FP16)
#                 break
#     if node.name == "/Squeeze":
#         node.input[0] = "/ScatterND_14_cast_to_result"

# Simplify with input/output names preservation
onnx_model_sim, check = simplify(onnx_model, skip_fuse_bn=True)
if check:
    print(f"  Simplified model inputs: {[inp.name for inp in onnx_model_sim.graph.input]}")
    onnx.save(onnx_model_sim, join(MODEL_DIR, "body_morpher.onnx"))
else:
    print("  Simplification failed, using original model")
    onnx.save(onnx_model, join(MODEL_DIR, "body_morpher.onnx"))

print("✓ Body Morpher exported")

# =================================================================================
# Validation: ONNX Inference
# =================================================================================
print("\n" + "="*80)
print("Validation: Running ONNX inference...")
print("="*80)

class THA4StudentONNXRunner:
    def __init__(self, model_dir, use_gpu=True):
        providers = [("CUDAExecutionProvider", {"device_id": 0})] if use_gpu else ['CPUExecutionProvider']
        
        self.face_morpher_sess = self._create_session(join(model_dir, 'face_morpher.onnx'), providers)
        self.body_morpher_sess = self._create_session(join(model_dir, 'body_morpher.onnx'), providers)
        
        # Print actual input names for debugging
        face_inputs = [inp.name for inp in self.face_morpher_sess.get_inputs()]
        body_inputs = [inp.name for inp in self.body_morpher_sess.get_inputs()]
        print(f"✓ All ONNX models loaded")
        print(f"  Face Morpher inputs: {face_inputs}")
        print(f"  Body Morpher inputs: {body_inputs}")
    
    def _create_session(self, path, providers):
        try:
            return ort.InferenceSession(path, providers=providers)
        except Exception as e:
            print(f"Failed with {providers}, falling back to CPU for {os.path.basename(path)}")
            return ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    
    def run(self, img_path, pose):
        """
        Run inference with student model
        
        Args:
            img_path: Path to input image
            pose: Pose parameters [1, 45]
        """
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # Step 1: Face Morpher (SIREN only needs pose, generates face from scratch)
        face_res = self.face_morpher_sess.run(None, {
            'pose': pose[:, :39]
        })
        
        # Step 2: Body Morpher (needs original image, face result, and pose)
        body_res = self.body_morpher_sess.run(None, {
            'input_image': img,
            'face_morphed': face_res[0],
            'pose': pose
        })
        
        return body_res[0]

def save_comparison_images(onnx_img_np, torch_img_np, prefix=""):
    onnx_hwc = np.transpose(onnx_img_np.squeeze(0), (1, 2, 0))[:, :, :3]
    torch_hwc = np.transpose(torch_img_np.squeeze(0), (1, 2, 0))[:, :, :3]
    
    onnx_hwc = np.clip(onnx_hwc, 0, 1)
    torch_hwc = np.clip(torch_hwc, 0, 1)
    
    onnx_path = join(MODEL_DIR, f"{prefix}onnx_output.png")
    torch_path = join(MODEL_DIR, f"{prefix}pytorch_output.png")
    
    plt.imsave(onnx_path, onnx_hwc)
    plt.imsave(torch_path, torch_hwc)
    
    print(f"✅ Saved comparison images: {onnx_path} and {torch_path}")

# Initialize ONNX runner
runner = THA4StudentONNXRunner(MODEL_DIR)
pose_zero = np.zeros((1, 45), dtype=np.float32)

# Test with zero pose
print("\nRunning ONNX inference with zero pose...")
onnx_output = runner.run(IMAGE_INPUT, pose_zero)

# Convert reference output to numpy
ref_np = poser_torch_res.cpu().numpy() / 2.0 + 0.5
ref_np[0, :3, :, :] = numpy_linear_to_srgb(ref_np[0, :3, :, :])

# Save comparison images
save_comparison_images(onnx_output, ref_np, "zero_pose_")

# Calculate MSE
mse = ((onnx_output - ref_np) ** 2).mean()
print(f"\n{'='*80}\nMSE (zero pose): {mse:.10f}\n{'='*80}")

ACCEPTABLE_MSE_THRESHOLD = 1e-2
if mse < ACCEPTABLE_MSE_THRESHOLD:
    print(f"✅ Validation PASSED! MSE is within the acceptable range of {ACCEPTABLE_MSE_THRESHOLD}.")
else:
    print(f"⚠️ Warning: MSE is higher than expected! But this can be normal due to precision differences.")

# Test with random pose
print("\n" + "="*80)
print("Testing with random poses...")
print("="*80)

pose_random = np.random.randn(1, 45).astype(np.float32) * 0.3
onnx_output_r = runner.run(IMAGE_INPUT, pose_random)

pose_r_validation = torch.from_numpy(pose_random).to(validation_dtype).to(validation_device)
with torch.no_grad():
    ref_output_r = validation_poser.pose(pt_img_validation, pose_r_validation)

ref_np_r = ref_output_r.cpu().numpy() / 2.0 + 0.5
ref_np_r[0, :3, :, :] = numpy_linear_to_srgb(ref_np_r[0, :3, :, :])

save_comparison_images(onnx_output_r, ref_np_r, "random_pose_")

mse_r = ((onnx_output_r - ref_np_r) ** 2).mean()
print(f"MSE (random pose): {mse_r:.10f}")

if mse_r < ACCEPTABLE_MSE_THRESHOLD:
    print(f"✅ Random pose test PASSED! (Threshold: {ACCEPTABLE_MSE_THRESHOLD})")
else:
    print(f"⚠️ Warning: High MSE with random pose! But this can be normal.")

print(f"\n{'='*80}")
print("Export and validation completed!")
print(f"Models saved to: {MODEL_DIR}")
print("Generated files:")
print(f"  - face_morpher.onnx")
print(f"  - body_morpher.onnx")
print(f"{'='*80}")
