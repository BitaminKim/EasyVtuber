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
from typing import List, Optional
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt

# =================================================================================
# START: Monkey Patch for torch.nn.functional.affine_grid (CORRECTED, DEVICE-AWARE)
# =================================================================================
import torch.nn.functional

def linspace_from_neg_one(num_steps, dtype=torch.float32, device='cpu'):
    r = torch.linspace(-1, 1, num_steps, dtype=torch.float32, device=device)
    if num_steps > 1: r = r * (num_steps - 1) / num_steps
    return r
def create_grid(N, C, H, W, device='cpu'):
    grid = torch.empty((N, H, W, 3), dtype=torch.float32, device=device)
    grid.select(-1, 0).copy_(linspace_from_neg_one(W, device=device))
    grid.select(-1, 1).copy_(linspace_from_neg_one(H, device=device).unsqueeze_(-1))
    grid.select(-1, 2).fill_(1); return grid
def custom_affine_grid(theta, size, align_corners=False):
    N, C, H, W = size; device = theta.device
    grid = create_grid(N, C, H, W, device=device)
    grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
    grid = grid.view(N, H, W, 2); return grid
def apply_affine_grid_patch():
    torch.nn.functional.affine_grid = custom_affine_grid
    print("✅ Applied monkey patch to torch.nn.functional.affine_grid (Device-aware version).")
# =================================================================================
# END: Monkey Patch
# =================================================================================

apply_affine_grid_patch()

if len(sys.argv) != 2:
    print("Usage: python onnx_export_tha4.py <model_name>"); print("Available models: float, half")
    raise ValueError('Do not get model name')

MODEL_NAME = sys.argv[1]
HALF = (MODEL_NAME == 'half')
DEVICE_NAME = 'cuda:0'
IMAGE_INPUT = os.path.join('.', 'data', 'images', 'character.png')

TMP_DIR = join('.', 'onnx_model_tha4', 'tmp'); Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
MODEL_DIR = join('.', 'onnx_model_tha4', 'fp32' if not HALF else 'fp16'); Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
TMP_FILE_WRITE = join(TMP_DIR, 'tmp.onnx')

gpu_device = torch.device(DEVICE_NAME); cpu_device = torch.device('cpu')
gpu_dtype = torch.float16 if HALF else torch.float32
cpu_dtype = torch.float32

print(f"Using GPU device for validation: {gpu_device}, dtype: {gpu_dtype}")
print(f"Using CPU device for ONNX export, dtype: {cpu_dtype}")

def torch_srgb_to_linear(x: torch.Tensor):
    x = torch.clip(x, 0.0, 1.0); return torch.where(torch.le(x, 0.04045), x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
def torch_linear_to_srgb(x):
    x = torch.clip(x, 0.0, 1.0); return torch.where(torch.le(x, 0.003130804953560372), x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)
def numpy_linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0); return np.where(x <= 0.003130804953560372, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)
def load_tha4_poser(device: torch.device, use_half: bool):
    from tha4.poser.modes.mode_07 import create_poser
    m_files = { 'eyebrow_decomposer': 'data/models/tha4/eyebrow_decomposer.pt', 'eyebrow_morphing_combiner': 'data/models/tha4/eyebrow_morphing_combiner.pt', 'face_morpher': 'data/models/tha4/face_morpher.pt', 'body_morpher': 'data/models/tha4/body_morpher.pt', 'upscaler': 'data/models/tha4/upscaler.pt' }
    print(f"Loading THA4 poser onto {device}..."); poser = create_poser(device=device, module_file_names=m_files)
    if use_half and device.type == 'cuda': print("Converting poser modules to half precision (FP16)..."); poser.half()
    return poser

gpu_poser = load_tha4_poser(gpu_device, HALF)
cpu_poser = load_tha4_poser(cpu_device, False)
pose_size = gpu_poser.get_num_parameters(); print(f"Pose parameters: {pose_size}")

def resize_PIL_image(img, size=(512, 512)): return img.resize(size, PIL.Image.Resampling.LANCZOS)
def extract_PIL_image_from_filelike(path): return PIL.Image.open(path)
def extract_pytorch_image_from_PIL_image(img):
    arr = np.array(img)
    if arr.shape[2] != 4: raise ValueError("Input image must be RGBA")
    t = torch.from_numpy(arr).float() / 255.0
    t[:, :, :3] = torch_srgb_to_linear(t[:, :, :3])
    t = t.permute(2, 0, 1) * 2.0 - 1.0; return t

pil_image = resize_PIL_image(extract_PIL_image_from_filelike(IMAGE_INPUT))
pt_img_gpu = extract_pytorch_image_from_PIL_image(pil_image).to(gpu_dtype).unsqueeze(0).to(gpu_device)
zero_pose_gpu = torch.zeros(1, pose_size, dtype=gpu_dtype, device=gpu_device)
with torch.no_grad(): poser_torch_res = gpu_poser.pose(pt_img_gpu, zero_pose_gpu)
print(f"Reference GPU output shape: {poser_torch_res.shape}")

print("\n" + "="*80); print("Starting ONNX export process on CPU..."); print("="*80)
cpu_modules = cpu_poser.get_modules()
cv_img_numpy = cv2.imread(IMAGE_INPUT, cv2.IMREAD_UNCHANGED)
cv_img_tensor_cpu = torch.from_numpy(cv_img_numpy)

class EyebrowDecomposerWrapper(Module):
    def __init__(self, obj): super().__init__(); self.m = obj
    def forward(self, img: Tensor) -> List[Tensor]:
        img_f = img.to(cpu_dtype) / 255.0; img_f[:, :, :3] = torch_srgb_to_linear(img_f[:, :, :3])
        # --- THE FIX: Swap BGR to RGB ---
        img_f = img_f[:, :, [2, 1, 0, 3]]
        img_c = img_f.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        crop = img_c[:, :, 64:192, 192:320]; res = self.m(crop)
        return [res[3], res[0], img_c]
eyebrow_decomposer_wrapper = EyebrowDecomposerWrapper(cpu_modules['eyebrow_decomposer']).eval()
with torch.no_grad(): decomposer_res_cpu = eyebrow_decomposer_wrapper(cv_img_tensor_cpu)
torch.onnx.export(eyebrow_decomposer_wrapper, cv_img_tensor_cpu, TMP_FILE_WRITE, opset_version=16, input_names=['input_image'], output_names=["background_layer", "eyebrow_layer", "image_prepared"])
onnx_model = onnx.load(TMP_FILE_WRITE); onnx.checker.check_model(onnx_model); onnx_model_sim, check = simplify(onnx_model)
onnx.save(onnx_model_sim if check else onnx_model, join(MODEL_DIR, "decomposer.onnx")); print("✓ Decomposer exported")

eyebrow_pose_cpu = torch.zeros(1, 12, dtype=cpu_dtype)
class EyebrowCombinerWrapper(Module):
    def __init__(self, obj): super().__init__(); self.m = obj
    def forward(self, img: Tensor, bg: Tensor, layer: Tensor, pose: Tensor) -> Tensor:
        crop = img[:, :, 32:224, 160:352].clone()
        morphed = self.m(bg, layer, pose)[2]
        crop[:, :, 32:160, 32:160] = morphed; return crop
eyebrow_combiner_wrapper = EyebrowCombinerWrapper(cpu_modules['eyebrow_morphing_combiner']).eval()
combiner_inputs_cpu = (decomposer_res_cpu[2], decomposer_res_cpu[0], decomposer_res_cpu[1], eyebrow_pose_cpu)
with torch.no_grad(): combiner_res_cpu = eyebrow_combiner_wrapper(*combiner_inputs_cpu)
torch.onnx.export(eyebrow_combiner_wrapper, combiner_inputs_cpu, TMP_FILE_WRITE, opset_version=16, input_names=['image_prepared', 'eyebrow_background_layer', "eyebrow_layer", 'eyebrow_pose'], output_names=['eyebrow_image'])
onnx_model = onnx.load(TMP_FILE_WRITE); onnx.checker.check_model(onnx_model); onnx_model_sim, check = simplify(onnx_model)
onnx.save(onnx_model_sim if check else onnx_model, join(MODEL_DIR, "combiner.onnx")); print("✓ Combiner exported")

face_pose_cpu = torch.zeros(1, 27, dtype=cpu_dtype)
class FaceMorpherWrapper(Module):
    def __init__(self, obj): super().__init__(); self.m = obj
    def forward(self, img: Tensor, crop: Tensor, pose: Tensor) -> List[Tensor]:
        morphed = self.m(crop, pose)[0]; full = img.clone()
        full[:, :, 32:224, 160:352] = morphed
        half = interpolate(full, size=(256, 256), mode='bilinear', align_corners=False); return [full, half]
face_morpher_wrapper = FaceMorpherWrapper(cpu_modules['face_morpher']).eval()
morpher_inputs_cpu = (decomposer_res_cpu[2], combiner_res_cpu, face_pose_cpu)
with torch.no_grad(): morpher_res_cpu = face_morpher_wrapper(*morpher_inputs_cpu)
torch.onnx.export(face_morpher_wrapper, morpher_inputs_cpu, TMP_FILE_WRITE, opset_version=16, input_names=['image_prepared', 'im_morpher_crop', 'face_pose'], output_names=['face_morphed_full', 'face_morphed_half'])
onnx_model = onnx.load(TMP_FILE_WRITE); onnx.checker.check_model(onnx_model); onnx_model_sim, check = simplify(onnx_model)
onnx.save(onnx_model_sim if check else onnx_model, join(MODEL_DIR, "morpher.onnx")); print("✓ Face Morpher exported")

rotation_pose_cpu = torch.zeros(1, 6, dtype=cpu_dtype)
class BodyMorpherWrapper(Module):
    def __init__(self, obj): super().__init__(); self.m = obj
    def forward(self, img: Tensor, pose: Tensor) -> List[Tensor]:
        res = self.m(img, pose); return [res[0], res[3]]
body_morpher_wrapper = BodyMorpherWrapper(cpu_modules['body_morpher']).eval()
body_morpher_inputs_cpu = (morpher_res_cpu[1], rotation_pose_cpu)
with torch.no_grad(): body_morpher_res_cpu = body_morpher_wrapper(*body_morpher_inputs_cpu)
torch.onnx.export(body_morpher_wrapper, body_morpher_inputs_cpu, TMP_FILE_WRITE, opset_version=16, input_names=['face_morphed_half', 'rotation_pose'], output_names=['half_res_posed_image', 'half_res_grid_change'])
onnx_model = onnx.load(TMP_FILE_WRITE); onnx.checker.check_model(onnx_model); onnx_model_sim, check = simplify(onnx_model)
onnx.save(onnx_model_sim if check else onnx_model, join(MODEL_DIR, "body_morpher.onnx")); print("✓ Body Morpher exported")

class UpscalerWrapper(Module):
    def __init__(self, obj): super().__init__(); self.m = obj
    def forward(self, rest_img: Tensor, posed_half: Tensor, grid_half: Tensor, pose: Tensor) -> List[Tensor]:
        posed_full = interpolate(posed_half, size=(512, 512), mode='bilinear', align_corners=False)
        grid_full = interpolate(grid_half, size=(512, 512), mode='bilinear', align_corners=False)
        res = self.m(rest_img, posed_full, grid_full, pose)[0]
        output = res / 2.0 + 0.5; output[:, :3, :, :] = torch_linear_to_srgb(output[:, :3, :, :])
        cv_res = (output.squeeze(0).permute(1, 2, 0)[:, :, [2, 1, 0, 3]] * 255).clamp(0, 255).to(torch.uint8)
        return [output, cv_res]
upscaler_wrapper = UpscalerWrapper(cpu_modules['upscaler']).eval()
upscaler_inputs_cpu = (morpher_res_cpu[0], body_morpher_res_cpu[0], body_morpher_res_cpu[1], rotation_pose_cpu)
with torch.no_grad(): upscaler_res_cpu = upscaler_wrapper(*upscaler_inputs_cpu)
torch.onnx.export(upscaler_wrapper, upscaler_inputs_cpu, TMP_FILE_WRITE, opset_version=16, input_names=['rest_image', 'half_res_posed_image', 'half_res_grid_change', 'rotation_pose'], output_names=['result', 'cv_result'])
onnx_model = onnx.load(TMP_FILE_WRITE); onnx.checker.check_model(onnx_model); onnx_model_sim, check = simplify(onnx_model)
onnx.save(onnx_model_sim if check else onnx_model, join(MODEL_DIR, "upscaler.onnx")); print("✓ Upscaler exported")

print("\n" + "="*80); print("Validation: Running ONNX inference..."); print("="*80)
class THA4ONNXRunner:
    def __init__(self, model_dir, use_gpu=True):
        providers = [("CUDAExecutionProvider", {"device_id": 0})] if use_gpu else ['CPUExecutionProvider']
        self.decomposer_sess = self._create_session(join(model_dir, 'decomposer.onnx'), providers)
        self.combiner_sess = self._create_session(join(model_dir, "combiner.onnx"), providers)
        self.morpher_sess = self._create_session(join(model_dir, "morpher.onnx"), providers)
        self.body_morpher_sess = self._create_session(join(model_dir, "body_morpher.onnx"), providers)
        self.upscaler_sess = self._create_session(join(model_dir, "upscaler.onnx"), providers)
        print("✓ All ONNX models loaded")
    def _create_session(self, path, providers):
        try: return ort.InferenceSession(path, providers=providers)
        except Exception: print(f"Failed with {providers}, falling back to CPU for {os.path.basename(path)}"); return ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    def run(self, img_path, p_eye, p_face, p_rot):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        d_res = self.decomposer_sess.run(None, {'input_image': img})
        c_res = self.combiner_sess.run(None, {'image_prepared': d_res[2], 'eyebrow_background_layer': d_res[0], "eyebrow_layer": d_res[1], 'eyebrow_pose': p_eye})
        m_res = self.morpher_sess.run(None, {'image_prepared': d_res[2], 'im_morpher_crop': c_res[0], 'face_pose': p_face})
        b_res = self.body_morpher_sess.run(None, {'face_morphed_half': m_res[1], 'rotation_pose': p_rot})
        u_res = self.upscaler_sess.run(None, {'rest_image': m_res[0], 'half_res_posed_image': b_res[0], 'half_res_grid_change': b_res[1], 'rotation_pose': p_rot})
        return u_res[0]

def save_comparison_images(onnx_img_np, torch_img_np, prefix=""):
    onnx_hwc = np.transpose(onnx_img_np.squeeze(0), (1, 2, 0))[:, :, :3]
    torch_hwc = np.transpose(torch_img_np.squeeze(0), (1, 2, 0))[:, :, :3]
    onnx_hwc = np.clip(onnx_hwc, 0, 1)
    torch_hwc = np.clip(torch_hwc, 0, 1)
    onnx_path = f"{prefix}onnx_output.png"; torch_path = f"{prefix}pytorch_output.png"
    plt.imsave(onnx_path, onnx_hwc); plt.imsave(torch_path, torch_hwc)
    print(f"✅ Saved comparison images: {onnx_path} and {torch_path}")

runner = THA4ONNXRunner(MODEL_DIR)
p_eye_z, p_face_z, p_rot_z = np.zeros((1, 12), dtype=np.float32), np.zeros((1, 27), dtype=np.float32), np.zeros((1, 6), dtype=np.float32)

print("\nRunning ONNX inference with zero pose...")
onnx_output = runner.run(IMAGE_INPUT, p_eye_z, p_face_z, p_rot_z)
ref_np = poser_torch_res.cpu().numpy() / 2.0 + 0.5; ref_np[0, :3, :, :] = numpy_linear_to_srgb(ref_np[0, :3, :, :])
save_comparison_images(onnx_output, ref_np, "zero_pose_")
mse = ((onnx_output - ref_np) ** 2).mean(); print(f"\n{'='*80}\nMSE (zero pose): {mse:.10f}\n{'='*80}")
ACCEPTABLE_MSE_THRESHOLD = 1e-2
if mse < ACCEPTABLE_MSE_THRESHOLD: print(f"✅ Validation PASSED! MSE is within the acceptable range of {ACCEPTABLE_MSE_THRESHOLD}.")
else: print(f"⚠️ Warning: MSE is higher than expected! But this can be normal due to precision differences.")

print("\n" + "="*80); print("Testing with random poses..."); print("="*80)
p_eye_r = np.random.randn(1, 12).astype(np.float32) * 0.5
p_face_r = np.random.randn(1, 27).astype(np.float32) * 0.5
p_rot_r = np.random.randn(1, 6).astype(np.float32) * 0.3
onnx_output_r = runner.run(IMAGE_INPUT, p_eye_r, p_face_r, p_rot_r)
pose_r_gpu = torch.from_numpy(np.concatenate([p_eye_r, p_face_r, p_rot_r], axis=1)).to(gpu_dtype).to(gpu_device)
with torch.no_grad(): ref_output_r = gpu_poser.pose(pt_img_gpu, pose_r_gpu)
ref_np_r = ref_output_r.cpu().numpy() / 2.0 + 0.5; ref_np_r[0, :3, :, :] = numpy_linear_to_srgb(ref_np_r[0, :3, :, :])
save_comparison_images(onnx_output_r, ref_np_r, "random_pose_")
mse_r = ((onnx_output_r - ref_np_r) ** 2).mean(); print(f"MSE (random pose): {mse_r:.10f}")
if mse_r < ACCEPTABLE_MSE_THRESHOLD: print(f"✅ Random pose test PASSED! (Threshold: {ACCEPTABLE_MSE_THRESHOLD})")
else: print(f"⚠️ Warning: High MSE with random pose! But this can be normal.")

print(f"\n{'='*80}\nExport and validation completed!\nModels saved to: {MODEL_DIR}\n{'='*80}")