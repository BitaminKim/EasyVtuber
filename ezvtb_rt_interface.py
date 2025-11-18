import os 
import sys
import subprocess
import urllib.request
import shutil

dir_path = os.path.dirname(os.path.realpath(__file__))
ezvtb_path = os.path.join(dir_path, 'ezvtuber-rt')
ezvtb_main_path = os.path.join(dir_path, 'ezvtuber-rt-main')

project_path = ''
if os.path.exists(ezvtb_path):
    project_path = ezvtb_path
else:
    project_path = ezvtb_main_path

if project_path not in sys.path:
    sys.path.append(project_path)

def get_core(
        #Device setting
        device_id:int = 0, 
        use_tensorrt:bool = True, 
        #THA3 model setting
        model_seperable:bool = True,
        model_half:bool = True, #If using directml+half, there is small numerical error on Nvidia, and huge numerical error on AMD
        model_cache_size:float = 1.0, #unit of GigaBytes, only works for tensorrt
        model_use_eyebrow:bool = True,
        #RIFE interpolation setting
        use_interpolation:bool = True,
        interpolation_scale:int = 2,
        interpolation_half:bool = True, #If using directml+half, there is small numerical error on Nvidia, and huge numerical error on AMD
        #Cacher setting
        cacher_quality:int = 85,
        cacher_ram_size:float = 2.0,#unit of GigaBytes
        #SR setting
        use_sr:bool = False,
        sr_x4:bool = True,
        sr_half:bool = True,
        sr_noise:int = 1,
        #THA4 model setting (NEW)
        use_tha4:bool = False  # Use THA4 instead of THA3
        ):
    support_trt = False
    if use_tensorrt:
        print('Verifying TensorRT')
        try:
            from ezvtb_rt.trt_utils import cudaSetDevice
            cudaSetDevice(device_id)
            support_trt = True
        except Exception:
            support_trt = False
    if not support_trt and use_tensorrt:
        print('TensorRT option selected but not supported')
        use_tensorrt = False

    if use_tensorrt:
        os.environ['CUDA_DEVICE'] = str(device_id)
        import pycuda.autoinit
        print(f'Using device {pycuda.autoinit.device.name()}')

    # Determine model directory based on THA version
    if use_tha4:
        # THA4: Use data/models/tha4 directory with precision selection
        tha_model_dir = os.path.join(
            '.', 'data', 'models', 'tha4',
            'fp16' if model_half else 'fp32')
        print(f'THA4 Mode - Model Path: {tha_model_dir}')
    else:
        # THA3: Use data/models/tha3 directory
        tha_model_dir = os.path.join(
            '.', 'data', 'models', 'tha3',
            'seperable' if model_seperable else 'standard',
            'fp16' if model_half else 'fp32')
        print(f'THA3 Mode - Model Path: {tha_model_dir}')

    rife_model_path = ''
    if use_interpolation:
        rife_model_path = os.path.join(
            '.', 'data', 'models', 'rife_512',
            f'x{interpolation_scale}',
            'fp16' if interpolation_half else 'fp32')

    sr_model_path = ''
    if use_sr:
        if sr_x4:
            if sr_half:
                sr_model_path = os.path.join(
                    '.', 'data', 'models', 'Real-ESRGAN',
                    'exported_256_fp16')
            else:
                sr_model_path = os.path.join(
                    '.', 'data', 'models', 'Real-ESRGAN',
                    'exported_256')
        else:
            if sr_half:
                sr_model_path = os.path.join(
                    '.', 'data', 'models', 'waifu2x_upconv',
                    'fp16', 'upconv_7', 'art',
                    f'noise{sr_noise}_scale2x')
            else:
                sr_model_path = os.path.join(
                    '.', 'data', 'models', 'waifu2x_upconv',
                    'fp32', 'upconv_7', 'art',
                    f'noise{sr_noise}_scale2x')

    print(f'RIFE Path: {rife_model_path}')
    print(f'SR Path: {sr_model_path}')

    core = None
    if use_tensorrt:
        if use_tha4:
            from ezvtb_rt.core_tha4 import CoreTHA4TRT
            core = CoreTHA4TRT(
                tha_model_dir,
                vram_cache_size=model_cache_size,
                use_eyebrow=model_use_eyebrow,
                rife_dir=rife_model_path if rife_model_path else None,
                sr_dir=sr_model_path if sr_model_path else None,
                cache_max_volume=cacher_ram_size,
                cache_quality=cacher_quality)
        else:
            from ezvtb_rt.core import CoreTRT
            core = CoreTRT(
                tha_model_dir,
                vram_cache_size=model_cache_size,
                use_eyebrow=model_use_eyebrow,
                rife_dir=rife_model_path if rife_model_path else None,
                sr_dir=sr_model_path if sr_model_path else None,
                cache_max_volume=cacher_ram_size,
                cache_quality=cacher_quality)
    else:
        if use_tha4:
            from ezvtb_rt.core_tha4_ort import CoreTHA4ORT
            core = CoreTHA4ORT(
                tha_model_dir,
                rife_path=rife_model_path if rife_model_path else None,
                sr_path=sr_model_path if sr_model_path else None,
                device_id=device_id,
                cache_max_volume=cacher_ram_size,
                cache_quality=cacher_quality,
                use_eyebrow=model_use_eyebrow)
        else:
            from ezvtb_rt.core_ort import CoreORT
            core = CoreORT(
                tha_model_dir,
                rife_path=rife_model_path if rife_model_path else None,
                sr_path=sr_model_path if sr_model_path else None,
                device_id=device_id,
                cache_max_volume=cacher_ram_size,
                cache_quality=cacher_quality,
                use_eyebrow=model_use_eyebrow)

    return core
    
if __name__ == '__main__':
    from ezvtb_rt.trt_utils import check_build_all_models, cudaSetDevice
    device_id = 0
    import sys
    if len(sys.argv) > 1:
        device_id = int(sys.argv[1])
    try:
        cudaSetDevice(device_id)
        check_build_all_models('./data/models')
    except:
        pass
