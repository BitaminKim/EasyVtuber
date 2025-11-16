"""
将 onnx_model_tha4/fp32 目录中的 ONNX 模型转换为 fp16 精度的 TensorRT 模型
并保存到 onnx_model_tha4/fp16 目录
"""
import os
import sys
from pathlib import Path

# 添加 ezvtuber-rt 到路径
sys.path.insert(0, str(Path(__file__).parent / 'ezvtuber-rt'))

from ezvtb_rt.trt_utils import build_engine, save_engine

def convert_onnx_to_trt():
    # 输入和输出目录
    input_dir = Path('onnx_model_tha4/fp32')
    output_dir = Path('onnx_model_tha4/fp16')
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 ONNX 文件
    onnx_files = list(input_dir.glob('*.onnx'))
    
    if not onnx_files:
        print(f"在 {input_dir} 中未找到 ONNX 模型文件")
        return
    
    print(f"找到 {len(onnx_files)} 个 ONNX 模型文件")
    print("开始转换为 fp16 精度的 TensorRT 模型...\n")
    
    for onnx_file in onnx_files:
        print(f"正在转换: {onnx_file.name}")
        
        # 构建输出文件路径
        trt_filename = onnx_file.stem + '.trt'
        trt_path = output_dir / trt_filename
        
        try:
            # 构建并保存 TensorRT 引擎
            engine = build_engine(str(onnx_file), 'fp16')
            save_engine(engine, str(trt_path))
            print(f"✓ 成功转换: {onnx_file.name} -> {trt_path}\n")
        except Exception as e:
            print(f"✗ 转换失败: {onnx_file.name}")
            print(f"  错误信息: {str(e)}\n")
    
    print("转换完成！")

if __name__ == '__main__':
    convert_onnx_to_trt()
