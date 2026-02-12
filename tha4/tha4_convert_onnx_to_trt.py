"""
将 ONNX 模型转换为 fp32 和 fp16 精度的 TensorRT 模型
并保存到对应的 fp32 和 fp16 目录
"""
import os
import sys
from pathlib import Path

# 添加 ezvtuber-rt 到路径
sys.path.insert(0, str(Path(__file__).parent / 'ezvtuber-rt'))

from ezvtb_rt.trt_utils import build_engine, save_engine

def convert_onnx_to_trt():
    # 基础目录
    base_dir = Path('data/models/tha4')
    onnx_dir = base_dir / 'fp32'
    
    # 检查 ONNX 文件目录是否存在
    if not onnx_dir.exists():
        print(f"错误: ONNX 模型目录不存在: {onnx_dir}")
        return
    
    # 指定要转换的文件名列表
    target_names = ['body_morpher.onnx', 'combiner.onnx', 'decomposer.onnx', 
                    'morpher.onnx', 'upscaler.onnx']
    
    # 获取指定的 ONNX 文件
    onnx_files = [onnx_dir / name for name in target_names if (onnx_dir / name).exists()]
    
    if not onnx_files:
        print(f"在 {onnx_dir} 中未找到指定的 ONNX 模型文件")
        print(f"需要的文件: {target_names}")
        return
    
    print(f"找到 {len(onnx_files)} 个 ONNX 模型文件")
    print("开始转换为 fp32 和 fp16 精度的 TensorRT 模型...\n")
    
    # 定义要转换的精度类型
    precisions = ['fp32', 'fp16']
    
    for precision in precisions:
        print(f"\n{'='*60}")
        print(f"开始转换 {precision.upper()} 精度模型")
        print(f"{'='*60}\n")
        
        # 设置输出目录
        output_dir = base_dir / precision
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for onnx_file in onnx_files:
            trt_filename = onnx_file.stem + '.trt'
            trt_path = output_dir / trt_filename
            
            # 检查是否已存在转换好的 TRT 文件
            if trt_path.exists():
                print(f"⊙ 跳过 ({precision}): {onnx_file.name} (TRT文件已存在)")
                continue
            
            print(f"正在转换 ({precision}): {onnx_file.name}")
            
            try:
                # 构建并保存 TensorRT 引擎
                engine = build_engine(str(onnx_file), precision)
                save_engine(engine, str(trt_path))
                print(f"✓ 成功转换 ({precision}): {onnx_file.name} -> {trt_filename}\n")
            except Exception as e:
                print(f"✗ 转换失败 ({precision}): {onnx_file.name}")
                print(f"  错误信息: {str(e)}\n")
    
    print("\n" + "="*60)
    print("全部转换完成！")
    print("="*60)

if __name__ == '__main__':
    convert_onnx_to_trt()
