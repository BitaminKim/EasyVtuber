"""
将学生模型 ONNX 转换为 fp32 和 fp16 精度的 TensorRT 模型
学生模型（mode_14）只有 2 个网络：face_morpher 和 body_morpher
"""
import os
import sys
from pathlib import Path

# 添加 ezvtuber-rt 到路径
sys.path.insert(0, str(Path(__file__).parent / 'ezvtuber-rt'))

from ezvtb_rt.trt_utils import save_engine
import tensorrt as trt


# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine_with_dynamic_shape(onnx_file_path: str, precision: str):
    """
    构建支持动态输入形状的 TensorRT 引擎
    专门用于 body_morpher，它有动态的图像输入
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse model file
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Loading ONNX file from path {onnx_file_path}...')
    with open(onnx_file_path, 'rb') as model:
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Beginning ONNX file parsing')
        parse_res = parser.parse(model.read())
        if not parse_res:
            for error in range(parser.num_errors):
                TRT_LOGGER.log(TRT_LOGGER.ERROR, parser.get_error(error))
            raise ValueError('Failed to parse the ONNX file.')
    
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed parsing of ONNX file')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Input number: {network.num_inputs}')
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Output number: {network.num_outputs}')
    
    # 检查是否有动态输入
    has_dynamic_shape = False
    dynamic_inputs = []
    
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_shape = input_tensor.shape
        input_name = input_tensor.name
        TRT_LOGGER.log(TRT_LOGGER.INFO, f'Input {i} ({input_name}): shape={input_shape}')
        # 检查是否有 -1（动态维度）
        if any(dim == -1 for dim in input_shape):
            has_dynamic_shape = True
            dynamic_inputs.append((input_name, input_shape))
    
    # 设置优化配置文件（如果有动态输入）
    if has_dynamic_shape:
        TRT_LOGGER.log(TRT_LOGGER.INFO, 'Detected dynamic shapes, adding optimization profile...')
        profile = builder.create_optimization_profile()
        
        for input_name, input_shape in dynamic_inputs:
            input_shape_list = list(input_shape)
            
            # 为 input_image 设置动态形状配置
            if 'image' in input_name.lower():
                # 图像输入格式为 [H, W, C]
                min_shape = [512, 512, 4]
                opt_shape = [512, 512, 4]
                max_shape = [512, 512, 4]
                
                TRT_LOGGER.log(TRT_LOGGER.INFO, 
                    f'Setting shape for {input_name}: {min_shape}')
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            
            else:
                # 对于 pose 和其他输入，固定为 batch_size=1
                # 这避免了条件层（If层）中的秩不匹配问题
                min_shape = [1 if dim == -1 else dim for dim in input_shape_list]
                opt_shape = [1 if dim == -1 else dim for dim in input_shape_list]
                max_shape = [1 if dim == -1 else dim for dim in input_shape_list]
                
                TRT_LOGGER.log(TRT_LOGGER.INFO,
                    f'Setting shape for {input_name}: shape={min_shape}')
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        
        config.add_optimization_profile(profile)
    
    # 设置内存限制
    def GiB(val):
        return val * 1 << 30
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(2))  # 2G for complex models
    
    # 设置精度
    if precision == 'fp32':
        config.set_flag(trt.BuilderFlag.TF32)
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        raise ValueError('precision must be one of fp32 or fp16')
    
    # Build engine
    TRT_LOGGER.log(TRT_LOGGER.INFO, f'Building an engine from file {onnx_file_path}; this may take a while...')
    serialized_engine = builder.build_serialized_network(network, config)
    TRT_LOGGER.log(TRT_LOGGER.INFO, 'Completed creating Engine')
    
    return serialized_engine


def convert_student_onnx_to_trt(model_name: str, precision: str = 'fp32'):
    """
    转换学生模型的 ONNX 文件为 TensorRT 引擎
    
    Args:
        model_name: 模型名称（例如：kanori_tha4）
        precision: 精度类型，'fp32' 或 'fp16'
    """
    # 基础目录
    base_dir = Path('data/models/custom_tha4_models') / model_name
    onnx_dir = base_dir 
    
    # 检查 ONNX 文件目录是否存在
    if not onnx_dir.exists():
        print(f"错误: ONNX 模型目录不存在: {onnx_dir}")
        print(f"请先运行: python onnx_export_tha4_student.py {model_name} --{precision}")
        return False
    
    # 学生模型只需要这两个文件
    target_names = ['face_morpher.onnx', 'body_morpher.onnx']
    
    # 获取 ONNX 文件
    onnx_files = [onnx_dir / name for name in target_names if (onnx_dir / name).exists()]
    
    if len(onnx_files) != 2:
        print(f"错误: 在 {onnx_dir} 中未找到所有必需的 ONNX 模型文件")
        print(f"需要的文件: {target_names}")
        print(f"找到的文件: {[f.name for f in onnx_files]}")
        return False
    
    print(f"\n{'='*70}")
    print(f"学生模型 TensorRT 转换工具")
    print(f"{'='*70}")
    print(f"模型名称: {model_name}")
    print(f"精度类型: {precision.upper()}")
    print(f"找到 {len(onnx_files)} 个 ONNX 模型文件")
    print(f"{'='*70}\n")
    
    # 设置输出目录（与 ONNX 文件同目录）
    output_dir = onnx_dir
    success_count = 0
    
    for onnx_file in onnx_files:
        trt_filename = onnx_file.stem + '.trt'
        trt_path = output_dir / trt_filename
        
        # 检查是否已存在转换好的 TRT 文件
        if trt_path.exists():
            user_input = input(f"⊙ TRT文件已存在: {trt_filename}\n  是否覆盖? (y/N): ")
            if user_input.lower() != 'y':
                print(f"  跳过: {onnx_file.name}\n")
                success_count += 1
                continue
            else:
                print(f"  将覆盖现有文件\n")
        
        print(f"{'─'*70}")
        print(f"正在转换: {onnx_file.name}")
        print(f"{'─'*70}")
        
        try:
            # 构建 TensorRT 引擎
            print(f"  → 加载 ONNX 文件...")
            
            # 使用支持动态输入的构建函数（两个模型都可能有动态输入）
            print(f"  → 构建 TensorRT 引擎（支持动态输入）...")
            engine = build_engine_with_dynamic_shape(str(onnx_file), precision)
            
            if engine is None:
                raise RuntimeError("引擎构建返回 None")
            
            print(f"  → 保存 TensorRT 引擎...")
            save_engine(engine, str(trt_path))
            
            # 验证文件是否成功保存
            if trt_path.exists():
                file_size = trt_path.stat().st_size / (1024 * 1024)  # MB
                print(f"  ✓ 成功转换: {onnx_file.name} -> {trt_filename}")
                print(f"    文件大小: {file_size:.2f} MB")
                success_count += 1
            else:
                raise RuntimeError("TRT 文件保存失败")
                
        except Exception as e:
            print(f"  ✗ 转换失败: {onnx_file.name}")
            print(f"    错误信息: {str(e)}")
        
        print()
    
    # 输出总结
    print(f"{'='*70}")
    if success_count == len(onnx_files):
        print(f"✓ 全部转换完成！({success_count}/{len(onnx_files)})")
        print(f"\nTensorRT 模型已保存到:")
        print(f"  {output_dir}")
        print(f"\n生成的文件:")
        for name in target_names:
            trt_name = name.replace('.onnx', '.trt')
            trt_file = output_dir / trt_name
            if trt_file.exists():
                file_size = trt_file.stat().st_size / (1024 * 1024)
                print(f"  - {trt_name} ({file_size:.2f} MB)")
        return True
    else:
        print(f"⚠ 部分转换失败 ({success_count}/{len(onnx_files)})")
        return False


def main():
    """主函数：处理命令行参数并执行转换"""
    if len(sys.argv) < 2:
        print("用法: python tha4_student_convert_onnx_to_trt.py <模型名称> [精度]")
        print("\n参数:")
        print("  模型名称: 自定义模型名称（例如：kanori_tha4）")
        print("  精度:     fp32 或 fp16（默认：fp32）")
        print("\n示例:")
        print("  python tha4_student_convert_onnx_to_trt.py kanori_tha4")
        print("  python tha4_student_convert_onnx_to_trt.py kanori_tha4 fp32")
        print("  python tha4_student_convert_onnx_to_trt.py kanori_tha4 fp16")
        print("\n注意:")
        print("  - 请确保已经使用 onnx_export_tha4_student.py 导出了 ONNX 模型")
        print("  - TensorRT 转换需要 CUDA 支持")
        print("  - 转换过程可能需要几分钟时间")
        sys.exit(1)
    
    model_name = sys.argv[1]
    precision = sys.argv[2].lower() if len(sys.argv) > 2 else 'fp32'
    
    # 验证精度参数
    if precision not in ['fp32', 'fp16']:
        print(f"错误: 不支持的精度类型 '{precision}'")
        print("支持的精度: fp32, fp16")
        sys.exit(1)
    
    # 检查 TensorRT 是否可用
    try:
        import tensorrt as trt
        print(f"TensorRT 版本: {trt.__version__}")
    except ImportError:
        print("错误: 未安装 TensorRT")
        print("请安装 TensorRT: pip install tensorrt")
        sys.exit(1)
    
    # 检查 CUDA 是否可用
    try:
        import pycuda.driver as cuda
        cuda.init()
        print(f"CUDA 设备数量: {cuda.Device.count()}")
        if cuda.Device.count() > 0:
            device = cuda.Device(0)
            print(f"GPU: {device.name()}")
    except Exception as e:
        print(f"警告: CUDA 初始化失败 - {e}")
        print("TensorRT 转换需要 CUDA 支持")
        sys.exit(1)
    
    print()
    
    # 执行转换
    success = convert_student_onnx_to_trt(model_name, precision)
    
    if success:
        print(f"\n{'='*70}")
        print("转换成功！可以在启动器中选择使用 TensorRT 模式")
        print(f"{'='*70}")
        sys.exit(0)
    else:
        print(f"\n{'='*70}")
        print("转换失败，请检查错误信息")
        print(f"{'='*70}")
        sys.exit(1)


if __name__ == '__main__':
    main()
