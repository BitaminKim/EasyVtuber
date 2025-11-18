import numpy as np
import onnxruntime as ort
import onnx
from onnxconverter_common import float16
import os
from onnxsim import simplify

names = ['body_morpher.onnx', 'combiner.onnx', 'decomposer.onnx', 'morpher.onnx', 'upscaler.onnx']

for name in names:
    model = onnx.load(os.path.join('data', 'models','tha4', 'fp32', name))
    model_fp16 = float16.convert_float_to_float16(model, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False, disable_shape_infer=False, op_block_list=None, node_block_list=None)

    for node in model_fp16.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    if attr.i == 1: # 1 means FLOAT (float32)
                        print(f"Changing {name} Cast node '{node.name}' ({node.output[0]}) from float32 -> float16")
                        attr.i = 10 # 10 means FLOAT16
    
    # Simplify first
    model_fp16, check = simplify(model_fp16)
    assert check, "Simplified ONNX model could not be validated"
    
    # Fix input/output types to ensure all are FP16 (after simplify)
    for input_tensor in model_fp16.graph.input:
        if input_tensor.type.tensor_type.elem_type == 1:  # FLOAT32
            print(f"Converting {name} input '{input_tensor.name}' from FP32 to FP16")
            input_tensor.type.tensor_type.elem_type = 10  # FLOAT16
    
    for output_tensor in model_fp16.graph.output:
        if output_tensor.type.tensor_type.elem_type == 1:  # FLOAT32
            print(f"Converting {name} output '{output_tensor.name}' from FP32 to FP16")
            output_tensor.type.tensor_type.elem_type = 10  # FLOAT16
                        
    onnx.save(model_fp16, os.path.join('data', 'models', 'tha4', 'fp16', name))