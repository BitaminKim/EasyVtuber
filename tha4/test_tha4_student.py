#!/usr/bin/env python3
"""Test script for THA4 Student model inference
Tests both TensorRT and ONNX Runtime backends
"""
import sys
import os
import numpy as np

# Add ezvtuber-rt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ezvtuber-rt'))


def test_tensorrt_backend():
    """Test TensorRT backend"""
    print("\n=== Testing TensorRT Backend ===")
    try:
        from ezvtb_rt import CoreTRT
        
        # Initialize with student model (kanori_tha4)
        core = CoreTRT(
            tha_model_version='v4_student',
            tha_model_name='kanori_tha4'
        )
        
        # Create dummy image (512x512 RGBA)
        test_image = np.random.randint(0, 256, (512, 512, 4), dtype=np.uint8)
        
        # Create dummy pose (1x45)
        test_pose = np.random.randn(1, 45).astype(np.float32)
        
        # Run inference
        core.setImage(test_image)
        result = core.inference(test_pose)
        
        print("PASS: TensorRT Backend: SUCCESS")
        print(f"  Input image shape: {test_image.shape}")
        print(f"  Input pose shape: {test_pose.shape}")
        out_shape = (result[0].shape if isinstance(result, list)
                     else result.shape)
        print(f"  Output shape: {out_shape}")
        return True
        
    except Exception as e:
        print("FAIL: TensorRT Backend: FAILED")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_onnxruntime_backend():
    """Test ONNX Runtime backend"""
    print("\n=== Testing ONNX Runtime Backend ===")
    try:
        from ezvtb_rt import CoreORT
        
        # Initialize with student model (kanori_tha4)
        core = CoreORT(
            tha_model_version='v4_student',
            tha_model_name='kanori_tha4'
        )
        
        # Create dummy image (512x512 RGBA)
        test_image = np.random.randint(0, 256, (512, 512, 4), dtype=np.uint8)
        
        # Create dummy pose (1x45)
        test_pose = np.random.randn(1, 45).astype(np.float32)
        
        # Run inference
        core.setImage(test_image)
        result = core.inference(test_pose)
        
        print("PASS: ONNX Runtime Backend: SUCCESS")
        print(f"  Input image shape: {test_image.shape}")
        print(f"  Input pose shape: {test_pose.shape}")
        out_shape = (result[0].shape if isinstance(result, list)
                     else result.shape)
        print(f"  Output shape: {out_shape}")
        return True
        
    except Exception as e:
        print("FAIL: ONNX Runtime Backend: FAILED")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_models():
    """Test direct model classes"""
    print("\n=== Testing Direct Model Classes ===")
    
    success = True
    
    # Test THA4Student (TensorRT)
    try:
        from ezvtb_rt.tha4_student import THA4Student
        model_dir = os.path.join(
            os.path.dirname(__file__),
            'data', 'models', 'custom_tha4_models', 'kanori_tha4'
        )
        
        if os.path.exists(model_dir):
            model = THA4Student(model_dir)
            test_image = np.random.randint(
                0, 256, (512, 512, 4), dtype=np.uint8
            )
            test_pose = np.random.randn(1, 45).astype(np.float32)
            
            model.setImage(test_image)
            result = model.inference(test_pose)
            
            print("PASS: THA4Student (TensorRT): SUCCESS")
            out_shape = result[0].shape if isinstance(result, list) else result.shape
            print(f"  Output shape: {out_shape}")
        else:
            print(f"SKIP: THA4Student: Model not found at {model_dir}")
            
    except Exception as e:
        print("FAIL: THA4Student (TensorRT): FAILED")
        print(f"  Error: {str(e)}")
        success = False
    
    # Test THA4StudentORT (ONNX Runtime)
    try:
        from ezvtb_rt.tha4_student_ort import THA4StudentORT
        model_dir = os.path.join(
            os.path.dirname(__file__),
            'data', 'models', 'custom_tha4_models', 'kanori_tha4'
        )
        
        if os.path.exists(model_dir):
            model = THA4StudentORT(model_dir, device_id=0)
            test_image = np.random.randint(
                0, 256, (512, 512, 4), dtype=np.uint8
            )
            test_pose = np.random.randn(1, 45).astype(np.float32)
            
            model.setImage(test_image)
            result = model.inference(test_pose)
            
            print("PASS: THA4StudentORT (ONNX Runtime): SUCCESS")
            out_shape = result[0].shape if isinstance(result, list) else result.shape
            print(f"  Output shape: {out_shape}")
        else:
            print(f"SKIP: THA4StudentORT: Model not found at {model_dir}")
            
    except Exception as e:
        print("FAIL: THA4StudentORT (ONNX Runtime): FAILED")
        print(f"  Error: {str(e)}")
        success = False
    
    return success


def main():
    print("THA4 Student Model Integration Tests")
    print("=" * 50)
    
    results = []
    
    # Test direct models first
    print("\n--- Direct Model Classes ---")
    results.append(("Direct Models", test_direct_models()))
    
    # Test backends
    print("\n--- Framework Backends ---")
    results.append(("TensorRT Backend", test_tensorrt_backend()))
    results.append(("ONNX Runtime Backend", test_onnxruntime_backend()))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<30} [{status}]")
    
    all_passed = all(r[1] for r in results)
    overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print("\nOverall: " + overall)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
