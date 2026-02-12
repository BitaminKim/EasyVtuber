from ..args import args
import numpy as np
import math

# Delay import to avoid loading torch during module initialization
ifm_converter = None

def _get_ifm_converter():
    global ifm_converter
    if ifm_converter is None:
        import tha2.poser.modes.mode_20_wx
        ifm_converter = tha2.poser.modes.mode_20_wx.IFacialMocapPoseConverter20()
    return ifm_converter

def pose_simplify(model_input):
    ifm_converter = _get_ifm_converter()
    simplify_arr = [1000] * ifm_converter.pose_size
    if args.simplify >= 1:
        simplify_arr = [200] * ifm_converter.pose_size
        simplify_arr[ifm_converter.eye_wink_left_index] = 50
        simplify_arr[ifm_converter.eye_wink_right_index] = 50
        simplify_arr[ifm_converter.eye_happy_wink_left_index] = 50
        simplify_arr[ifm_converter.eye_happy_wink_right_index] = 50
        simplify_arr[ifm_converter.eye_surprised_left_index] = 30
        simplify_arr[ifm_converter.eye_surprised_right_index] = 30
        simplify_arr[ifm_converter.iris_rotation_x_index] = 25
        simplify_arr[ifm_converter.iris_rotation_y_index] = 25
        simplify_arr[ifm_converter.eye_raised_lower_eyelid_left_index] = 10
        simplify_arr[ifm_converter.eye_raised_lower_eyelid_right_index] = 10
        simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 5
        simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 5
        simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 5
        simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 5
    if args.simplify >= 2:
        simplify_arr[ifm_converter.head_x_index] = 100
        simplify_arr[ifm_converter.head_y_index] = 100
        simplify_arr[ifm_converter.eye_surprised_left_index] = 10
        simplify_arr[ifm_converter.eye_surprised_right_index] = 10
        model_input[ifm_converter.eye_wink_left_index] += model_input[
            ifm_converter.eye_happy_wink_left_index]
        model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                    ifm_converter.eye_wink_left_index] / 2
        model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                ifm_converter.eye_wink_left_index] / 2
        model_input[ifm_converter.eye_wink_right_index] += model_input[
            ifm_converter.eye_happy_wink_right_index]
        model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                    ifm_converter.eye_wink_right_index] / 2
        model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                ifm_converter.eye_wink_right_index] / 2

        uosum = model_input[ifm_converter.mouth_uuu_index] + \
                model_input[ifm_converter.mouth_ooo_index]
        model_input[ifm_converter.mouth_ooo_index] = uosum
        model_input[ifm_converter.mouth_uuu_index] = 0
        is_open = (model_input[ifm_converter.mouth_aaa_index] + model_input[
            ifm_converter.mouth_iii_index] + uosum) > 0
        model_input[ifm_converter.mouth_lowered_corner_left_index] = 0
        model_input[ifm_converter.mouth_lowered_corner_right_index] = 0
        model_input[ifm_converter.mouth_raised_corner_left_index] = 0.5 if is_open else 0
        model_input[ifm_converter.mouth_raised_corner_right_index] = 0.5 if is_open else 0
        simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 0
        simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 0
        simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 0
        simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 0
    if args.simplify >= 3:
        simplify_arr[ifm_converter.iris_rotation_x_index] = 20
        simplify_arr[ifm_converter.iris_rotation_y_index] = 20
        simplify_arr[ifm_converter.eye_wink_left_index] = 32
        simplify_arr[ifm_converter.eye_wink_right_index] = 32
        simplify_arr[ifm_converter.eye_happy_wink_left_index] = 32
        simplify_arr[ifm_converter.eye_happy_wink_right_index] = 32
    if args.simplify >= 4:
        simplify_arr[ifm_converter.head_x_index] = 50
        simplify_arr[ifm_converter.head_y_index] = 50
        simplify_arr[ifm_converter.neck_z_index] = 100
        model_input[ifm_converter.eye_raised_lower_eyelid_left_index] = 0
        model_input[ifm_converter.eye_raised_lower_eyelid_right_index] = 0
        simplify_arr[ifm_converter.iris_rotation_x_index] = 10
        simplify_arr[ifm_converter.iris_rotation_y_index] = 10
        simplify_arr[ifm_converter.eye_wink_left_index] = 24
        simplify_arr[ifm_converter.eye_wink_right_index] = 24
        simplify_arr[ifm_converter.eye_happy_wink_left_index] = 24
        simplify_arr[ifm_converter.eye_happy_wink_right_index] = 24
        simplify_arr[ifm_converter.eye_surprised_left_index] = 8
        simplify_arr[ifm_converter.eye_surprised_right_index] = 8
        model_input[ifm_converter.eye_wink_left_index] += model_input[
            ifm_converter.eye_wink_right_index]
        model_input[ifm_converter.eye_wink_right_index] = model_input[
                                                                ifm_converter.eye_wink_left_index] / 2
        model_input[ifm_converter.eye_wink_left_index] = model_input[
                                                                ifm_converter.eye_wink_left_index] / 2

        model_input[ifm_converter.eye_surprised_left_index] += model_input[
            ifm_converter.eye_surprised_right_index]
        model_input[ifm_converter.eye_surprised_right_index] = model_input[
                                                                    ifm_converter.eye_surprised_left_index] / 2
        model_input[ifm_converter.eye_surprised_left_index] = model_input[
                                                                    ifm_converter.eye_surprised_left_index] / 2

        model_input[ifm_converter.eye_happy_wink_left_index] += model_input[
            ifm_converter.eye_happy_wink_right_index]
        model_input[ifm_converter.eye_happy_wink_right_index] = model_input[
                                                                    ifm_converter.eye_happy_wink_left_index] / 2
        model_input[ifm_converter.eye_happy_wink_left_index] = model_input[
                                                                    ifm_converter.eye_happy_wink_left_index] / 2
        model_input[ifm_converter.mouth_aaa_index] = min(
            model_input[ifm_converter.mouth_aaa_index] +
            model_input[ifm_converter.mouth_ooo_index] / 2 +
            model_input[ifm_converter.mouth_iii_index] / 2 +
            model_input[ifm_converter.mouth_uuu_index] / 2, 1
        )
        model_input[ifm_converter.mouth_ooo_index] = 0
        model_input[ifm_converter.mouth_iii_index] = 0
        model_input[ifm_converter.mouth_uuu_index] = 0
    for i in range(4, args.simplify):
        simplify_arr = [max(math.ceil(x * 0.8), 5) for x in simplify_arr]
    for i in range(0, len(simplify_arr)):
        if simplify_arr[i] > 0:
            model_input[i] = round(model_input[i] * simplify_arr[i]) / simplify_arr[i]

    input_pose = np.zeros((1, 45), dtype=np.float32)
    if args.eyebrow:
        for i in range(12):
            input_pose[0, i] = model_input[i]
    for i in range(27):
        input_pose[0, i + 12] = model_input[i + 12]
    for i in range(6):
        input_pose[0, i + 12 + 27] = model_input[i + 27 + 12]
    return input_pose