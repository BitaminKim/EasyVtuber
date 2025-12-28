import argparse
import re


def convert_to_byte(size):
    result = re.search('(\d+\.?\d*)(b|kb|mb|gb|tb)', size.lower())
    if (result and result.groups()):
        unit = result.groups()[1]
        amount = float(result.groups()[0])
        index = ['b', 'kb', 'mb', 'gb', 'tb'].index(unit)
        return amount * pow(1024, index)
    raise ValueError("Invalid size provided, value is " + size)


parser = argparse.ArgumentParser()
parser.add_argument('--character', type=str, default='lambda_00')

parser.add_argument('--debug_input', action='store_true')
parser.add_argument('--cam_input', action='store_true')
parser.add_argument('--mouse_input', type=str)
parser.add_argument('--ifm_input', type=str)
parser.add_argument('--osf_input', type=str)

parser.add_argument('--output_virtual_cam', action='store_true')
parser.add_argument('--output_spout2', action='store_true')
parser.add_argument('--output_debug', action='store_true')

parser.add_argument('--alpha_split', action='store_true')
parser.add_argument('--bongo', action='store_true')
parser.add_argument('--extend_movement', action='store_true')

parser.add_argument('--filter_min_cutoff', type=float, default=1.0)
parser.add_argument('--filter_beta', type=float, default=0.0)

parser.add_argument('--simplify', type=int, default=1)

parser.add_argument('--use_tensorrt', action='store_true')
parser.add_argument('--model_version', type=str, default='v3')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--model_seperable', action='store_true')
parser.add_argument('--model_half', action='store_true')
parser.add_argument('--gpu_cache', type=str, default='512mb')
parser.add_argument('--eyebrow', action='store_true')

parser.add_argument('--use_interpolation', action='store_true')
parser.add_argument('--interpolation_scale', type=int, default=1)
parser.add_argument('--interpolation_half', action='store_true')

parser.add_argument('--use_sr', action='store_true')
parser.add_argument('--sr_x4', action='store_true')
parser.add_argument('--sr_half', action='store_true')
parser.add_argument('--anime4k', action='store_true')

parser.add_argument('--cache', type=str, default='256mb')

parser.add_argument('--frame_rate_limit', type=int, default=30)

parser.add_argument('--alpha_clean', action='store_true')

args = parser.parse_args()

if args.cache is not None:
    args.max_ram_cache_len = convert_to_byte(args.cache) / pow(1024, 3)  # In gigabytes
else:
    args.max_ram_cache_len = 0
if args.gpu_cache is not None:
    args.max_gpu_cache_len = convert_to_byte(args.gpu_cache) / pow(1024, 3)  # In gigabytes
else:
    args.max_gpu_cache_len = 0

if args.simplify == 0:
    args.max_ram_cache_len = 0 # Disable cacher if simplify is Off

if not args.output_virtual_cam and not args.output_spout2: 
    args.output_debug = True # Default to debug output

if not args.cam_input and args.mouse_input is None \
    and args.ifm_input is None and args.osf_input is None:
    args.debug_input = True  # Default to debug input

if args.use_sr:
    args.anime4k = False  # Disable anime4k if SR is enabled
if args.anime4k:
    args.use_sr = False  # Disable SR if anime4k is enabled

args.model_output_size = 1024 if args.use_sr or args.anime4k else 512

print("Arguments:", args)